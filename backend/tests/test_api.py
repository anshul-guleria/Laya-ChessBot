"""HTTP behaviour.

Every test here runs against a `FakeAgent` and a throwaway SQLite file, so the 842 MB
checkpoint is never loaded: `app.main.load_model` finds the fake already installed in
`app.state` and returns without reading the model directory.
`test_the_suite_never_imports_the_model` guards that.

The assertions describe the contract the frontend depends on -- status codes, error
shapes, and the invariant that the stored move list is the only source of truth for a
position. They deliberately say nothing about how strong the bot is: its moves are
scripted.
"""

from __future__ import annotations

import sys

import chess
import pytest
from fastapi.testclient import TestClient

from app import state
from app.config import settings
from app.routers import games as games_router
from chessbot.engine import board_from_moves, legal_uci_moves

STAGE_FIELDS = {
    "stage",
    "question_id",
    "state",
    "instructions",
    "options",
    "choice",
    "probabilities",
    "latency_ms",
    "skipped_reason",
}

GAME_FIELDS = {
    "id",
    "fen",
    "moves",
    "turn",
    "human_side",
    "status",
    "result",
    "termination",
    "is_check",
    "legal_moves",
    "last_move",
    "bot_trace",
}


# --- Creating a game -------------------------------------------------------


def test_creating_a_game_answers_201_with_the_full_state_shape(client):
    response = client.post("/api/games")

    assert response.status_code == 201
    assert set(response.json()) == GAME_FIELDS


def test_creating_a_game_randomises_the_human_side(client):
    """A coin flip stuck on one colour would mean half the players never play it."""
    sides = [client.post("/api/games").json()["human_side"] for _ in range(16)]

    assert set(sides) <= {"white", "black"}
    assert set(sides) == {"white", "black"}, f"the flip produced only {set(sides)}"


def test_a_new_game_starts_from_the_initial_position(client):
    game = _game_with_human_side(client, "white")

    assert game["human_side"] == "white"
    assert game["fen"] == chess.STARTING_FEN
    assert game["moves"] == []
    assert game["turn"] == "white"
    assert game["status"] == "active"
    assert game["result"] is None
    assert game["termination"] is None
    assert game["is_check"] is False
    assert game["last_move"] is None
    assert game["bot_trace"] is None
    assert game["legal_moves"] == legal_uci_moves(chess.Board())


def test_the_bot_opens_when_the_human_is_black(client):
    """The human must never be asked to move first."""
    game = _game_with_human_side(client, "black")

    assert len(game["moves"]) == 1, "the bot must have opened before the human's turn"
    assert game["moves"][0] in legal_uci_moves(chess.Board())
    assert game["turn"] == "black", "the bot opened, so the human's turn is next"
    assert game["last_move"] == game["moves"][0]
    assert game["status"] == "active"
    assert game["is_check"] is False
    assert game["fen"] == board_from_moves(game["moves"]).fen()
    assert game["legal_moves"] == legal_uci_moves(board_from_moves(game["moves"]))

    assert game["bot_trace"] is not None, "an opening the player cannot see explained is a bug"
    assert game["bot_trace"]["move"] == game["moves"][0]
    assert game["bot_trace"]["stages"]
    assert all(set(stage) == STAGE_FIELDS for stage in game["bot_trace"]["stages"])


# --- Playing a move --------------------------------------------------------


def test_playing_a_move_returns_both_plies_and_the_bot_trace(client):
    game = _game_with_human_side(client, "white")

    response = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e4"})

    assert response.status_code == 200
    body = response.json()

    assert body["moves"][0] == "e2e4"
    assert len(body["moves"]) == 2, "the human's ply and the bot's reply land together"
    assert body["last_move"] == body["moves"][1]
    assert body["turn"] == "white"
    assert body["human_side"] == "white"
    assert body["status"] == "active"
    assert body["fen"] == board_from_moves(body["moves"]).fen()
    assert body["legal_moves"] == legal_uci_moves(board_from_moves(body["moves"]))

    trace = body["bot_trace"]
    assert trace is not None
    assert trace["move"] == body["moves"][1], "the trace must explain the stored move"
    assert trace["stages"], "the trace must explain the reply"
    assert all(set(stage) == STAGE_FIELDS for stage in trace["stages"])
    assert {stage["stage"] for stage in trace["stages"]} == {"piece", "destination"}
    assert all(stage["state"] for stage in trace["stages"])
    assert trace["total_latency_ms"] > 0


def test_an_illegal_move_is_rejected_with_400(client):
    game = _game_with_human_side(client, "white")

    response = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e5"})

    assert response.status_code == 400
    assert "not legal" in response.json()["detail"]
    assert client.get(f"/api/games/{game['id']}").json()["moves"] == []


def test_an_unparseable_move_is_rejected_with_400(client):
    game = _game_with_human_side(client, "white")

    response = client.post(f"/api/games/{game['id']}/move", json={"uci": "zzzz"})

    assert response.status_code == 400
    assert "cannot parse" in response.json()["detail"]
    assert client.get(f"/api/games/{game['id']}").json()["moves"] == []


def test_playing_out_of_turn_is_409(client, seed_game):
    """It is black to move but the human is white, so this is the bot's turn.

    The API can never leave a game in that state -- both plies land in one request -- so
    the row is written directly.
    """
    game = seed_game(human_side="white", status="active", moves=["e2e4"])

    response = client.post(f"/api/games/{game.id}/move", json={"uci": "d2d4"})

    assert response.status_code == 409
    assert "turn" in response.json()["detail"]
    assert client.get(f"/api/games/{game.id}").json()["moves"] == ["e2e4"]


def test_an_unknown_game_is_404(client):
    assert client.get("/api/games/9999").status_code == 404
    assert client.post("/api/games/9999/move", json={"uci": "e2e4"}).status_code == 404
    assert client.post("/api/games/9999/undo").status_code == 404
    assert client.get("/api/games/9999/pgn").status_code == 404


def test_a_finished_game_refuses_further_moves(client, agent):
    """Fool's mate, scripted: the human opens badly and the bot's replies end it."""
    game = _game_with_human_side(client, "white")
    agent.script = _fools_mate_script()

    for uci in ["f2f3", "g2g4"]:
        response = client.post(f"/api/games/{game['id']}/move", json={"uci": uci})
        assert response.status_code == 200, response.text
    body = response.json()

    assert body["moves"] == ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert body["status"] == "black_won"
    assert body["termination"] == "checkmate"
    assert body["result"] == "0-1"
    assert body["is_check"] is True
    assert body["last_move"] == "d8h4"

    refused = client.post(f"/api/games/{game['id']}/move", json={"uci": "e1f1"})
    assert refused.status_code == 409
    assert "over" in refused.json()["detail"]


# --- Undo ------------------------------------------------------------------


def test_undo_takes_back_the_human_move_and_the_reply(client):
    game = _game_with_human_side(client, "white")
    played = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e4"}).json()
    assert len(played["moves"]) == 2

    response = client.post(f"/api/games/{game['id']}/undo")
    assert response.status_code == 200
    undone = response.json()

    assert undone["moves"] == [], "both plies go, or the player is left on the bot's turn"
    assert undone["fen"] == chess.STARTING_FEN
    assert undone["turn"] == "white"
    assert undone["last_move"] is None
    assert undone["status"] == "active"
    assert undone["legal_moves"] == legal_uci_moves(chess.Board())


def test_undo_clears_the_stored_bot_trace(client):
    game = _game_with_human_side(client, "white")
    played = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e4"}).json()
    assert played["bot_trace"] is not None
    assert client.get(f"/api/games/{game['id']}").json()["bot_trace"] is not None

    undone = client.post(f"/api/games/{game['id']}/undo").json()

    assert undone["bot_trace"] is None
    reread = client.get(f"/api/games/{game['id']}").json()
    assert reread["bot_trace"] is None, "the trace must be cleared on the row, not just omitted"


def test_undo_on_a_game_with_nothing_to_undo_is_409(client):
    game = _game_with_human_side(client, "white")

    response = client.post(f"/api/games/{game['id']}/undo")

    assert response.status_code == 409
    assert "nothing to undo" in response.json()["detail"]


def test_undo_restores_a_finished_game_to_active(client, agent):
    """The human's own last move is `2.g4`, so its mate and the reply both come off."""
    game = _game_with_human_side(client, "white")
    agent.script = _fools_mate_script()
    for uci in ["f2f3", "g2g4"]:
        assert client.post(f"/api/games/{game['id']}/move", json={"uci": uci}).status_code == 200
    assert client.get(f"/api/games/{game['id']}").json()["status"] == "black_won"

    undone = client.post(f"/api/games/{game['id']}/undo").json()

    assert undone["moves"] == ["f2f3", "e7e5"]
    assert undone["status"] == "active"
    assert undone["result"] is None
    assert undone["termination"] is None
    assert undone["turn"] == "white"
    assert undone["is_check"] is False
    assert client.get(f"/api/games/{game['id']}").json()["status"] == "active"

    resumed = client.post(f"/api/games/{game['id']}/move", json={"uci": "g2g4"})
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["moves"][:3] == ["f2f3", "e7e5", "g2g4"]
    assert resumed.json()["status"] == "active", "2.g4 must not mate this time"


# --- Reading games ---------------------------------------------------------


@pytest.mark.parametrize("human_side", ["white", "black"])
def test_pgn_export_names_both_players_and_contains_the_moves(client, human_side):
    game = _game_with_human_side(client, human_side)
    if human_side == "white":
        played = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e4"})
        assert played.status_code == 200, played.text
        game = played.json()
    assert game["moves"], "there must be something to export"

    response = client.get(f"/api/games/{game['id']}/pgn")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    white, black = (
        ('[White "Human"]', '[Black "Laya"]')
        if human_side == "white"
        else ('[White "Laya"]', '[Black "Human"]')
    )
    assert white in response.text
    assert black in response.text
    assert '[Result "*"]' in response.text
    for san in _san_line(game["moves"]):
        assert san in response.text


def test_listing_games_is_newest_first_with_move_counts(client):
    first = client.post("/api/games").json()
    played = _game_with_human_side(client, "white")
    second = client.post("/api/games").json()
    assert client.post(f"/api/games/{played['id']}/move", json={"uci": "e2e4"}).status_code == 200

    listed = client.get("/api/games").json()

    ids = [row["id"] for row in listed]
    assert ids == sorted(ids, reverse=True)
    assert ids.index(second["id"]) < ids.index(played["id"]) < ids.index(first["id"])
    rows = {row["id"]: row for row in listed}
    assert set(rows[first["id"]]) == {
        "id",
        "human_side",
        "status",
        "move_count",
        "created_at",
    }
    assert rows[played["id"]]["move_count"] == 2
    assert rows[first["id"]]["move_count"] == (1 if first["human_side"] == "black" else 0)
    assert rows[second["id"]]["move_count"] == (1 if second["human_side"] == "black" else 0)
    assert rows[first["id"]]["created_at"]


# --- Degraded service ------------------------------------------------------


def test_health_is_ok_when_a_model_is_installed(client, agent):
    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == settings.laya_device
    assert body["model_dir"] == str(settings.laya_model_dir)
    assert state.get_agent() is agent, "the fake must be the model answering requests"


def test_health_reports_degraded_when_the_process_agent_is_cleared(client):
    state.clear_agent()

    body = client.get("/api/health").json()

    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["device"] == settings.laya_device


def test_a_move_without_a_model_is_503(client):
    game = _game_with_human_side(client, "white")
    state.clear_agent()

    response = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e4"})

    assert response.status_code == 503
    assert "model" in response.json()["detail"]
    assert client.get(f"/api/games/{game['id']}").json()["moves"] == []


def test_creating_a_game_without_a_model_is_503(client, monkeypatch):
    """The dependency answers before the handler runs, so no row is ever written."""
    monkeypatch.setattr(games_router, "SIDES", ("black",))
    state.clear_agent()

    response = client.post("/api/games")

    assert response.status_code == 503
    assert "model" in response.json()["detail"]
    assert client.get("/api/games").json() == []


def test_creating_a_game_the_bot_cannot_open_saves_nothing(client, monkeypatch):
    """A model that answers and then fails must not leave a game with nobody to play."""
    monkeypatch.setattr(games_router, "SIDES", ("black",))

    def explode(*args, **kwargs):
        raise RuntimeError("torch exploded")

    monkeypatch.setattr(games_router, "decide_move", explode)
    response = client.post("/api/games")

    assert response.status_code == 503
    assert "could not open" in response.json()["detail"]
    assert client.get("/api/games").json() == [], "no orphan row may be left behind"


def test_a_bot_failure_is_503_and_leaves_the_history_untouched(client, monkeypatch):
    """A player who loses the model's answer has to be able to replay the same move."""
    game = _game_with_human_side(client, "white")
    real_decide_move = games_router.decide_move

    def explode(*args, **kwargs):
        raise RuntimeError("torch exploded")

    monkeypatch.setattr(games_router, "decide_move", explode)
    failed = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e4"})

    assert failed.status_code == 503
    assert "nothing was saved" in failed.json()["detail"]
    after = client.get(f"/api/games/{game['id']}").json()
    assert after["moves"] == [], "the human's ply must roll back with the bot's failure"
    assert after["last_move"] is None
    assert after["bot_trace"] is None

    monkeypatch.setattr(games_router, "decide_move", real_decide_move)
    retried = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e4"})

    assert retried.status_code == 200, retried.text
    assert retried.json()["moves"][0] == "e2e4"
    assert len(retried.json()["moves"]) == 2


def test_the_suite_never_imports_the_model():
    """No test may load the 842 MB checkpoint, directly or by pulling in torch."""
    assert "laya" not in sys.modules
    assert "torch" not in sys.modules


# --- Helpers ---------------------------------------------------------------


def _game_with_human_side(client: TestClient, side: str) -> dict:
    """Create games until the coin flip lands on `side`, then return that one.

    Args:
        client: The `TestClient` fixture.
        side: `"white"` or `"black"`.

    Returns:
        The `GameState` body of the created game.

    Raises:
        AssertionError: If twenty consecutive flips all missed.
    """
    for _ in range(20):
        response = client.post("/api/games")
        assert response.status_code == 201, response.text
        body = response.json()
        if body["human_side"] == side:
            return body
    raise AssertionError(f"the coin flip never produced a {side} game in twenty tries")


def _fools_mate_script() -> dict[str, str | list[str]]:
    """Return a script that walks the bot through black's half of fool's mate.

    The lists are consumed one entry per call and there are exactly two bot moves, so the
    first is `1...e5` -- a pawn, which has no direction stage -- and the second is
    `2...Qh4`, a queen travelling down the d8-h4 diagonal. Fresh every call, because
    `FakeAgent` pops from the lists in place.

    Returns:
        A `FakeAgent.script` mapping.
    """
    return {
        "piece_selection": ["P", "Q"],
        "direction_selection": "diagonal_nw_se",
        "destination_selection": ["e5", "h4"],
    }


def _san_line(moves: list[str]) -> list[str]:
    """Render a UCI move list as SAN, the way PGN prints it.

    Args:
        moves: UCI move strings in order.

    Returns:
        One SAN move per ply, in order.
    """
    board = chess.Board()
    line: list[str] = []
    for uci in moves:
        move = board.parse_uci(uci)
        line.append(board.san(move))
        board.push(move)
    return line
