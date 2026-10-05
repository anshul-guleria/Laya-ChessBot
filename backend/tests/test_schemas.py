"""Response shaping, persistence and the move-list-is-the-truth invariant.

The `games` table stores no FEN. A stored FEN can drift from the move history, and a
player looking at one position while the bot reasons about another is a genuinely nasty
class of bug, so the FEN is derived by replaying `moves` on every read and the status is
recomputed from the position rather than trusted from the row.
"""

from __future__ import annotations

import threading
from datetime import timedelta

import chess
import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import db as db_module
from app import schemas
from app.db import Base, make_engine
from app.models import Game
from chessbot.engine import RESULT_BLACK, STATUS_ACTIVE, STATUS_BLACK_WON, STATUS_DRAW


def test_a_fresh_game_is_the_starting_position():
    game = Game(id=1, human_side="white", status=STATUS_ACTIVE, moves=[])
    state = schemas.build_game_state(game)

    assert state.fen == chess.STARTING_FEN
    assert state.turn == "white"
    assert state.moves == []
    assert state.last_move is None
    assert state.bot_trace is None
    assert state.result is None
    assert state.termination is None
    assert state.is_check is False
    assert state.status == STATUS_ACTIVE
    assert len(state.legal_moves) == 20


def test_the_fen_is_derived_from_the_move_list():
    game = Game(id=1, human_side="white", status=STATUS_ACTIVE, moves=["e2e4", "c7c5", "g1f3"])
    reference = chess.Board()
    for uci in ["e2e4", "c7c5", "g1f3"]:
        reference.push_uci(uci)

    state = schemas.build_game_state(game)

    assert state.fen == reference.fen()
    assert state.turn == "black"
    assert state.last_move == "g1f3"


def test_status_is_recomputed_from_the_position_not_trusted_from_the_row():
    """A stale `status` column must not survive into the response."""
    game = Game(
        id=1,
        human_side="white",
        status=STATUS_ACTIVE,
        moves=["f2f3", "e7e5", "g2g4", "d8h4"],
    )
    state = schemas.build_game_state(game)

    assert state.status == STATUS_BLACK_WON
    assert state.result == RESULT_BLACK
    assert state.termination == "checkmate"
    assert state.is_check is True


def test_result_and_termination_are_recomputed_too():
    """A row carrying a stale result and reason must lose both to the replay."""
    game = Game(
        id=1,
        human_side="black",
        status=STATUS_DRAW,
        result="1-0",
        termination="resignation",
        moves=["f2f3", "e7e5", "g2g4", "d8h4"],
    )
    state = schemas.build_game_state(game)

    assert state.status == STATUS_BLACK_WON
    assert state.result == RESULT_BLACK
    assert state.termination == "checkmate"


def test_check_is_reported():
    """1.e4 d5 2.exd5 Qxd5 3.Nc3 Qxd2+ leaves white in check but not mated."""
    game = Game(
        id=1,
        human_side="black",
        status=STATUS_ACTIVE,
        moves=["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5d2"],
    )

    assert schemas.build_game_state(game).is_check is True


def test_a_stored_trace_is_returned():
    trace = {
        "move": "e7e5",
        "stages": [
            {
                "stage": "piece",
                "question_id": "piece_selection",
                "state": "Chess position. FEN: ... Stage: piece.",
                "instructions": "Choose the best chess piece for the move.",
                "options": {"P": "Pawn"},
                "choice": "P",
                "probabilities": {"P": 1.0},
                "latency_ms": 41.0,
                "skipped_reason": None,
            }
        ],
        "total_latency_ms": 41.0,
        "forced": False,
        "tie_break": None,
    }
    game = Game(id=1, human_side="white", status=STATUS_ACTIVE, moves=["e2e4"], last_trace=trace)
    state = schemas.build_game_state(game)

    assert state.bot_trace is not None
    assert state.bot_trace.move == "e7e5"
    assert state.bot_trace.stages[0].choice == "P"
    assert state.bot_trace.stages[0].probabilities == {"P": 1.0}


def test_an_explicit_trace_overrides_the_stored_one():
    stored = {"move": "e7e5", "stages": [], "total_latency_ms": 8.0}
    game = Game(id=1, human_side="white", status=STATUS_ACTIVE, moves=["e2e4"], last_trace=stored)
    fresh = schemas.BotTrace(move="c7c5", stages=[], total_latency_ms=12.0)

    assert schemas.build_game_state(game, bot_trace=fresh).bot_trace.move == "c7c5"


def test_build_game_state_needs_no_persisted_row():
    """No `created_at`, no `result`, no `termination`: shaping touches none of them."""
    game = Game(id=7, human_side="black", status=STATUS_ACTIVE, moves=["e2e4"])
    assert game.created_at is None

    state = schemas.build_game_state(game)

    assert state.id == 7
    assert state.human_side == "black"
    assert state.turn == "black"
    assert state.result is None


def test_the_state_round_trips_through_json():
    game = Game(id=1, human_side="white", status=STATUS_ACTIVE, moves=["e2e4", "e7e5"])
    state = schemas.build_game_state(game)

    assert schemas.GameState.model_validate_json(state.model_dump_json()).fen == state.fen


def test_move_request_rejects_an_over_long_uci():
    with pytest.raises(ValidationError):
        schemas.MoveRequest(uci="a" * 40)


def test_move_request_accepts_a_promotion():
    assert schemas.MoveRequest(uci="e7e8q").uci == "e7e8q"


def test_there_is_no_stored_fen_column():
    columns = {column.key for column in Game.__table__.columns}
    assert "moves" in columns
    assert "fen" not in columns


def test_a_game_round_trips_through_the_database(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'games.db').as_posix()}")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        game = Game(human_side="black", moves=["e2e4", "e7e5"], last_trace={"move": "g1f3"})
        session.add(game)
        session.commit()
        loaded = session.scalars(select(Game).order_by(Game.id)).one()

    assert loaded.moves == ["e2e4", "e7e5"]
    assert loaded.last_trace == {"move": "g1f3"}
    assert loaded.status == STATUS_ACTIVE
    assert loaded.result is None


def test_timestamps_default_to_utc_aware_datetimes(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'stamps.db').as_posix()}")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        game = Game(human_side="white", moves=[])
        session.add(game)
        session.flush()

        assert game.created_at.tzinfo is not None
        assert game.created_at.utcoffset() == timedelta(0)
        assert game.updated_at.tzinfo is not None
        assert game.updated_at.utcoffset() == timedelta(0)


def test_make_engine_configures_sqlite_for_cross_thread_use(monkeypatch):
    """Pin the connect arguments `make_engine` hands to SQLAlchemy.

    Nothing else can observe this choice: SQLAlchemy 2.1's own SQLite dialect already
    defaults `check_same_thread` to `False`, so an end-to-end test passes whether or not
    the wrapper asks for it. Asserting the argument keeps the explicit request, which is
    what carries the behaviour on 2.0, whose dialect default is the opposite.
    """
    recorded: list[tuple[str, dict[str, object]]] = []

    def fake_create_engine(url: str, **kwargs: object) -> object:
        recorded.append((url, kwargs))
        return object()

    monkeypatch.setattr(db_module, "create_engine", fake_create_engine)

    make_engine("sqlite:///games.db")
    make_engine("postgresql://localhost/games")

    assert recorded == [
        ("sqlite:///games.db", {"connect_args": {"check_same_thread": False}}),
        ("postgresql://localhost/games", {"connect_args": {}}),
    ]


def test_a_connection_can_actually_be_used_from_another_thread(tmp_path):
    """The end-to-end property behind that argument: no `ProgrammingError`."""
    engine = make_engine(f"sqlite:///{(tmp_path / 'threads.db').as_posix()}")
    Base.metadata.create_all(engine)

    failures: list[Exception] = []

    with engine.connect() as connection:

        def reuse_the_connection() -> None:
            try:
                connection.execute(select(Game)).all()
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=reuse_the_connection)
        thread.start()
        thread.join()

    assert failures == []
