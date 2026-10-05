"""Game endpoints.

Every endpoint rebuilds the board by replaying `game.moves`, so the position the bot
reasons about is always the position the player sees. `build_game_state` is the single
response shape, and it recomputes the status from that replay, so a stale column can never
reach the client.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated

import chess
import chess.pgn
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.db import get_db
from app.deps import get_agent
from app.engine_support import STATUS_ACTIVE, board_from_moves, game_status, uci_to_move
from app.models import Game
from app.schemas import (
    BotTrace,
    GameState,
    GameSummary,
    MoveRequest,
    StageTrace,
    build_game_state,
)
from chessbot.agent import Agent
from chessbot.decider import Decision, decide_move

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/games", tags=["games"])

SIDES = ("white", "black")

# Declared with `Annotated` rather than as a `Depends(...)` default: the annotation form is
# what FastAPI documents today, and it keeps the dependency call out of the signature's
# default value, which is exactly what `Depends(...)` there means -- an object built at
# import time and reused for every request.
DbSession = Annotated[Session, Depends(get_db)]
AgentDep = Annotated[Agent, Depends(get_agent)]


def datetime_now() -> datetime:
    """Return an aware UTC timestamp.

    Returns:
        The current time, for `updated_at`.
    """
    return datetime.now(UTC)


def colour_of(side: str) -> chess.Color:
    """Return the chess colour for a stored side name.

    Args:
        side: `"white"` or `"black"`, as stored on the row.

    Returns:
        `chess.WHITE` or `chess.BLACK`. Any other value is read as white, which is the only
        value this application ever writes.
    """
    return chess.BLACK if side == "black" else chess.WHITE


def finish(game: Game, status_value: str, result: str | None, termination: str | None) -> None:
    """Record a finished game on the row.

    These columns are a cache of a replay, kept so the game list can be answered without
    replaying every game in the table. `build_game_state` recomputes all three on every
    read, so nothing here can contradict the move list.

    Args:
        game: The game to update.
        status_value: `STATUS_ACTIVE`, `STATUS_WHITE_WON`, `STATUS_BLACK_WON` or
            `STATUS_DRAW`.
        result: The PGN result, or `None` while live.
        termination: Why the game ended, or `None`.
    """
    live = status_value == STATUS_ACTIVE
    game.status = status_value
    game.result = None if live else result
    game.termination = None if live else termination
    game.updated_at = datetime_now()


def load_game(db: Session, game_id: int) -> Game:
    """Fetch a game or fail with `404`.

    Args:
        db: The request-scoped session.
        game_id: The game to fetch.

    Returns:
        The `Game`.

    Raises:
        HTTPException: `404` when no such game exists.
    """
    game = db.get(Game, game_id)
    if game is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"no game {game_id}")
    return game


def active_board(game: Game) -> chess.Board:
    """Rebuild the board and require the game to still be running.

    The status is recomputed from the replay rather than read off the row, so a row whose
    cached `status` was written before the last move still refuses further play.

    Args:
        game: The stored game.

    Returns:
        The replayed `chess.Board`.

    Raises:
        HTTPException: `409` when the game is already finished.
    """
    board = board_from_moves(game.moves or [])
    status_value, _, _ = game_status(board)
    if status_value != STATUS_ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"game {game.id} is over ({status_value})",
        )
    return board


def bot_trace_of(decision: Decision) -> BotTrace:
    """Convert a `Decision` into the API trace shape.

    `StageCall` is converted field by field rather than handed over as-is, because
    `BotTrace` declares `list[StageTrace]`. Relying on pydantic to coerce a dataclass into
    the model it was not declared as would make the response shape an accident of the
    library's validation settings rather than a decision anyone made.

    Args:
        decision: What `decide_move` returned.

    Returns:
        The `BotTrace` to store on the game and return to the client.
    """
    return BotTrace(
        move=decision.move.uci(),
        stages=[StageTrace.model_validate(asdict(stage)) for stage in decision.stages],
        total_latency_ms=decision.total_latency_ms,
        forced=decision.forced,
        tie_break=decision.tie_break,
    )


async def play_bot_move(agent: Agent, board: chess.Board) -> tuple[chess.Move, BotTrace]:
    """Ask the bot for a move without blocking the event loop.

    `decide_move` takes 1-8 seconds on CPU, so it runs in the threadpool. On the event
    loop it would stall every other request -- including the health check the frontend
    polls -- for the duration of the model's answer.

    Args:
        agent: The loaded model wrapper.
        board: The position the bot is to move in.

    Returns:
        The chosen move and the trace explaining it.

    Raises:
        Exception: Whatever `decide_move` raised. Callers turn that into `503` and undo any
            ply the human had already staged.
    """
    decision = await run_in_threadpool(decide_move, agent, board)
    return decision.move, bot_trace_of(decision)


@router.post("", response_model=GameState, status_code=status.HTTP_201_CREATED)
async def create_game(db: DbSession, agent: AgentDep) -> GameState:
    """Create a game, randomising which side the human plays.

    The human's side is chosen here rather than by the client so a player cannot
    accidentally always play the same colour, and so the first move -- whether the human
    or the bot -- is decided by one place.

    When the human draws black the bot opens. That opening is decided before the row is
    written, so a model that cannot answer leaves no game behind for the human to stare
    at with nobody to play against.

    Args:
        db: The request-scoped session.
        agent: The loaded model wrapper.

    Returns:
        The new game's state, including the bot's opening move when the human is black.

    Raises:
        HTTPException: `503` when the bot was to open and could not produce a move.
    """
    # A coin flip, not a secret.
    human_side = random.choice(SIDES)

    moves: list[str] = []
    opening_trace: BotTrace | None = None
    if human_side == "black":
        try:
            opening, opening_trace = await play_bot_move(agent, chess.Board())
        except Exception as error:
            logger.exception("the bot could not open a game; nothing was saved")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the bot could not open the game; nothing was saved",
            ) from error
        moves.append(opening.uci())
        logger.info("the bot opened with %s", opening.uci())

    game = Game(human_side=human_side, status=STATUS_ACTIVE, moves=moves)
    if opening_trace is not None:
        game.last_trace = opening_trace.model_dump()
    db.add(game)
    db.commit()
    db.refresh(game)

    logger.info("game %s created, human plays %s", game.id, human_side)
    return build_game_state(game, bot_trace=opening_trace)


@router.get("", response_model=list[GameSummary])
def list_games(db: DbSession) -> list[GameSummary]:
    """List games, newest first.

    Args:
        db: The request-scoped session.

    Returns:
        One `GameSummary` per game.
    """
    games = db.scalars(select(Game).order_by(Game.id.desc())).all()
    return [
        GameSummary(
            id=game.id,
            human_side=game.human_side,
            status=game.status,
            move_count=len(game.moves or []),
            created_at=game.created_at.isoformat(),
        )
        for game in games
    ]


@router.get("/{game_id}", response_model=GameState)
def get_game(game_id: int, db: DbSession) -> GameState:
    """Return one game's full state.

    Args:
        game_id: The game to read.
        db: The request-scoped session.

    Returns:
        The `GameState`.

    Raises:
        HTTPException: `404` when no such game exists.
    """
    return build_game_state(load_game(db, game_id))


@router.post("/{game_id}/move", response_model=GameState)
async def play_move(
    game_id: int,
    payload: MoveRequest,
    db: DbSession,
    agent: AgentDep,
) -> GameState:
    """Play the human's move, then the bot's reply.

    The human move and the bot reply land in the same request, so the client has one
    response to render and the two plies can never be persisted apart.

    Args:
        game_id: The game to play in.
        payload: The human's move.
        db: The request-scoped session.
        agent: The loaded model wrapper.

    Returns:
        The state after both plies, including the bot's trace.

    Raises:
        HTTPException: `404` unknown game, `409` game over or not the human's turn, `400`
            unparseable or illegal move, `503` if the bot's move could not be produced.
    """
    game = load_game(db, game_id)
    board = active_board(game)

    if board.turn != colour_of(game.human_side):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"it is not the human's turn in game {game.id}",
        )

    try:
        move = uci_to_move(board, payload.uci)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    board.push(move)
    game.moves = list(game.moves or []) + [move.uci()]
    logger.info("game %s: human played %s", game.id, move.uci())

    status_value, result, termination = game_status(board)
    if status_value != STATUS_ACTIVE:
        # The human's move ended the game, so there is no reply to ask for.
        finish(game, status_value, result, termination)
        db.commit()
        db.refresh(game)
        logger.info("game %s finished: %s (%s)", game.id, status_value, termination)
        return build_game_state(game)

    try:
        bot_move, trace = await play_bot_move(agent, board)
    except Exception as error:
        logger.exception("game %s: the bot failed to decide a move", game.id)
        # Nothing was committed, so rolling the session back leaves the stored history
        # exactly as the player last saw it. They can retry the same move, and it will be
        # evaluated against the same position rather than against their own move.
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the bot could not decide a move; nothing was saved",
        ) from error

    board.push(bot_move)
    game.moves = list(game.moves or []) + [bot_move.uci()]
    game.last_trace = trace.model_dump()
    status_value, result, termination = game_status(board)
    finish(game, status_value, result, termination)
    db.commit()
    db.refresh(game)

    logger.info(
        "game %s: bot replied %s in %.0fms", game.id, bot_move.uci(), trace.total_latency_ms
    )
    return build_game_state(game, bot_trace=trace)


@router.post("/{game_id}/undo", response_model=GameState)
def undo(game_id: int, db: DbSession) -> GameState:
    """Take back the last human move and the bot's reply.

    One ply is removed, and a second when that leaves the bot to move: leaving a position
    on the bot's turn would hand the player a game they cannot act in. A game that has
    already ended is restored to active by the same edit, because the status is recomputed
    from the shortened move list.

    Args:
        game_id: The game to undo in.
        db: The request-scoped session.

    Returns:
        The state after the rollback.

    Raises:
        HTTPException: `404` unknown game, `409` when there is nothing to undo.
    """
    game = load_game(db, game_id)
    moves = list(game.moves or [])
    if not moves:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="there is nothing to undo")

    moves.pop()
    board = board_from_moves(moves)
    if board.turn != colour_of(game.human_side):
        # Taking back only the human's own move would leave the bot's reply in place and
        # the game on the bot's turn.
        if not moves:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="there is nothing to undo; the bot's opening is all that has been played",
            )
        moves.pop()

    game.moves = moves
    game.last_trace = None
    game.status = STATUS_ACTIVE
    game.result = None
    game.termination = None
    game.updated_at = datetime_now()
    db.commit()
    db.refresh(game)

    logger.info("game %s: undid back to ply %d", game.id, len(moves))
    return build_game_state(game)


@router.get("/{game_id}/pgn", response_class=Response)
def export_pgn(game_id: int, db: DbSession) -> Response:
    """Export the game as PGN.

    Args:
        game_id: The game to export.
        db: The request-scoped session.

    Returns:
        The PGN as `text/plain`.

    Raises:
        HTTPException: `404` when no such game exists.
    """
    game = load_game(db, game_id)
    board = board_from_moves(game.moves or [])
    human_is_white = game.human_side == "white"
    # Recomputed rather than read off the row, so the header cannot contradict the moves.
    # `Game.from_board` would set `Result` from `board.result()`, which knows about mate
    # and stalemate but not about a fifty-move or repetition draw.
    _, result, _ = game_status(board)
    export = chess.pgn.Game.from_board(board)
    export.headers["White"] = "Human" if human_is_white else "Laya"
    export.headers["Black"] = "Laya" if human_is_white else "Human"
    export.headers["Result"] = result or "*"
    return Response(content=str(export), media_type="text/plain")


__all__ = [
    "create_game",
    "export_pgn",
    "get_game",
    "list_games",
    "play_move",
    "router",
    "undo",
]
