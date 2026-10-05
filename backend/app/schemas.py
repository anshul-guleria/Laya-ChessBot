"""Pydantic request and response models.

`GameState` is the single response shape for every game endpoint. It is built from the
move list on every read, so it always reflects the position the bot reasoned about, and
its `status`, `result` and `termination` are recomputed from that position rather than
read off the row.
"""

from __future__ import annotations

import chess
from pydantic import BaseModel, Field

from app.engine_support import board_from_moves, game_status, legal_uci_moves
from app.models import Game


class MoveRequest(BaseModel):
    """A human move.

    Attributes:
        uci: The move in UCI notation, e.g. `"e2e4"` or `"e7e8q"`. Five characters is the
            longest legal value, being a promotion such as `e7e8q`.
    """

    uci: str = Field(min_length=2, max_length=5, examples=["e2e4"])


class StageTrace(BaseModel):
    """One question asked of the model, and what it answered.

    Attributes:
        stage: `"piece"`, `"direction"` or `"destination"`.
        question_id: The key the answer came back under.
        state: The exact state text shown to the model.
        instructions: The exact instruction text shown to the model.
        options: Mapping of option key to label, as offered.
        choice: The option the model selected, or `None` if the stage was skipped.
        probabilities: Probability per offered option.
        latency_ms: How long the `predict` call took.
        skipped_reason: Set when the stage was not asked.
    """

    stage: str
    question_id: str
    state: str
    instructions: str
    options: dict[str, str] = Field(default_factory=dict)
    choice: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    skipped_reason: str | None = None


class BotTrace(BaseModel):
    """Everything the model did to produce one move.

    Attributes:
        move: The move it chose, in UCI.
        stages: One entry per stage, in order, including stages that were skipped.
        total_latency_ms: Sum of the stage latencies.
        forced: True when a single legal move made the questions unnecessary.
        tie_break: Note about an ambiguity and how lowest-UCI resolved it, if any.
    """

    move: str
    stages: list[StageTrace] = Field(default_factory=list)
    total_latency_ms: float
    forced: bool = False
    tie_break: str | None = None


class GameSummary(BaseModel):
    """A row in the game list.

    Attributes:
        id: Game id.
        human_side: The human's colour.
        status: Current status.
        move_count: How many plies have been played.
        created_at: When the game was created, ISO 8601.
    """

    id: int
    human_side: str
    status: str
    move_count: int
    created_at: str


class GameState(BaseModel):
    """The full state of a game. Returned by every game endpoint.

    Attributes:
        id: Game id.
        fen: The current position, derived from `moves`.
        moves: The UCI move list.
        turn: Which colour is to move.
        human_side: The human's colour.
        status: Current status, recomputed from `fen`.
        result: PGN result, or `None` while live.
        termination: Why the game ended, or `None`.
        is_check: Whether the side to move is in check.
        legal_moves: Legal moves for the side to move, as UCI. Sent so client and server
            agree on legality by construction; the server still re-validates.
        last_move: The most recent move played, or `None` in a fresh game.
        bot_trace: The most recent bot decision, or `None` before the bot's first move.
    """

    id: int
    fen: str
    moves: list[str]
    turn: str
    human_side: str
    status: str
    result: str | None = None
    termination: str | None = None
    is_check: bool = False
    legal_moves: list[str] = Field(default_factory=list)
    last_move: str | None = None
    bot_trace: BotTrace | None = None


class HealthResponse(BaseModel):
    """Whether the service can decide a move.

    Attributes:
        status: `"ok"` or `"degraded"`.
        model_loaded: Whether the checkpoint is in memory.
        device: Torch device in use.
        model_dir: The checkpoint path, or the path that failed to load.
    """

    status: str
    model_loaded: bool
    device: str
    model_dir: str


def build_game_state(game: Game, bot_trace: BotTrace | None = None) -> GameState:
    """Build the API view of a game from its move list.

    The board is rebuilt by replaying `moves`, and `status`, `result` and `termination`
    are taken from that position rather than from the row's columns. A row that went stale
    -- a status written before the last move, or hand-edited -- therefore cannot reach the
    client.

    Args:
        game: The stored game. Only `id`, `human_side`, `moves` and `last_trace` are read,
            so a `Game` built in a test without going through the database works.
        bot_trace: A freshly produced trace to attach, overriding the stored one. Used by
            the move endpoint, where the trace was just created.

    Returns:
        The `GameState` for the position the move list describes.
    """
    board = board_from_moves(game.moves or [])
    status, result, termination = game_status(board)

    trace = bot_trace
    if trace is None and game.last_trace:
        trace = BotTrace.model_validate(game.last_trace)

    return GameState(
        id=game.id,
        fen=board.fen(),
        moves=list(game.moves or []),
        turn="white" if board.turn == chess.WHITE else "black",
        human_side=game.human_side,
        status=status,
        result=result,
        termination=termination,
        is_check=board.is_check(),
        legal_moves=legal_uci_moves(board),
        last_move=game.moves[-1] if game.moves else None,
        bot_trace=trace,
    )
