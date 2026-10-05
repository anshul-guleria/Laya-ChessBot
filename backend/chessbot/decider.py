"""Turns a position into one legal move, by asking the model a chain of questions.

The model cannot name a move. It can only answer a multiple-choice question about the
position. A move is therefore recovered by narrowing: which piece moves, then for
bishops, rooks and queens which direction, then which destination square.

Every option offered at every stage is derived from `board.legal_moves`. That is what
makes an illegal move structurally impossible rather than merely unlikely, and it is why
a destination label can be shared by two moves -- two rooks reaching b1 -- without the
question chain pretending the model could tell them apart.

Selection is greedy -- the argmax at each stage -- so the bot is deterministic and its
decisions are reproducible. That is how the checkpoint was evaluated, and per-stage
argmax is not the same as maximising the product of the stage probabilities.

Nothing in this module imports `laya` or `torch`; it talks to the `Agent` protocol, so
the tests drive it with a fake.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import chess

from chessbot.agent import Agent, ChoiceResult
from chessbot.engine import destination_label, direction_of, piece_symbol_of
from chessbot.prompts import (
    INSTRUCTIONS,
    QUESTION_DESTINATION,
    QUESTION_DIRECTION,
    QUESTION_PIECE,
    SLIDING_PIECES,
    destination_options,
    destination_state,
    direction_options,
    direction_state,
    piece_options,
    piece_state,
)

logger = logging.getLogger(__name__)

FORCED_REASON = "only one candidate move"


@dataclass(frozen=True)
class StageCall:
    """One question asked of the model, and what it answered.

    Attributes:
        stage: `"piece"`, `"direction"` or `"destination"`.
        question_id: The key the answer came back under.
        state: The exact state text shown to the model.
        instructions: The exact instruction text shown to the model.
        options: Mapping of option key to label, as offered. Empty when the stage was
            skipped, because nothing was offered.
        choice: The option the decider used. `None` when the stage was skipped.
        probabilities: Probability per offered option, empty when the stage was skipped.
        latency_ms: How long the `predict_choice` call took, zero when it was skipped.
        skipped_reason: Set, instead of `choice`, when the stage was not asked because
            only one candidate move remained.
    """

    stage: str
    question_id: str
    state: str
    instructions: str
    options: dict[str, str] = field(default_factory=dict)
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    skipped_reason: str | None = None


@dataclass(frozen=True)
class Decision:
    """The chosen move, plus everything needed to explain it.

    Attributes:
        move: The legal move to play, always a member of `board.legal_moves`.
        stages: One `StageCall` per stage, in the order they were considered, including
            the stages that were skipped. A pawn or knight leaves two entries; a bishop,
            rook or queen leaves three, whether or not each was asked.
        total_latency_ms: Sum of the stage latencies, so a skipped stage costs nothing.
        forced: True when the position had exactly one legal move, which made every
            question unnecessary. A chain that merely narrowed to one move part-way
            through is not forced: the model was asked, and its answer is in the trace.
        tie_break: Note describing an ambiguity and how lowest UCI resolved it, or `None`
            when the choice was unambiguous.
    """

    move: chess.Move
    stages: list[StageCall]
    total_latency_ms: float
    forced: bool = False
    tie_break: str | None = None


def decide_move(agent: Agent, board: chess.Board) -> Decision:
    """Choose one legal move for the side to move.

    Runs the piece stage, then the direction stage for sliding pieces, then the
    destination stage. If at any point only one candidate move remains, the remaining
    stages are skipped but still recorded, because the answer is already determined. At
    1-8 seconds per call on CPU, skipping two calls in a forced position is worth the
    branch.

    Args:
        agent: The model wrapper to ask.
        board: The position to move in. `board.turn`'s side must be the one moving.

    Returns:
        A `Decision` whose `move` is a member of `board.legal_moves`.

    Raises:
        ValueError: If the position has no legal moves, or if the chain eliminated every
            candidate, which should be impossible because every option offered came from
            `board.legal_moves` and the agent's answer is repaired before it is used.
    """
    candidates = list(board.legal_moves)
    if not candidates:
        raise ValueError(f"no legal moves in {board.fen()}")

    stages: list[StageCall] = []
    forced = len(candidates) == 1

    piece = decide_piece(agent, board, candidates, stages)
    candidates = narrow(candidates, lambda move: piece_symbol_of(board, move) == piece)

    direction: str | None = None
    if piece in SLIDING_PIECES:
        if len(candidates) == 1:
            # Nothing to ask, but the destination state the model would have been shown
            # still names the direction, and this one is already determined.
            direction = direction_of(candidates[0])
            stages.append(
                skipped_stage("direction", QUESTION_DIRECTION, direction_state(board, piece))
            )
        else:
            call = ask(
                agent,
                "direction",
                QUESTION_DIRECTION,
                direction_state(board, piece),
                direction_options(candidates),
            )
            stages.append(call)
            direction = call.choice
            candidates = narrow(candidates, lambda move: direction_of(move) == direction)

    if len(candidates) == 1:
        stages.append(
            skipped_stage(
                "destination",
                QUESTION_DESTINATION,
                destination_state(board, piece, direction),
            )
        )
    else:
        call = ask(
            agent,
            "destination",
            QUESTION_DESTINATION,
            destination_state(board, piece, direction),
            destination_options(candidates),
        )
        stages.append(call)
        chosen = call.choice
        candidates = narrow(candidates, lambda move: destination_label(move) == chosen)

    move, tie_break = resolve(candidates)
    total = sum(stage.latency_ms for stage in stages)

    logger.info(
        "chose %s in %s (%.0fms%s%s)",
        move.uci(),
        board.fen(),
        total,
        ", forced" if forced else "",
        f", {tie_break}" if tie_break else "",
    )

    return Decision(
        move=move,
        stages=stages,
        total_latency_ms=total,
        forced=forced,
        tie_break=tie_break,
    )


def decide_piece(
    agent: Agent, board: chess.Board, candidates: list[chess.Move], stages: list[StageCall]
) -> str:
    """Run the piece stage and record it.

    Args:
        agent: The model wrapper.
        board: The position to read.
        candidates: The legal moves to choose among. Always non-empty.
        stages: The trace to append the stage to.

    Returns:
        The chosen piece symbol: `"P"`, `"N"`, `"B"`, `"R"`, `"Q"` or `"K"`. A move's
        piece has to be read off `board`, because `chess.Move` carries only squares --
        which also means a castle is attributed to the king that made it.
    """
    if len(candidates) == 1:
        stages.append(skipped_stage("piece", QUESTION_PIECE, piece_state(board)))
        return piece_symbol_of(board, candidates[0])

    call = ask(agent, "piece", QUESTION_PIECE, piece_state(board), piece_options(board, candidates))
    stages.append(call)
    return str(call.choice)


def ask(
    agent: Agent, stage: str, question_id: str, state: str, options: dict[str, str]
) -> StageCall:
    """Ask one stage's question and record the outcome.

    A choice that is not one of the offered keys is replaced by the first offered key
    rather than used: the chain filters candidates on the choice, so a key that was never
    offered would empty the candidate set and leave the bot unable to move. The trace
    records the key that was actually used, which is the one worth replaying.

    Args:
        agent: The model wrapper.
        stage: The stage name, for the trace.
        question_id: The question key.
        state: The state text.
        options: The options offered, always derived from legal moves and never empty.

    Returns:
        A `StageCall` with the answer filled in and a `choice` that is one of `options`.
    """
    instructions = INSTRUCTIONS[question_id]
    result = agent.predict_choice(state, question_id, instructions, options)

    choice = result.choice
    if choice not in options:
        fallback = next(iter(options))
        logger.warning(
            "%s stage returned choice %r for %s, which was not offered; falling back to %r",
            stage,
            choice,
            question_id,
            fallback,
        )
        choice = fallback

    logger.debug(
        "%s stage\n  state: %s\n  options: %s\n  choice: %s\n  probabilities: %s\n  took %.0fms",
        stage,
        state,
        options,
        choice,
        result.probabilities,
        result.latency_ms,
    )

    return StageCall(
        stage=stage,
        question_id=question_id,
        state=state,
        instructions=instructions,
        options=options,
        choice=choice,
        probabilities=result.probabilities,
        latency_ms=result.latency_ms,
    )


def skipped_stage(stage: str, question_id: str, state: str) -> StageCall:
    """Build a `StageCall` for a stage that was not asked.

    The state text is still recorded, so the trace shows what the model would have been
    asked even when the answer was already determined. Nothing else is recorded: no
    options were offered, so there is nothing to report a choice or a probability for.

    Args:
        stage: The stage name.
        question_id: The question key.
        state: The state text that would have been sent.

    Returns:
        A `StageCall` with `skipped_reason` set and no choice.
    """
    return StageCall(
        stage=stage,
        question_id=question_id,
        state=state,
        instructions=INSTRUCTIONS[question_id],
        skipped_reason=FORCED_REASON,
    )


def narrow(candidates: list[chess.Move], keep: Callable[[chess.Move], bool]) -> list[chess.Move]:
    """Keep only the candidates a stage's answer leaves standing.

    Args:
        candidates: The legal moves in contention. Always non-empty.
        keep: The predicate the answer implies, e.g. `destination_label(move) == choice`.

    Returns:
        The candidates that matched, never empty.

    Raises:
        ValueError: If the answer matched nothing. Unreachable while every option came
            from these candidates and the answer is one of the options, so it is a
            guard against a future stage that forgets one of those two things.
    """
    surviving = [move for move in candidates if keep(move)]
    if not surviving:
        raise ValueError(
            "the decision chain eliminated every candidate; every option offered was "
            "derived from these legal moves, so this should be impossible"
        )
    return surviving


def resolve(candidates: list[chess.Move]) -> tuple[chess.Move, str | None]:
    """Pick one move from the surviving candidates.

    Nothing in the question chain separates two same-type pieces that reach the same
    square -- two rooks both able to play to e1, for instance. The tie-break is fixed and
    always lowest UCI, and it is reported, so the ambiguity is visible rather than
    something a player has to infer from a strange move.

    Args:
        candidates: The surviving legal moves. Must be non-empty.

    Returns:
        A `(move, tie_break)` pair, where `tie_break` is `None` when there was exactly
        one candidate.
    """
    if len(candidates) == 1:
        return candidates[0], None

    ordered = sorted(candidates, key=lambda move: move.uci())
    chosen = ordered[0]
    note = (
        f"{len(candidates)} candidates matched "
        f"({', '.join(move.uci() for move in ordered)}); chose {chosen.uci()} by lowest UCI"
    )
    logger.warning("ambiguous decision: %s", note)
    return chosen, note


__all__ = ["ChoiceResult", "Decision", "StageCall", "decide_move"]
