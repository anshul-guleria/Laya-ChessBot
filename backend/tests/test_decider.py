"""The three-stage decision chain.

Every test drives a `FakeAgent`, so none load the checkpoint. The tests assert four
things: the chosen move is always legal, every option offered came from `legal_moves`,
the chain asks the right number of questions, and the trace faithfully records what
happened.

Positions are built from move lists wherever an opening line reaches the invariant being
tested, and from FENs where the invariant is artificial -- a rook with exactly one move,
say, or a slider that can only travel one way.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import deque

import chess
import pytest

from chessbot.agent import ChoiceResult
from chessbot.decider import decide_move
from chessbot.engine import (
    board_from_moves,
    destination_label,
    direction_of,
    piece_symbol_of,
)
from chessbot.prompts import QUESTION_DIRECTION, QUESTION_PIECE
from tests.support import FakeAgent

# White to move, a rook with a move up the a-file and along the first rank. Castling
# rights are deliberately absent so every rook move is an ordinary rook move.
ROOK_BOARD = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"

# The same, but a white knight on b1 and a black pawn on a2 leave the rook on a1 with
# exactly one move, which is the capture of that pawn.
ONE_ROOK_MOVE_BOARD = "4k3/8/8/8/8/8/p7/RN2K3 w - - 0 1"

# A knight on b1 blocks the rook on a1 along the first rank, so the rook's seven moves
# all travel the same way up the a-file.
ONE_DIRECTION_BOARD = "4k3/8/8/8/8/8/8/RN2K3 w - - 0 1"

# Nothing but a king, and it has five moves.
KING_ONLY_BOARD = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"

# A pawn on e7 with the black king well out of the way, so all four promotions are legal.
# Putting the black king on e8 instead -- the obvious transcription -- leaves the pawn
# with no legal move at all, because it may not capture a king.
PROMOTION_BOARD = "7k/4P3/8/8/8/8/8/4K3 w - - 0 1"

# Two rooks that can both reach b1, c1 and d1, and a king that can reach none of the
# squares only the rook on e1 can.
TWO_ROOKS_BOARD = "4k3/8/8/8/8/8/8/R3R2K w - - 0 1"


class RogueAgent(FakeAgent):
    """A `FakeAgent` that answers with a key the decider never offered.

    `LocalLayaAgent` promises a choice drawn from the options, so this double breaks the
    promise on purpose: it is the only way to reach the decider's defensive path.
    """

    def predict_choice(
        self, state: str, question_id: str, instructions: str, options: dict[str, str]
    ) -> ChoiceResult:
        """Return the scripted answer with its choice replaced by an unoffered key.

        Args:
            state: The state text.
            question_id: The question key.
            instructions: The instruction text.
            options: The options offered.

        Returns:
            A `ChoiceResult` whose `choice` is in none of `options`.
        """
        result = super().predict_choice(state, question_id, instructions, options)
        return dataclasses.replace(result, choice="9Q-not-offered")


def sample_positions() -> list[chess.Board]:
    """Return a spread of positions covering every stage shape.

    Returns:
        Boards with only pawns and knights to move, boards that reach a slider's
        direction stage, an en passant position, a promotion, and a castling position.
    """
    return [
        board_from_moves([]),
        board_from_moves(["e2e4"]),
        board_from_moves(["e2e4", "d7d5"]),
        board_from_moves(["e2e4", "d7d5", "g1f3", "d5e4", "f1e2"]),
        board_from_moves(["e2e4", "a7a6", "e4e5", "d7d5"]),
        chess.Board(ROOK_BOARD),
        chess.Board(PROMOTION_BOARD),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ]


# --- the basic contract ----------------------------------------------------


def test_the_chosen_move_is_always_legal():
    decision = decide_move(FakeAgent(), chess.Board())
    assert decision.move in chess.Board().legal_moves


@pytest.mark.parametrize("board", sample_positions(), ids=lambda board: board.fen()[:44])
def test_every_option_offered_came_from_a_legal_move(board: chess.Board):
    """An illegal move has to be structurally impossible, not merely unlikely.

    At every stage, the options offered are exactly the ones the legal moves still in
    contention produce: the piece types that can move, the directions the chosen piece can
    travel, and the squares it can reach along the chosen direction.
    """
    legal = list(board.legal_moves)
    decision = decide_move(FakeAgent(), board)

    assert decision.move in legal

    piece = decision.stages[0].choice
    piece_moves = [move for move in legal if piece_symbol_of(board, move) == piece]

    for index, stage in enumerate(decision.stages):
        if stage.skipped_reason is not None:
            continue
        if stage.question_id == QUESTION_PIECE:
            assert set(stage.options) == {piece_symbol_of(board, move) for move in legal}
        elif stage.question_id == QUESTION_DIRECTION:
            assert set(stage.options) == {direction_of(move) for move in piece_moves}
        else:
            in_play = piece_moves
            previous = decision.stages[index - 1]
            if previous.question_id == QUESTION_DIRECTION and previous.choice is not None:
                in_play = [move for move in piece_moves if direction_of(move) == previous.choice]
            assert set(stage.options) == {destination_label(move) for move in in_play}


def test_the_same_position_always_gives_the_same_decision():
    """Selection is per-stage argmax over legal moves, so it is reproducible."""
    first = decide_move(FakeAgent(), chess.Board())
    second = decide_move(FakeAgent(), chess.Board())
    assert first.move == second.move
    assert [stage.choice for stage in first.stages] == [stage.choice for stage in second.stages]
    assert [stage.options for stage in first.stages] == [stage.options for stage in second.stages]


def test_selection_is_the_greedy_argmax_at_every_stage():
    board = board_from_moves(["e2e4", "d7d5"])
    agent = FakeAgent(
        probabilities={"piece_selection": {"N": 1.0}, "destination_selection": {"f3": 1.0}}
    )
    decision = decide_move(agent, board)

    piece_stage, destination_stage = decision.stages
    assert piece_stage.choice == "N"
    assert piece_stage.choice == max(piece_stage.probabilities, key=piece_stage.probabilities.get)
    assert destination_stage.choice == "f3"
    assert destination_stage.choice == max(
        destination_stage.probabilities, key=destination_stage.probabilities.get
    )
    assert decision.move.uci() == "g1f3"


def test_a_choice_that_was_never_offered_falls_back_to_the_first_option(caplog):
    agent = RogueAgent()
    board = chess.Board()

    with caplog.at_level(logging.WARNING, logger="chessbot.decider"):
        decision = decide_move(agent, board)

    assert decision.move in board.legal_moves
    for stage in decision.stages:
        if stage.skipped_reason is None:
            assert stage.choice in stage.options
            assert stage.choice == next(iter(stage.options))
    assert "not offered" in caplog.text
    assert "falling back to" in caplog.text


# --- how many questions ---------------------------------------------------


def test_pawn_path_asks_two_questions():
    """Pawns, knights and kings skip the direction stage, so two calls, not three."""
    board = board_from_moves(["e2e4", "d7d5"])
    agent = FakeAgent(script={"piece_selection": "P", "destination_selection": "d5"})
    decision = decide_move(agent, board)
    assert agent.question_ids == ["piece_selection", "destination_selection"]
    assert decision.move.uci() == "e4d5"
    assert [stage.stage for stage in decision.stages] == ["piece", "destination"]


def test_rook_path_asks_three_questions_in_order():
    board = chess.Board(ROOK_BOARD)
    agent = FakeAgent(script={"piece_selection": "R", "direction_selection": "vertical"})
    decision = decide_move(agent, board)
    assert agent.question_ids == [
        "piece_selection",
        "direction_selection",
        "destination_selection",
    ]
    assert [stage.stage for stage in decision.stages] == ["piece", "direction", "destination"]
    assert "Selected direction: vertical." in decision.stages[2].state


def test_a_slider_with_one_direction_still_asks_a_direction_and_a_destination():
    """A stage is skipped for having one *move*, not for having one option."""
    board = chess.Board(ONE_DIRECTION_BOARD)
    agent = FakeAgent(script={"piece_selection": "R"})
    decision = decide_move(agent, board)

    assert agent.question_ids == [
        "piece_selection",
        "direction_selection",
        "destination_selection",
    ]
    assert set(decision.stages[1].options) == {"vertical"}
    assert decision.move.uci().startswith("a1a")
    assert decision.move in board.legal_moves


def test_a_single_piece_type_is_still_asked_about():
    """Only the king can move, but it has five moves, so the piece stage is still asked."""
    board = chess.Board(KING_ONLY_BOARD)
    agent = FakeAgent()
    decision = decide_move(agent, board)

    assert decision.stages[0].options == {"K": "King"}
    assert decision.stages[0].choice == "K"
    assert decision.stages[0].skipped_reason is None
    assert decision.forced is False
    assert agent.question_ids == ["piece_selection", "destination_selection"]
    assert decision.move in board.legal_moves


# --- forced-move short-circuit --------------------------------------------


def test_a_position_with_one_legal_move_asks_nothing():
    """With one legal move there is nothing to decide, so no model calls are made."""
    board, only_uci = find_single_legal_move_position()
    agent = FakeAgent()
    decision = decide_move(agent, board)

    assert board.turn is not None
    assert decision.move.uci() == only_uci
    assert decision.forced is True
    assert agent.call_count == 0
    assert decision.stages, "the trace must still explain why nothing was asked"
    assert all(stage.skipped_reason for stage in decision.stages)


def test_more_than_one_legal_move_is_not_forced():
    decision = decide_move(FakeAgent(), chess.Board())
    assert decision.forced is False
    assert decision.stages[0].skipped_reason is None


def test_a_short_circuited_chain_still_records_the_skipped_stages():
    """The rook has exactly one move, so only the piece stage was worth asking."""
    board = chess.Board(ONE_ROOK_MOVE_BOARD)
    agent = FakeAgent(script={"piece_selection": "R"})
    decision = decide_move(agent, board)

    assert agent.question_ids == ["piece_selection"]
    assert [stage.stage for stage in decision.stages] == ["piece", "direction", "destination"]

    skipped = decision.stages[1:]
    assert all(stage.skipped_reason for stage in skipped)
    for stage in skipped:
        assert stage.choice is None
        assert stage.options == {}
        assert stage.probabilities == {}
        assert stage.latency_ms == 0.0

    assert "Selected direction: vertical." in skipped[1].state
    assert decision.move.uci() == "a1a2"
    assert decision.forced is False


def test_a_position_with_no_legal_moves_is_rejected():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate
    assert not list(board.legal_moves)
    with pytest.raises(ValueError, match="no legal moves"):
        decide_move(FakeAgent(), board)


# --- tie-break ------------------------------------------------------------


def test_tie_break_picks_lowest_uci_when_two_rooks_share_a_destination():
    """Nothing in the question chain separates two rooks that reach the same square."""
    board = chess.Board(TWO_ROOKS_BOARD)
    agent = FakeAgent(script={"piece_selection": "R", "direction_selection": "horizontal"})
    decision = decide_move(agent, board)

    surviving = sorted(
        move.uci()
        for move in board.legal_moves
        if piece_symbol_of(board, move) == "R"
        and direction_of(move) == "horizontal"
        and destination_label(move) == decision.stages[-1].choice
    )
    assert len(surviving) > 1, "this position is only interesting while the tie survives"

    assert decision.tie_break is not None
    assert "lowest UCI" in decision.tie_break
    assert surviving[0] == "a1b1"
    assert decision.move.uci() == surviving[0]
    assert decision.move.uci() == "a1b1"
    assert all(uci in decision.tie_break for uci in surviving)


def test_tie_break_is_absent_when_the_choice_is_unambiguous():
    assert decide_move(FakeAgent(), chess.Board()).tie_break is None


# --- edge positions -------------------------------------------------------


def test_promotion_is_offered_with_the_promotion_piece():
    board = chess.Board(PROMOTION_BOARD)
    agent = FakeAgent(script={"piece_selection": "P", "destination_selection": "e8=Q"})
    decision = decide_move(agent, board)
    assert decision.move.promotion == chess.QUEEN
    assert decision.move.uci() == "e7e8q"
    assert "e8=Q" in decision.stages[-1].options
    assert set(decision.stages[-1].options) == {"e8=B", "e8=N", "e8=Q", "e8=R"}


def test_castling_is_offered_as_a_destination():
    """python-chess spells castling as a king move from e1, so it is offered as one."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    agent = FakeAgent(script={"piece_selection": "K", "destination_selection": "g1"})
    decision = decide_move(agent, board)

    assert board.is_castling(decision.move)
    assert decision.move.uci() == "e1g1"
    assert "g1" in decision.stages[-1].options


def test_castling_sideways_rook_moves_do_not_shadow_a_castle():
    """The tie-break must still return something legal when a castle shares a direction."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    decision = decide_move(
        FakeAgent(script={"piece_selection": "R", "direction_selection": "horizontal"}), board
    )
    assert decision.move in board.legal_moves


def test_en_passant_is_reachable():
    board = board_from_moves(["e2e4", "a7a6", "e4e5", "d7d5"])
    agent = FakeAgent(script={"piece_selection": "P", "destination_selection": "d6"})
    decision = decide_move(agent, board)
    assert board.is_en_passant(decision.move)
    assert decision.move.uci() == "e5d6"


# --- the trace ------------------------------------------------------------


def test_the_first_question_is_byte_exact():
    board = board_from_moves(["e2e4", "d7d5"])
    agent = FakeAgent()
    decide_move(agent, board)
    question_id, state, instructions, options = agent.calls[0]
    assert question_id == "piece_selection"
    assert state == f"Chess position. FEN: {board.fen()}. Side to move: white. Stage: piece."
    assert instructions == "Choose the best chess piece for the move."
    # No rook: a1 and h1 are still boxed in by their own pawns and knights.
    assert options == {
        "P": "Pawn",
        "N": "Knight",
        "B": "Bishop",
        "Q": "Queen",
        "K": "King",
    }


def test_every_offered_destination_is_a_square_the_chosen_piece_can_reach():
    board = chess.Board(ROOK_BOARD)
    agent = FakeAgent(script={"piece_selection": "R", "direction_selection": "vertical"})
    decide_move(agent, board)
    question_id, _, _, options = agent.calls[-1]
    assert question_id == "destination_selection"
    reachable = {
        destination_label(move)
        for move in board.legal_moves
        if piece_symbol_of(board, move) == "R" and direction_of(move) == "vertical"
    }
    assert reachable == {"a2", "a3", "a4", "a5", "a6", "a7", "a8"}
    assert set(options) == reachable


def test_the_move_matches_every_stage_choice():
    board = board_from_moves(["e2e4", "d7d5"])
    decision = decide_move(FakeAgent(script={"piece_selection": "N"}), board)
    assert piece_symbol_of(board, decision.move) == decision.stages[0].choice
    assert destination_label(decision.move) == decision.stages[-1].choice


def test_total_latency_is_the_sum_of_the_stages():
    decision = decide_move(FakeAgent(), chess.Board())
    assert decision.total_latency_ms == pytest.approx(
        sum(stage.latency_ms for stage in decision.stages)
    )


def test_probabilities_cover_exactly_the_offered_options():
    decision = decide_move(FakeAgent(), chess.Board())
    asked = [stage for stage in decision.stages if stage.skipped_reason is None]
    assert asked
    for stage in asked:
        assert set(stage.probabilities) == set(stage.options)
        assert sum(stage.probabilities.values()) == pytest.approx(1.0)


def test_each_stage_records_the_instructions_it_used():
    decision = decide_move(FakeAgent(), chess.Board())
    by_id = {stage.question_id: stage.instructions for stage in decision.stages}
    assert by_id["piece_selection"] == "Choose the best chess piece for the move."
    assert by_id["destination_selection"] == "Choose the best chess piece destination."


def test_a_skipped_stage_still_records_the_state_it_would_have_sent():
    board, _ = find_single_legal_move_position()
    decision = decide_move(FakeAgent(), board)
    assert decision.stages[0].state == (
        f"Chess position. FEN: {board.fen()}. Side to move: black. Stage: piece."
    )
    assert decision.stages[0].instructions == "Choose the best chess piece for the move."
    assert decision.stages[0].question_id == "piece_selection"


# --- helpers --------------------------------------------------------------


def find_single_legal_move_position(max_depth: int = 3) -> tuple[chess.Board, str]:
    """Find a position where the side to move has exactly one legal move.

    Breadth-first from the start position, deduplicated by FEN, so the answer is the
    shallowest such position and the search terminates: at `max_depth` there are fewer
    than ten thousand distinct positions. Searching rather than pasting a FEN means the
    assertion is about the invariant rather than about one memorised position, and a
    mistranscribed FEN cannot make the test pass for the wrong reason.

    Args:
        max_depth: How many plies of the opening to explore. A position with a single
            legal move first appears at three plies, after 1. e4 e5 2. Qh5.

    Returns:
        A `(board, uci)` pair, where `uci` is the only legal move in `board` for
        `board.turn`.

    Raises:
        AssertionError: If no such position exists within `max_depth` plies. Raise the
            depth rather than pasting a FEN that may be wrong.
    """
    seen = {chess.STARTING_FEN}
    frontier = deque([chess.Board()])
    for depth in range(max_depth + 1):
        following = deque()
        while frontier:
            board = frontier.popleft()
            legal = list(board.legal_moves)
            if len(legal) == 1:
                return board, legal[0].uci()
            if depth == max_depth:
                continue
            for move in legal:
                board.push(move)
                fen = board.fen()
                if fen not in seen:
                    seen.add(fen)
                    following.append(board.copy())
                board.pop()
        frontier = following

    raise AssertionError(
        f"no position with exactly one legal move within {max_depth} plies; raise max_depth"
    )
