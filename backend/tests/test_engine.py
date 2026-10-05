"""Board helpers: piece labels, destination labels, replay, parsing, game status."""

from __future__ import annotations

import chess
import pytest

from chessbot.engine import (
    board_from_moves,
    destination_label,
    game_status,
    legal_uci_moves,
    piece_symbol_of,
    uci_to_move,
)

# The black king stands on h8 so that e8 is empty and e7-e8 is a legal promotion.
PROMOTION_FEN = "7k/4P3/8/8/8/8/8/4K3 w - - 0 1"


def test_piece_symbol_of() -> None:
    """The symbol comes from the board, since a bare `chess.Move` carries no piece."""
    board = chess.Board()
    moves = {m.uci(): m for m in board.legal_moves}
    assert piece_symbol_of(board, moves["e2e4"]) == "P"
    assert piece_symbol_of(board, moves["b1c3"]) == "N"


def test_piece_symbol_of_rejects_a_hand_built_move() -> None:
    """A move whose origin square holds no piece did not come from a board."""
    with pytest.raises(ValueError, match="no piece type"):
        piece_symbol_of(chess.Board(), chess.Move(chess.A3, chess.A4))


def test_destination_label_is_a_plain_square() -> None:
    moves = {m.uci(): m for m in chess.Board().legal_moves}
    assert destination_label(moves["e2e4"]) == "e4"


def test_destination_label_carries_the_promotion_piece() -> None:
    moves = {m.uci(): m for m in chess.Board(PROMOTION_FEN).legal_moves}
    assert destination_label(moves["e7e8q"]) == "e8=Q"
    assert destination_label(moves["e7e8r"]) == "e8=R"
    assert destination_label(moves["e7e8b"]) == "e8=B"
    assert destination_label(moves["e7e8n"]) == "e8=N"


def test_capture_destination_has_no_suffix() -> None:
    moves = {m.uci(): m for m in chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").legal_moves}
    assert destination_label(moves["e4d5"]) == "d5"


def test_board_from_moves_replays() -> None:
    uci_list = ["e2e4", "e7e5", "g1f3"]
    reference = chess.Board()
    for uci in uci_list:
        reference.push_uci(uci)

    board = board_from_moves(uci_list)
    assert board.fen() == reference.fen()
    assert board.turn == chess.BLACK


def test_board_from_moves_on_an_empty_history() -> None:
    assert board_from_moves([]).fen() == chess.Board().fen()


def test_board_from_moves_rejects_an_illegal_move() -> None:
    with pytest.raises(ValueError, match="illegal move"):
        board_from_moves(["e2e5"])


def test_board_from_moves_rejects_garbage() -> None:
    """`push_uci` wants full UCI, not the `e4`-style shorthand a human would type."""
    with pytest.raises(ValueError, match="illegal move"):
        board_from_moves(["e4"])


def test_uci_to_move_accepts_a_legal_move() -> None:
    assert uci_to_move(chess.Board(), "e2e4").uci() == "e2e4"


def test_uci_to_move_rejects_an_illegal_move() -> None:
    with pytest.raises(ValueError, match="not legal"):
        uci_to_move(chess.Board(), "e2e5")


def test_uci_to_move_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        uci_to_move(chess.Board(), "not-a-move")


def test_legal_uci_moves_is_sorted_and_complete() -> None:
    moves = legal_uci_moves(chess.Board())
    assert moves == sorted(moves)
    assert len(moves) == 20
    assert "e2e4" in moves


def test_game_status_active() -> None:
    assert game_status(chess.Board()) == ("active", None, None)


def test_game_status_checkmate() -> None:
    """Fool's mate: black's queen mates, so black wins."""
    board = board_from_moves(["f2f3", "e7e5", "g2g4", "d8h4"])
    assert game_status(board) == ("black_won", "0-1", "checkmate")


def test_game_status_checkmate_for_white() -> None:
    """Scholar's mate: white's queen mates, so white wins."""
    board = board_from_moves(["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"])
    assert game_status(board) == ("white_won", "1-0", "checkmate")


def test_game_status_stalemate() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert game_status(board) == ("draw", "1/2-1/2", "stalemate")


def test_game_status_insufficient_material() -> None:
    assert game_status(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")) == (
        "draw",
        "1/2-1/2",
        "insufficient_material",
    )
