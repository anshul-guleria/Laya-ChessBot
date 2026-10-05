"""Direction classification.

The dataset splits the two diagonals into separate classes, so getting the sign
convention backwards silently corrupts every bishop, rook and queen move.
"""

from __future__ import annotations

import chess
import pytest

from chessbot.engine import direction_of

CASES = [
    # Same rank: a1-a4 is a file move, not a rank move, so it is not here.
    ("a1b1", "horizontal"),
    ("h1g1", "horizontal"),
    ("d5a5", "horizontal"),
    # Same file, upwards or downwards.
    ("a1a4", "vertical"),
    ("a1a5", "vertical"),
    ("h1h7", "vertical"),
    ("c2c8", "vertical"),
    ("a4a1", "vertical"),
    # Equal signed deltas: travelling up-and-right, or down-and-left.
    ("a1b2", "diagonal_ne_sw"),
    ("a1h8", "diagonal_ne_sw"),
    ("h7g6", "diagonal_ne_sw"),
    ("b2a1", "diagonal_ne_sw"),
    # Opposite signed deltas: travelling up-and-left, or down-and-right.
    ("h1a8", "diagonal_nw_se"),
    ("h1g2", "diagonal_nw_se"),
    ("g2h1", "diagonal_nw_se"),
    ("e5f4", "diagonal_nw_se"),
]


@pytest.mark.parametrize(("uci", "expected"), CASES)
def test_direction_of(uci: str, expected: str) -> None:
    """Square pairs on a rank, a file or a diagonal map to their contract class.

    Args:
        uci: A UCI string from `CASES`, built with `Move.from_uci` so no board is
            needed -- classification is purely geometric.
        expected: The direction key the fine-tuned dataset expects.
    """
    assert direction_of(chess.Move.from_uci(uci)) == expected


def test_the_two_long_diagonals_are_different_classes() -> None:
    """a1-h8 and h1-a8 are the same length but lie on opposite diagonals."""
    assert direction_of(chess.Move.from_uci("a1h8")) == "diagonal_ne_sw"
    assert direction_of(chess.Move.from_uci("h1a8")) == "diagonal_nw_se"


def test_direction_classification_is_about_the_axis_not_the_piece() -> None:
    """A one-square step classifies like a full-length move along the same axis."""
    assert direction_of(chess.Move.from_uci("b1a2")) == "diagonal_nw_se"
    assert direction_of(chess.Move.from_uci("b1c2")) == "diagonal_ne_sw"


def test_knight_move_is_not_a_direction() -> None:
    """The direction stage is for sliders only; a knight move must be rejected."""
    with pytest.raises(ValueError, match="straight line or 45-degree"):
        direction_of(chess.Move.from_uci("b1c3"))
