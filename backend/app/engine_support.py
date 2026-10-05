"""Narrow re-export of the `chessbot.engine` helpers the web layer needs.

Keeping this explicit means `app.*` never reaches into the library's internals, so the
library can grow or move without a scatter of import updates.
"""

from __future__ import annotations

from chessbot.engine import (
    RESULT_BLACK,
    RESULT_DRAW,
    RESULT_WHITE,
    STATUS_ACTIVE,
    STATUS_BLACK_WON,
    STATUS_DRAW,
    STATUS_WHITE_WON,
    board_from_moves,
    game_status,
    legal_uci_moves,
    uci_to_move,
)

__all__ = [
    "RESULT_BLACK",
    "RESULT_DRAW",
    "RESULT_WHITE",
    "STATUS_ACTIVE",
    "STATUS_BLACK_WON",
    "STATUS_DRAW",
    "STATUS_WHITE_WON",
    "board_from_moves",
    "game_status",
    "legal_uci_moves",
    "uci_to_move",
]
