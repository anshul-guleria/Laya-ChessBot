"""Pure `python-chess` helpers for reading and rebuilding a position.

Nothing here imports the model, the prompt contract or the web layer, so this module
stays importable without torch. These functions answer what the prompt builders ask
-- which direction is this move, what is its destination label, is this game over --
and they own the invariant that a move list is the single source of truth for a
position: a stored game is replayed, never trusted as a stored FEN.
"""

from __future__ import annotations

from collections.abc import Iterable

import chess

STATUS_ACTIVE = "active"
STATUS_WHITE_WON = "white_won"
STATUS_BLACK_WON = "black_won"
STATUS_DRAW = "draw"

RESULT_WHITE = "1-0"
RESULT_BLACK = "0-1"
RESULT_DRAW = "1/2-1/2"


def piece_symbol_of(board: chess.Board, move: chess.Move) -> str:
    """Return the uppercase piece symbol for a move.

    A `chess.Move` carries only squares, so the piece has to be read off the board the
    move belongs to. That is also what makes a forged move detectable: a move whose
    origin square holds nothing is rejected rather than guessed at.

    Args:
        board: The position `move` was generated from, before it was pushed.
        move: The move to identify, e.g. from `board.legal_moves`.

    Returns:
        One of `"P"`, `"N"`, `"B"`, `"R"`, `"Q"`, `"K"`.

    Raises:
        ValueError: If no piece stands on the move's origin square, meaning the move did
            not come from this board.
    """
    piece_type = board.piece_type_at(move.from_square)
    if piece_type is None:
        raise ValueError(
            f"move {move.uci()} has no piece type at {chess.square_name(move.from_square)}; "
            "it was not produced by this board"
        )
    return chess.piece_symbol(piece_type).upper()


def direction_of(move: chess.Move) -> str:
    """Classify a move's direction for the direction-selection stage.

    Four classes, matching the fine-tuned dataset. With `df` the change in file and
    `dr` the change in rank:

    - `dr == 0` -> `horizontal`, same rank
    - `df == 0` -> `vertical`, same file
    - `df == dr` -> `diagonal_ne_sw`, travelling up-and-right or down-and-left, as in
      a1 to b2
    - `df == -dr` -> `diagonal_nw_se`, travelling up-and-left or down-and-right, as in
      h1 to a8

    Args:
        move: The move to classify. Expected to belong to a sliding piece.

    Returns:
        One of `"horizontal"`, `"vertical"`, `"diagonal_nw_se"`,
        `"diagonal_ne_sw"`.

    Raises:
        ValueError: If the move is neither a straight line nor a 45-degree diagonal,
            which means a non-slider's move was passed in.
    """
    df = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dr = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)

    # The class names are the axes, not the direction of travel: a1 to b2 climbs toward
    # the north-east, but it sits on the north-east/south-west axis, so equal signed
    # deltas are `ne_sw` and opposite signed deltas are `nw_se`.
    if dr == 0:
        return "horizontal"
    if df == 0:
        return "vertical"
    if df == dr:
        return "diagonal_ne_sw"
    if df == -dr:
        return "diagonal_nw_se"
    raise ValueError(
        f"move {move.uci()} is not a straight line or 45-degree diagonal; "
        "only bishops, rooks and queens have a direction stage"
    )


def destination_label(move: chess.Move) -> str:
    """Return the destination label the model is offered for a move.

    Args:
        move: The move to label.

    Returns:
        A square name, suffixed with `=<piece>` for a promotion, e.g. `"e8=Q"`. This
        matches the form in the training data.
    """
    label = chess.square_name(move.to_square)
    if move.promotion is not None:
        label += "=" + chess.piece_symbol(move.promotion).upper()
    return label


def board_from_moves(moves: Iterable[str]) -> chess.Board:
    """Rebuild a board by replaying a UCI move list from the start position.

    This is how a stored game becomes a position. The move list is authoritative, so
    the board the player sees and the board the bot reasons about cannot drift apart.

    Args:
        moves: Full UCI move strings in order, e.g. `"e2e4"` rather than `"e4"`.

    Returns:
        The board after every move has been played.

    Raises:
        ValueError: If a move is unparseable or illegal in the position it is replayed
            into.
    """
    board = chess.Board()
    for uci in moves:
        try:
            board.push_uci(uci)
        except (ValueError, AssertionError) as exc:
            raise ValueError(f"illegal move {uci!r} in recorded game: {exc}") from exc
    return board


def uci_to_move(board: chess.Board, uci: str) -> chess.Move:
    """Parse a UCI string against a board and require that it be legal.

    The browser is not a trust boundary, so every move it sends is re-validated here.
    Parsing is done with `Move.from_uci` rather than `Board.parse_uci` because
    `parse_uci` reports an illegal move and a malformed string with the same exception,
    and the two need different messages.

    Args:
        board: The position the move must be legal in.
        uci: The move string, e.g. `"e2e4"` or `"e7e8q"`.

    Returns:
        The legal `chess.Move`.

    Raises:
        ValueError: If the string does not parse, or is not legal in `board`.
    """
    try:
        move = chess.Move.from_uci(uci)
    except (ValueError, chess.InvalidMoveError) as exc:
        raise ValueError(f"cannot parse move {uci!r}: {exc}") from exc
    if move not in board.legal_moves:
        raise ValueError(f"move {uci!r} is not legal in {board.fen()}")
    return move


def legal_uci_moves(board: chess.Board) -> list[str]:
    """Return every legal move for the side to move, as UCI, sorted.

    Args:
        board: The position to enumerate.

    Returns:
        Sorted UCI strings. The client receives this list so it and the server agree
        on legality by construction.
    """
    return sorted(move.uci() for move in board.legal_moves)


def game_status(board: chess.Board) -> tuple[str, str | None, str | None]:
    """Report the status, result and termination reason for a position.

    Args:
        board: The position to classify.

    Returns:
        A `(status, result, termination)` triple. `status` is one of `"active"`,
        `"white_won"`, `"black_won"`, `"draw"`. `result` is PGN notation (`"1-0"`,
        `"0-1"`, `"1/2-1/2"`) or `None` while the game is live. `termination` is
        python-chess's name for the reason, or `None`.

    Note:
        Checkmate is tested before the draw conditions because a mated position is also
        one where the side to move has no legal move, which several draw predicates
        would otherwise accept.
    """
    if board.is_checkmate():
        white_won = board.turn == chess.BLACK
        return (
            STATUS_WHITE_WON if white_won else STATUS_BLACK_WON,
            RESULT_WHITE if white_won else RESULT_BLACK,
            "checkmate",
        )
    if board.is_stalemate():
        return STATUS_DRAW, RESULT_DRAW, "stalemate"
    if board.is_insufficient_material():
        return STATUS_DRAW, RESULT_DRAW, "insufficient_material"
    if board.is_repetition(4):
        return STATUS_DRAW, RESULT_DRAW, "fivefold_repetition"
    if board.is_repetition(3):
        return STATUS_DRAW, RESULT_DRAW, "threefold_repetition"
    if board.is_seventyfive_moves():
        return STATUS_DRAW, RESULT_DRAW, "seventyfive_moves"
    if board.is_fifty_moves():
        return STATUS_DRAW, RESULT_DRAW, "fifty_move_rule"
    return STATUS_ACTIVE, None, None
