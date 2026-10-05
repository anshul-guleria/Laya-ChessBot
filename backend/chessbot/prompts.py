"""The exact text format the fine-tuned Laya chess model was trained on.

These strings are not free text and must never be reworded, re-indented, or
Unicode-normalized. The model was fine-tuned on prompts of exactly this shape, so a
changed string raises no error -- it silently degrades move quality.

Authoritative source
--------------------
`model/laya_finetuned_1000_after_finetuning.json` holds 1,000 held-out predictions,
each carrying the precise state text, instruction text, question id and option labels
that produced it. Every constant below was extracted from that file, and
`tests/test_prompts.py` re-checks them against it byte for byte.

`laya_chess_bot_move_guide.md` disagrees with the data in three places and must not
be used as the reference:

1. It describes three directions (`horizontal` / `vertical` / `diagonal`). The
   dataset uses four: the diagonals are split into `diagonal_nw_se` and
   `diagonal_ne_sw`.
2. It leaves the direction instruction text as an unfilled placeholder.
3. It omits the `Selected direction: <dir>.` clause that the destination stage
   carries for bishops, rooks and queens.

Following the guide would mis-classify every slider move and feed the model text it
never saw.

Every stage suffix below carries its own leading space. The prefix ends in a full
stop and the dataset has exactly one space before `Stage:`, so the space belongs to
the suffix -- putting it anywhere else concatenates into text the model never saw.
"""

from __future__ import annotations

import chess

from chessbot.engine import destination_label, direction_of

# --- Option labels ---------------------------------------------------------

# Display labels for the piece-selection stage, keyed by piece symbol.
PIECE_LABELS: dict[str, str] = {
    "P": "Pawn",
    "N": "Knight",
    "B": "Bishop",
    "R": "Rook",
    "Q": "Queen",
    "K": "King",
}

# The order piece options are offered in. Fixed so a given position always produces
# the same criteria dict, which keeps decision traces diffable between runs.
PIECE_ORDER: tuple[str, ...] = ("P", "N", "B", "R", "Q", "K")

# Direction keys and labels for the direction stage.
#
# The two diagonal labels contain U+2013 EN DASH, not an ASCII hyphen. That is what
# the dataset contains; substituting a hyphen would show the model a string it was
# never trained on. `tests/test_prompts.py` pins the codepoint.
DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("horizontal", "Horizontal"),
    ("vertical", "Vertical"),
    ("diagonal_nw_se", "Diagonal NW\u2013SE"),
    ("diagonal_ne_sw", "Diagonal NE\u2013SW"),
)

DIRECTION_KEYS: tuple[str, ...] = tuple(key for key, _ in DIRECTIONS)

# Bishops, rooks and queens get a direction stage; pawns, knights and kings do not.
SLIDING_PIECES: frozenset[str] = frozenset("BRQ")

# --- Question definitions --------------------------------------------------

# Keyed by question id, which is also the key the answer comes back under.
INSTRUCTIONS: dict[str, str] = {
    "piece_selection": "Choose the best chess piece for the move.",
    "direction_selection": "Choose the best direction for the chess piece.",
    "destination_selection": "Choose the best chess piece destination.",
}

QUESTION_PIECE = "piece_selection"
QUESTION_DIRECTION = "direction_selection"
QUESTION_DESTINATION = "destination_selection"

# --- State templates -------------------------------------------------------

# Held as `.format()` templates rather than pre-rendered strings so the test suite
# can turn them into regexes and match real corpus rows against them.

STATE_PREFIX_TEMPLATE = "Chess position. FEN: {fen}. Side to move: {side}."
PIECE_STATE_SUFFIX = " Stage: piece."
DIRECTION_STATE_SUFFIX = " Stage: direction. Selected piece: {piece}."
DESTINATION_STATE_SUFFIX = " Stage: destination. Selected piece: {piece}."
DESTINATION_DIRECTION_SUFFIX = " Selected direction: {direction}."


def side_text(board: chess.Board) -> str:
    """Return the lowercase name of the side to move.

    Args:
        board: The position to read.

    Returns:
        Either `"white"` or `"black"`, matching the wording used in the dataset.
    """
    return "white" if board.turn == chess.WHITE else "black"


def _prefix(board: chess.Board) -> str:
    """Render the shared `Chess position. FEN: ... Side to move: ...` prefix.

    Args:
        board: The position to read.

    Returns:
        The formatted prefix, without any stage suffix.
    """
    return STATE_PREFIX_TEMPLATE.format(fen=board.fen(), side=side_text(board))


def piece_state(board: chess.Board) -> str:
    """Return the state text for the piece-selection stage.

    Args:
        board: The position to read.

    Returns:
        The full state string, ending in `Stage: piece.`
    """
    return _prefix(board) + PIECE_STATE_SUFFIX


def direction_state(board: chess.Board, piece: str) -> str:
    """Return the state text for the direction-selection stage.

    Args:
        board: The position to read.
        piece: The piece symbol the model already chose, e.g. `"R"`.

    Returns:
        The full state string, ending in `Selected piece: <piece>.`
    """
    return _prefix(board) + DIRECTION_STATE_SUFFIX.format(piece=piece)


def destination_state(board: chess.Board, piece: str, direction: str | None) -> str:
    """Return the state text for the destination-selection stage.

    Args:
        board: The position to read.
        piece: The piece symbol chosen at the previous stage.
        direction: The direction key chosen at the previous stage, or `None` for
            pawns, knights and kings, which skip the direction stage.

    Returns:
        The full state string. For sliders it ends in `Selected direction:
        <direction>.`; for the others it does not.
    """
    state = _prefix(board) + DESTINATION_STATE_SUFFIX.format(piece=piece)
    if direction is not None:
        state += DESTINATION_DIRECTION_SUFFIX.format(direction=direction)
    return state


# --- Option builders -------------------------------------------------------


def piece_options(board: chess.Board, moves: list[chess.Move]) -> dict[str, str]:
    """Build the criteria for the piece-selection stage.

    The board is needed because `chess.Move` carries only the two squares and the
    promotion piece -- it does not record which piece is moving, so the piece type has
    to be read off `board` at `move.from_square`.

    Args:
        board: The position `moves` were generated from.
        moves: The legal moves to consider. Always a subset of `board.legal_moves`.

    Returns:
        A mapping of piece symbol to label covering only the piece types that have
        at least one legal move, in `PIECE_ORDER`.

    Raises:
        ValueError: If a move starts on a square `board` holds no piece on, which
            means the move was not generated from this board.
    """
    present = set()
    for move in moves:
        piece_type = board.piece_type_at(move.from_square)
        if piece_type is None:
            raise ValueError(
                f"move {move.uci()} starts on {chess.square_name(move.from_square)}, "
                f"which is empty in {board.fen()}"
            )
        present.add(chess.piece_symbol(piece_type).upper())
    return {symbol: PIECE_LABELS[symbol] for symbol in PIECE_ORDER if symbol in present}


def direction_options(moves: list[chess.Move]) -> dict[str, str]:
    """Build the criteria for the direction-selection stage.

    Args:
        moves: Legal moves of a single sliding piece.

    Returns:
        A mapping of direction key to label for the directions that actually occur
        among these moves, in `DIRECTIONS` order.

    Raises:
        ValueError: If a move is neither a straight line nor a 45-degree diagonal,
            which means a non-slider's move was passed in.
    """
    present = {direction_of(move) for move in moves}
    unknown = present - set(DIRECTION_KEYS)
    if unknown:
        raise ValueError(f"unclassifiable direction(s): {sorted(unknown)}")
    return {key: label for key, label in DIRECTIONS if key in present}


def destination_options(moves: list[chess.Move]) -> dict[str, str]:
    """Build the criteria for the destination-selection stage.

    Args:
        moves: The moves still in contention after the earlier stages.

    Returns:
        A mapping of destination label to the same label, deduplicated and sorted so
        the output is deterministic. Promotions carry the promotion piece, so a
        promotion appears as `"e8=Q"`.

    Two different rooks reaching the same square produce one label, not two: the
    question chain has no question that distinguishes them, so offering both would be
    offering the model a distinction it cannot make.
    """
    labels = {destination_label(move) for move in moves}
    return {label: label for label in sorted(labels)}
