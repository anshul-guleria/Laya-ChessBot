"""Guard the trained prompt contract against silent drift.

The strings in `chessbot.prompts` are not free text. They were fine-tuned into the
checkpoint, so a reworded string raises no error -- it just makes the bot worse.
These tests read the 1,000 real predictions in
`model/laya_finetuned_1000_after_finetuning.json` and assert that every string the
application would send appeared verbatim in the data the model was trained and
evaluated on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import chess
import pytest

from chessbot import prompts

PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "model" / "laya_finetuned_1000_after_finetuning.json"
)

# The stage each question id must have produced, keyed by question id. The
# destination stage carries two shapes in the data, so it is narrowed further below.
EXPECTED_SHAPE_BY_QUESTION = {
    "piece_selection": "piece",
    "direction_selection": "direction",
    "destination_selection": "destination",
}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    """The real prediction rows, each carrying its exact prompt text."""
    with PREDICTIONS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)["predictions"]


def template_to_regex(template: str) -> re.Pattern[str]:
    """Turn a `.format()` template into an anchored regex with named groups.

    A `{placeholder}` becomes a non-greedy capture, so any FEN or piece symbol
    matches, while the literal prose around it must match byte for byte.

    Args:
        template: A template string using `{name}` placeholders.

    Returns:
        An anchored compiled pattern.
    """
    parts = re.split(r"(\{[a-z_]+\})", template)
    body = "".join(
        rf"(?P<{token[1:-1]}>.+?)" if token.startswith("{") else re.escape(token) for token in parts
    )
    return re.compile(f"^{body}$")


def test_corpus_is_present(rows):
    assert len(rows) == 1000


def test_instruction_strings_match_corpus(rows):
    observed = {(row["question_id"], row["question"]["instructions"]) for row in rows}
    expected = {
        ("piece_selection", "Choose the best chess piece for the move."),
        ("direction_selection", "Choose the best direction for the chess piece."),
        ("destination_selection", "Choose the best chess piece destination."),
    }
    assert observed == expected
    for question_id, instructions in expected:
        assert prompts.INSTRUCTIONS[question_id] == instructions


def test_direction_keys_and_labels_match_corpus(rows):
    observed: dict[str, str] = {}
    for row in rows:
        if row["question_id"] == "direction_selection":
            observed.update(row["question"]["criteria"])

    assert observed == dict(prompts.DIRECTIONS)
    assert set(prompts.DIRECTION_KEYS) == {
        "horizontal",
        "vertical",
        "diagonal_nw_se",
        "diagonal_ne_sw",
    }


def test_diagonal_labels_use_en_dash(rows):
    """An ASCII hyphen is a silent accuracy loss, so pin the codepoint."""
    for key, label in prompts.DIRECTIONS:
        if "NW" in label or "NE" in label:
            assert "\u2013" in label, f"{key}: {label!r} must contain U+2013 EN DASH"
            assert "-" not in label, f"{key}: {label!r} must not contain an ASCII hyphen"

    corpus_labels = {
        label
        for row in rows
        if row["question_id"] == "direction_selection"
        for label in row["question"]["criteria"].values()
    }
    assert corpus_labels == {label for _, label in prompts.DIRECTIONS}


def test_piece_labels_match_corpus(rows):
    observed: dict[str, str] = {}
    for row in rows:
        if row["question_id"] == "piece_selection":
            observed.update(row["question"]["criteria"])
    assert observed == prompts.PIECE_LABELS
    assert list(prompts.PIECE_ORDER) == ["P", "N", "B", "R", "Q", "K"]


def test_every_corpus_state_matches_one_of_our_templates(rows):
    """Each of the 1,000 real states is one of the four shapes we can generate."""
    templates = {
        "piece": prompts.STATE_PREFIX_TEMPLATE + prompts.PIECE_STATE_SUFFIX,
        "direction": prompts.STATE_PREFIX_TEMPLATE + prompts.DIRECTION_STATE_SUFFIX,
        "destination": prompts.STATE_PREFIX_TEMPLATE + prompts.DESTINATION_STATE_SUFFIX,
        "destination_with_direction": (
            prompts.STATE_PREFIX_TEMPLATE
            + prompts.DESTINATION_STATE_SUFFIX
            + prompts.DESTINATION_DIRECTION_SUFFIX
        ),
    }
    patterns = {name: template_to_regex(text) for name, text in templates.items()}

    for row in rows:
        hits = {name for name, pattern in patterns.items() if pattern.match(row["state"])}
        assert hits, f"no template matched {row['state']!r}"

        # Name the shape from the data rather than from the match set: a slider
        # destination state matches the plain destination template too, so the hit set
        # alone cannot tell the two apart.
        expected = EXPECTED_SHAPE_BY_QUESTION[row["question_id"]]
        if "Selected direction:" in row["state"]:
            expected = "destination_with_direction"
        assert expected in hits, f"{row['state']!r} did not match the {expected} template"


def test_built_states_are_byte_exact():
    board = chess.Board()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    assert prompts.piece_state(board) == (
        f"Chess position. FEN: {fen}. Side to move: white. Stage: piece."
    )
    assert prompts.direction_state(board, "Q") == (
        f"Chess position. FEN: {fen}. Side to move: white. Stage: direction. Selected piece: Q."
    )
    assert prompts.destination_state(board, "P", None) == (
        f"Chess position. FEN: {fen}. Side to move: white. Stage: destination. Selected piece: P."
    )
    assert prompts.destination_state(board, "R", "horizontal") == (
        f"Chess position. FEN: {fen}. Side to move: white. "
        "Stage: destination. Selected piece: R. Selected direction: horizontal."
    )


def test_side_text_follows_the_turn():
    board = chess.Board()
    assert prompts.side_text(board) == "white"
    board.push_uci("e2e4")
    assert prompts.side_text(board) == "black"


def test_piece_options_lists_only_piece_types_with_legal_moves():
    board = chess.Board()
    options = prompts.piece_options(board, list(board.legal_moves))
    assert options == {"P": "Pawn", "N": "Knight"}
    assert list(options) == ["P", "N"]


def test_piece_options_omit_types_with_no_legal_move():
    """After 1.e4 only the pawn has a quiet move for some pieces; ordering must hold."""
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    options = prompts.piece_options(board, list(board.legal_moves))
    assert set(options) == {"P", "K"}
    assert list(options) == ["P", "K"]


def test_piece_options_reject_a_move_from_a_different_board():
    """`chess.Move` carries no piece type, so the board is the only source for it."""
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    stray = chess.Move.from_uci("e2e4")
    with pytest.raises(ValueError, match="is empty in"):
        prompts.piece_options(board, [stray])


def test_destination_options_are_deduplicated_and_labelled_by_themselves():
    board = chess.Board("4k3/8/8/8/8/8/8/QQ3K2 w - - 0 1")
    moves = [m for m in board.legal_moves if board.piece_type_at(m.from_square) == chess.QUEEN]
    options = prompts.destination_options(moves)
    assert all(key == label for key, label in options.items())
    assert list(options) == sorted(options)
    assert len(options) < len(moves), "both queens reach a2 and b2, so labels must dedupe"
    assert "c1" in options
