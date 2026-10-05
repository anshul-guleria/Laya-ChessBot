# Laya Chess Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A web app where a human plays chess against the fine-tuned Laya model, with the bot's stage-by-stage decision shown after every move.

**Architecture:** `backend/` is its own uv project containing a Django-free `chessbot/` library (prompts, agent wrapper, move decider) and a FastAPI `app/` web layer. `frontend/` is a Next.js App Router app that proxies `/api/*` to the backend so the browser sees one origin. Bot moves are produced synchronously inside the move request, with the blocking inference pushed onto the ASGI threadpool.

**Tech Stack:** Python 3.14, FastAPI, SQLAlchemy 2, `python-chess`, `laya` 0.3.28, torch 2.14 (CPU), pytest, ruff. Next.js, React, TypeScript, `react-chessboard`.

**Spec:** `docs/superpowers/specs/2026-10-06-laya-chessbot-design.md`

## Global Constraints

- **Never commit.** The user has explicitly forbidden git commits for this project. There is no commit step in any task. Leave all changes uncommitted in the working tree.
- Repo root is `C:\Users\anshu\Desktop\Laya-ChessBot`. All paths below are relative to it.
- **Prompt strings are load-bearing.** Never reword, normalize Unicode, or reflow them. `Diagonal NW–SE` uses U+2013 EN DASH, not an ASCII hyphen. `laya_chess_bot_move_guide.md` is wrong about directions, instructions, and the destination state suffix; the spec §4 and `model/laya_finetuned_1000_after_finetuning.json` are authoritative.
- Python 3.14. The venv lives at `backend/.venv`. Run Python commands as `uv run <cmd>` from `backend/`.
- No GPU on this machine. `LAYA_DEVICE` defaults to `cpu`.
- Docstrings on every module, public class and public function, **Google style** (`Args:`, `Returns:`, `Raises:`). Comments only for reasoning the code cannot express. No narration, no commented-out code.
- Two logger trees: `chessbot.*` for the library, `app.*` for the web layer. Never `print()`.
- Every backend test must run without loading the 842 MB checkpoint. Tests inject a `FakeAgent`.
- Ruff, line length 100.

## Verified API Reference

Confirmed by reading the `laya` 0.3.28 wheel source, not from memory. Do not re-derive these.

- `laya.Agent(model_id_or_path, device=None, ...)` loads a local checkpoint directory.
- `Agent.predict` is an alias for `Agent.system_one` (`laya/agent.py:1970`).
- Questions dict: `{"<qid>": {"type": "choice", "instructions": str, "criteria": {key: label}}}`.
- Returns `{"model": ..., "answers": {"<qid>": {"type", "choice", "probabilities": {key: prob}, "confidence", "answer_confidence", "action"}}, "usage": {...}}`.
- `Agent.__init__` already calls `_fix_tokenizer_config` (`laya/agent.py:609`). Do not call it manually.
- **Raises `ValueError`** when a question's option markers do not fit the checkpoint's `max_len=1024` / `head_max_len=256` budget. `predict_choice` must absorb this and degrade (Task 4).

---

## Parallelization

Tasks within a wave have no dependency on each other and may be dispatched to parallel subagents. A wave is a hard barrier.

| Wave | Tasks | Gate before moving on |
| --- | --- | --- |
| A | 1 | Toolchain works |
| B | 2, 3, 4, 9, 12 | Independent units exist: core helpers, agent wrapper, frontend scaffold, doc correction |
| C | 5, 6, 10, 11 | Core library complete; frontend components have their types |
| D | 7 | API is wired |
| E | 8, 13 | Everything verified |

Tasks 10 and 11 run in parallel but must not touch the same files. Task 10 owns `components/Board.tsx`, `app/games/[id]/page.tsx`. Task 11 owns `components/DecisionTrace.tsx`, `components/MoveList.tsx`, `app/page.tsx`, `app/globals.css`.

Task 4 writes `tests/support.py`, which Tasks 5 and 8 import. Wave B dispatches Task 4 in the same wave as 2 and 3 but they have no mutual dependency; Waves C and E are where the import matters, so both barriers are sufficient.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `backend/chessbot/prompts.py` | The trained prompt contract: state templates, instructions, option labels, option builders. Zero I/O. |
| `backend/chessbot/engine.py` | Pure `python-chess` helpers: direction classification, destination labels, history replay, status. |
| `backend/chessbot/agent.py` | `ChoiceResult`, `Agent` protocol, `LocalLayaAgent`. The only module that imports `laya`. |
| `backend/chessbot/decider.py` | The three-stage chain. `(agent, board) -> Decision`. |
| `backend/app/config.py` | Environment-backed settings. |
| `backend/app/db.py` | SQLAlchemy engine, session factory, `Base`, `get_db`. |
| `backend/app/models.py` | `Game` ORM model. |
| `backend/app/schemas.py` | Pydantic request/response models and `build_game_state`. |
| `backend/app/engine_support.py` | Narrow re-export of the `chessbot.engine` helpers `app` needs. |
| `backend/app/state.py` | Process-wide holder for the loaded agent. |
| `backend/app/deps.py` | `get_agent`. |
| `backend/app/main.py` | `create_app()`, lifespan, logging setup, router registration. |
| `backend/app/routers/games.py` | Game endpoints. |
| `backend/app/routers/health.py` | Health endpoint. |
| `backend/tests/support.py` | `FakeAgent`. Imported by other test modules. |
| `backend/tests/conftest.py` | Shared fixtures. |
| `frontend/lib/types.ts` | TypeScript mirror of the API response shapes. |
| `frontend/lib/api.ts` | Typed fetch wrappers. |
| `frontend/components/*.tsx` | Presentational pieces. |
| `frontend/app/*` | Routes. |

`chessbot.engine` does not import `chessbot.prompts`; the dependency runs one way, `prompts` → `engine`.

---

### Task 1: Backend toolchain

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/.python-version`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `backend/.venv` where `uv run pytest` and `uv run ruff` run, and `import chess`, `import fastapi`, `import laya`, `import sqlalchemy`, `import torch` all succeed. `pythonpath = ["."]` in the pytest config is what lets later tasks `import chessbot` and `import app` without installing the backend as a package.

- [ ] **Step 1: Write `backend/pyproject.toml`**

Replace the generated file with:

```toml
[project]
name = "laya-chessbot-backend"
version = "0.1.0"
description = "FastAPI backend for the Laya chess bot."
requires-python = ">=3.14"
dependencies = [
    "fastapi>=0.121",
    "uvicorn[standard]>=0.40",
    "sqlalchemy>=2.1",
    "pydantic>=2.13",
    "pydantic-settings>=2.13",
    "python-chess>=1.999",
    "laya>=0.3.28",
    "torch>=2.14",
]

[dependency-groups]
dev = [
    "pytest>=9.0",
    "httpx>=0.28",
    "ruff>=0.15",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- [ ] **Step 2: Create the venv and install**

Run from `backend/`:
```bash
uv sync
```

Expect a large download for torch. It may take several minutes.

- [ ] **Step 3: Write `backend/.python-version`**

The single line `3.14`.

- [ ] **Step 4: Write `backend/tests/__init__.py`**

Empty file. It must exist so `from tests.support import FakeAgent` resolves.

- [ ] **Step 5: Write `backend/tests/conftest.py`**

```python
"""Shared pytest fixtures.

Nothing here loads the real checkpoint. Tests that need the model substitute `FakeAgent`
through the `get_agent` dependency override. Task 8 appends the `agent`, `client` and
`seed_game` fixtures to this file, once `app.main` exists.
"""

from __future__ import annotations

import chess
import pytest


@pytest.fixture
def start_board() -> chess.Board:
    """Return a fresh board at the standard starting position.

    Returns:
        A `chess.Board` with no moves played.
    """
    return chess.Board()
```

- [ ] **Step 6: Update `.gitignore`**

Replace the file contents with:

```
laya_finetuned_typed_decisions/
.env
.env.local

.venv/
*.db
*.db-journal

node_modules/
.next/
out/
```

`.venv/` and `*.db` carry no leading slash, so they match at any depth — covering `backend/.venv` and `backend/app/chessbot.db`.

- [ ] **Step 7: Verify the toolchain**

Run from `backend/`:
```bash
uv run python -c "import chess, fastapi, laya, sqlalchemy, torch; print('imports ok')"
uv run pytest
uv run ruff --version
```

Expected: `imports ok`, a ruff version string, and pytest reporting no tests collected. That last one exits with code 5, which is expected at this point and is not a task failure.

---

### Task 2: `chessbot/prompts.py`

**Files:**
- Create: `backend/chessbot/__init__.py`
- Create: `backend/chessbot/prompts.py`
- Create: `backend/tests/test_prompts.py`

**Interfaces:**
- Consumes: `python-chess` for `chess.Board` and `chess.Move`; `chessbot.engine.direction_of` and `chessbot.engine.destination_label`, both created in Task 3. Until Task 3 lands, import them inside the function bodies; Task 3's Step 5 moves them to module level.
- Produces:
  - Constants `STATE_PREFIX_TEMPLATE`, `PIECE_STATE_SUFFIX`, `DIRECTION_STATE_SUFFIX`, `DESTINATION_STATE_SUFFIX`, `DESTINATION_DIRECTION_SUFFIX`, `INSTRUCTIONS`, `PIECE_LABELS`, `PIECE_ORDER`, `DIRECTIONS`, `DIRECTION_KEYS`, `SLIDING_PIECES`, `QUESTION_PIECE`, `QUESTION_DIRECTION`, `QUESTION_DESTINATION`
  - `side_text(board: chess.Board) -> str`
  - `piece_state(board: chess.Board) -> str`
  - `direction_state(board: chess.Board, piece: str) -> str`
  - `destination_state(board: chess.Board, piece: str, direction: str | None) -> str`
  - `piece_options(board: chess.Board, moves: list[chess.Move]) -> dict[str, str]`
  - `direction_options(moves: list[chess.Move]) -> dict[str, str]`
  - `destination_options(moves: list[chess.Move]) -> dict[str, str]`

- [ ] **Step 1: Create `backend/chessbot/__init__.py`**

Empty file.

- [ ] **Step 2: Write the failing test `backend/tests/test_prompts.py`**

```python
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
    Path(__file__).resolve().parents[2]
    / "model"
    / "laya_finetuned_1000_after_finetuning.json"
)


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
        rf"(?P<{token[1:-1]}>.+?)" if token.startswith("{") else re.escape(token)
        for token in parts
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

        # A plain destination state is also matched by the longer template's prefix,
        # so name the shape from the data rather than from the match set.
        if row["question_id"] == "direction_selection":
            assert "direction" in hits
        elif "Selected direction:" in row["state"]:
            assert "destination_with_direction" in hits
        else:
            assert "destination" in hits


def test_built_states_are_byte_exact():
    board = chess.Board()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    assert prompts.piece_state(board) == (
        f"Chess position. FEN: {fen}. Side to move: white. Stage: piece."
    )
    assert prompts.direction_state(board, "Q") == (
        f"Chess position. FEN: {fen}. Side to move: white. "
        "Stage: direction. Selected piece: Q."
    )
    assert prompts.destination_state(board, "P", None) == (
        f"Chess position. FEN: {fen}. Side to move: white. "
        "Stage: destination. Selected piece: P."
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
    options = prompts.piece_options(list(board.legal_moves))
    assert options == {"P": "Pawn", "N": "Knight"}
    assert list(options) == ["P", "N"]


def test_piece_options_omit_types_with_no_legal_move():
    """After 1.e4 only the pawn has a quiet move for some pieces; ordering must hold."""
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    options = prompts.piece_options(list(board.legal_moves))
    assert set(options) == {"P", "K"}
    assert list(options) == ["P", "K"]


def test_destination_options_are_deduplicated_and_labelled_by_themselves():
    board = chess.Board("4k3/8/8/8/8/8/8/QQ3K2 w - - 0 1")
    moves = [m for m in board.legal_moves if m.piece_type == chess.QUEEN]
    options = prompts.destination_options(moves)
    assert all(key == label for key, label in options.items())
    assert list(options) == sorted(options)
    assert len(options) < len(moves), "a1 and h1 both reach c1, so labels must dedupe"
    assert "c1" in options
```

- [ ] **Step 3: Run the test to verify it fails**

Run from `backend/`:
```bash
uv run pytest tests/test_prompts.py
```

Expected: collection error, `ModuleNotFoundError: No module named 'chessbot'`.

- [ ] **Step 4: Write `backend/chessbot/prompts.py`**

```python
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
"""

from __future__ import annotations

import chess

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
PIECE_STATE_SUFFIX = "Stage: piece."
DIRECTION_STATE_SUFFIX = "Stage: direction. Selected piece: {piece}."
DESTINATION_STATE_SUFFIX = "Stage: destination. Selected piece: {piece}."
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

    Args:
        moves: The legal moves to consider. Always a subset of `board.legal_moves`.

    Returns:
        A mapping of piece symbol to label covering only the piece types that have
        at least one legal move, in `PIECE_ORDER`.
    """
    present = {chess.piece_symbol(move.piece_type).upper() for move in moves}
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
    from chessbot.engine import direction_of

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
    from chessbot.engine import destination_label

    labels = {destination_label(move) for move in moves}
    return {label: label for label in sorted(labels)}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run from `backend/`:
```bash
uv run pytest tests/test_prompts.py
```

Expected: all tests pass, including the one that matches all 1,000 real corpus states.

- [ ] **Step 6: Lint**

Run from `backend/`:
```bash
uv run ruff check chessbot tests
uv run ruff format chessbot tests
```

The two function-local `from chessbot.engine import ...` statements are deliberate for now: `engine.py` does not exist yet. Task 3 Step 5 moves them to module level.

---

### Task 3: `chessbot/engine.py`

**Files:**
- Create: `backend/chessbot/engine.py`
- Create: `backend/tests/test_directions.py`
- Create: `backend/tests/test_engine.py`
- Modify: `backend/chessbot/prompts.py` (hoist the deferred imports)

**Interfaces:**
- Consumes: nothing from `prompts.py`. This module is deliberately independent of it.
- Produces:
  - `piece_symbol_of(board: chess.Board, move: chess.Move) -> str`
  - `direction_of(move: chess.Move) -> str`
  - `destination_label(move: chess.Move) -> str`
  - `board_from_moves(moves: Iterable[str]) -> chess.Board`
  - `uci_to_move(board: chess.Board, uci: str) -> chess.Move`
  - `legal_uci_moves(board: chess.Board) -> list[str]`
  - `game_status(board: chess.Board) -> tuple[str, str | None, str | None]`

- [ ] **Step 1: Write the failing test `backend/tests/test_directions.py`**

```python
"""Direction classification.

The dataset splits the two diagonals into separate classes, so getting the sign
convention backwards silently corrupts every bishop, rook and queen move.
"""

from __future__ import annotations

import chess
import pytest

from chessbot.engine import direction_of

CASES = [
    # Same rank.
    ("a1a4", "horizontal"),
    ("h1g1", "horizontal"),
    ("d5a5", "horizontal"),
    # Same file.
    ("a1a5", "vertical"),
    ("h1h7", "vertical"),
    ("c2c8", "vertical"),
    # Equal signed deltas: travelling up-and-right, or down-and-left.
    ("a1b2", "diagonal_ne_sw"),
    ("a1h8", "diagonal_ne_sw"),
    ("h7g6", "diagonal_ne_sw"),
    ("b2a1", "diagonal_ne_sw"),
    # Opposite signed deltas: travelling up-and-left, or down-and-right.
    ("h1a8", "diagonal_nw_se"),
    ("h1g2", "diagonal_nw_se"),
    ("g2h1", "diagonal_nw_se"),
    ("e5d4", "diagonal_nw_se"),
]


@pytest.mark.parametrize(("uci", "expected"), CASES)
def test_direction_of(uci: str, expected: str):
    assert direction_of(chess.Move.from_uci(uci)) == expected


def test_the_two_long_diagonals_are_different_classes():
    assert direction_of(chess.Move.from_uci("a1h8")) == "diagonal_ne_sw"
    assert direction_of(chess.Move.from_uci("h1a8")) == "diagonal_nw_se"


def test_knight_move_is_not_a_direction():
    """The direction stage is for sliders only; a knight move must be rejected."""
    with pytest.raises(ValueError, match="straight line or 45-degree"):
        direction_of(chess.Move.from_uci("b1c3"))
```

- [ ] **Step 2: Write the failing test `backend/tests/test_engine.py`**

```python
"""Board helpers: labels, replay, move parsing, game status."""

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


def test_piece_symbol_of():
    moves = {m.uci(): m for m in chess.Board().legal_moves}
    assert piece_symbol_of(moves["e2e4"]) == "P"
    assert piece_symbol_of(moves["b1c3"]) == "N"


def test_piece_symbol_of_rejects_a_hand_built_move():
    """A move with no piece type did not come from a board."""
    with pytest.raises(ValueError, match="no piece type"):
        piece_symbol_of(chess.Move(chess.A1, chess.A2))


def test_destination_label_is_a_plain_square():
    moves = {m.uci(): m for m in chess.Board().legal_moves}
    assert destination_label(moves["e2e4"]) == "e4"


def test_destination_label_carries_the_promotion_piece():
    moves = {m.uci(): m for m in chess.Board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1").legal_moves}
    assert destination_label(moves["e7e8q"]) == "e8=Q"
    assert destination_label(moves["e7e8r"]) == "e8=R"
    assert destination_label(moves["e7e8b"]) == "e8=B"
    assert destination_label(moves["e7e8n"]) == "e8=N"


def test_capture_destination_has_no_suffix():
    moves = {m.uci(): m for m in chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").legal_moves}
    assert destination_label(moves["e4d5"]) == "d5"


def test_board_from_moves_replays():
    uci_list = ["e2e4", "e7e5", "g1f3"]
    reference = chess.Board()
    for uci in uci_list:
        reference.push_uci(uci)

    board = board_from_moves(uci_list)
    assert board.fen() == reference.fen()
    assert board.turn == chess.WHITE


def test_board_from_moves_on_an_empty_history():
    assert board_from_moves([]).fen() == chess.Board().fen()


def test_board_from_moves_rejects_an_illegal_move():
    with pytest.raises(ValueError, match="illegal move"):
        board_from_moves(["e2e5"])


def test_uci_to_move_accepts_a_legal_move():
    assert uci_to_move(chess.Board(), "e2e4").uci() == "e2e4"


def test_uci_to_move_rejects_an_illegal_move():
    with pytest.raises(ValueError, match="not legal"):
        uci_to_move(chess.Board(), "e2e5")


def test_uci_to_move_rejects_garbage():
    with pytest.raises(ValueError, match="cannot parse"):
        uci_to_move(chess.Board(), "not-a-move")


def test_legal_uci_moves_is_sorted_and_complete():
    moves = legal_uci_moves(chess.Board())
    assert moves == sorted(moves)
    assert len(moves) == 20
    assert "e2e4" in moves


def test_game_status_active():
    assert game_status(chess.Board()) == ("active", None, None)


def test_game_status_checkmate():
    board = board_from_moves(["f3", "e5", "g4", "Qh4"])
    assert game_status(board) == ("black_won", "0-1", "checkmate")


def test_game_status_checkmate_for_white():
    board = board_from_moves(["f3", "e5", "g4", "Qh4", "Nc6", "Bc5", "Nf6"])
    status, _, _ = game_status(board)
    assert status == "black_won"


def test_game_status_stalemate():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert game_status(board) == ("draw", "1/2-1/2", "stalemate")


def test_game_status_insufficient_material():
    assert game_status(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")) == (
        "draw",
        "1/2-1/2",
        "insufficient_material",
    )
```

- [ ] **Step 3: Run the tests to verify they fail**

Run from `backend/`:
```bash
uv run pytest tests/test_directions.py tests/test_engine.py
```

Expected: collection error, `ModuleNotFoundError: No module named 'chessbot.engine'`.

- [ ] **Step 4: Write `backend/chessbot/engine.py`**

```python
"""Pure `python-chess` helpers for reading and rebuilding a position.

Nothing here imports the model or the web layer. These functions answer what the
prompt builders ask -- which direction is this move, what is its destination label,
is this game over -- and own the invariant that the move list is the single source
of truth for a position.
"""

from __future__ import annotations

from collections.abc import Iterable

import chess

STATUS_ACTIVE = "active"
STATUS_WHITE_WON = "white_won"
STATUS_BLACK_WON = "black_won"
STATUS_DRAW = "draw"


def piece_symbol_of(move: chess.Move) -> str:
    """Return the uppercase piece symbol for a move.

    Args:
        move: A move from `board.legal_moves` or `board.parse_uci`, both of which
            populate `piece_type`.

    Returns:
        One of `"P"`, `"N"`, `"B"`, `"R"`, `"Q"`, `"K"`.

    Raises:
        ValueError: If the move carries no piece type, meaning it was not produced by
            a board.
    """
    if move.piece_type is None:
        raise ValueError(f"move {move} has no piece type; it was not produced by a board")
    return chess.piece_symbol(move.piece_type).upper()


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
        moves: UCI move strings in order.

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

    Args:
        board: The position the move must be legal in.
        uci: The move string, e.g. `"e2e4"` or `"e7e8q"`.

    Returns:
        The legal `chess.Move`.

    Raises:
        ValueError: If the string does not parse, or is not legal in `board`.
    """
    try:
        move = board.parse_uci(uci)
    except ValueError as exc:
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
        Checkmate is tested before the draw conditions because a mated position is
        also one where the side to move has no legal move.
    """
    if board.is_checkmate():
        white_won = board.turn == chess.BLACK
        return (
            STATUS_WHITE_WON if white_won else STATUS_BLACK_WON,
            "1-0" if white_won else "0-1",
            "checkmate",
        )
    if board.is_stalemate():
        return STATUS_DRAW, "1/2-1/2", "stalemate"
    if board.is_insufficient_material():
        return STATUS_DRAW, "1/2-1/2", "insufficient_material"
    if board.is_repetition(4):
        return STATUS_DRAW, "1/2-1/2", "fivefold_repetition"
    if board.is_repetition(3):
        return STATUS_DRAW, "1/2-1/2", "threefold_repetition"
    if board.is_seventyfive_moves():
        return STATUS_DRAW, "1/2-1/2", "seventyfive_moves"
    if board.is_fifty_moves():
        return STATUS_DRAW, "1/2-1/2", "fifty_move_rule"
    return STATUS_ACTIVE, None, None
```

- [ ] **Step 5: Hoist the deferred imports in `prompts.py` to module level**

Now that `engine.py` exists, replace this in `prompts.py`:

```python
import chess
```

with:

```python
import chess

from chessbot.engine import destination_label, direction_of
```

Then delete the two function-local import lines inside `direction_options` and `destination_options`:

```python
    from chessbot.engine import direction_of
```

```python
    from chessbot.engine import destination_label
```

There is no circular import: `engine.py` does not import `prompts.py`.

- [ ] **Step 6: Run the whole backend suite**

Run from `backend/`:
```bash
uv run pytest
```

Expected: `test_prompts.py`, `test_directions.py` and `test_engine.py` all pass.

- [ ] **Step 7: Lint**

Run from `backend/`:
```bash
uv run ruff check chessbot tests
uv run ruff format chessbot tests
```

---

### Task 4: `chessbot/agent.py` and `tests/support.py`

**Files:**
- Create: `backend/chessbot/agent.py`
- Create: `backend/tests/support.py`
- Create: `backend/tests/test_agent.py`

**Interfaces:**
- Consumes: nothing from the rest of `chessbot`. This is the only module that imports `laya`, and it does so inside `LocalLayaAgent.__init__` so that importing the module does not pull in torch.
- Produces:
  - `AgentLoadError(RuntimeError)`
  - `ChoiceResult` frozen dataclass: `choice: str`, `probabilities: dict[str, float]`, `latency_ms: float`
  - `Agent` runtime-checkable `Protocol` with `predict_choice(state: str, question_id: str, instructions: str, options: dict[str, str]) -> ChoiceResult`
  - `LocalLayaAgent(model_dir: str | os.PathLike[str], device: str = "cpu", *, load: bool = True)` implementing `Agent`. `load=False` is the test seam: it skips the checkpoint load so the suite never pays for 842 MB.
  - `tests.support.FakeAgent` dataclass with fields `script: dict[str, str]`, `probabilities: dict[str, dict[str, float]]`, `calls: list[tuple[str, str, str, dict[str, str]]]`, plus properties `question_ids` and `call_count`

Tasks 5 and 8 import `FakeAgent` from `tests.support`. Task 7 imports `LocalLayaAgent` and `AgentLoadError`.

- [ ] **Step 1: Write the failing test `backend/tests/test_agent.py`**

```python
"""The agent wrapper's non-model behaviour: normalisation, fallback, serialisation.

No test here loads the checkpoint. `LocalLayaAgent` is exercised by replacing its
internal laya object after construction, which is the seam that keeps the rest of the
suite fast.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from chessbot.agent import ChoiceResult, LocalLayaAgent

STUB_RESULT = {
    "model": "laya-rl-agent",
    "answers": {
        "piece_selection": {
            "type": "choice",
            "choice": "N",
            "probabilities": {"P": 0.4, "N": 0.6},
        }
    },
    "usage": {},
}


class StubLaya:
    """Stand-in for `laya.Agent`, recording the calls it receives."""

    def __init__(self, result=None, raises=None, delay=0.0):
        self.result = STUB_RESULT if result is None else result
        self.raises = raises
        self.delay = delay
        self.calls: list[tuple[str, dict]] = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.delay:
            time.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        return self.result


def make_agent(stub: StubLaya) -> LocalLayaAgent:
    """Build an agent with a stub in place of the loaded checkpoint.

    `load=False` is what keeps this test suite fast: the constructor skips the 842 MB load
    and the internal object is swapped for a stub.

    Args:
        stub: The stand-in to install.

    Returns:
        A `LocalLayaAgent` whose `predict` goes to `stub`.
    """
    agent = LocalLayaAgent(model_dir="unused", device="cpu", load=False)
    agent._laya = stub
    return agent


def test_predict_choice_returns_the_choice_and_probabilities():
    agent = make_agent(StubLaya())
    result = agent.predict_choice("state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight"})
    assert isinstance(result, ChoiceResult)
    assert result.choice == "N"
    assert result.latency_ms > 0


def test_predict_choice_sends_the_expected_payload():
    stub = StubLaya()
    agent = make_agent(stub)
    agent.predict_choice("state text", "piece_selection", "Choose.", {"P": "Pawn"})
    assert stub.calls == [
        (
            "state text",
            {
                "piece_selection": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": {"P": "Pawn"},
                }
            },
        )
    ]


def test_missing_probability_keys_are_filled_with_zero():
    agent = make_agent(StubLaya())
    result = agent.predict_choice(
        "state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight", "Q": "Queen"}
    )
    assert set(result.probabilities) == {"P", "N", "Q"}
    assert result.probabilities["Q"] == 0.0


def test_probabilities_are_renormalised_over_the_offered_options():
    """laya rounds to four decimals, so the values rarely sum to exactly 1."""
    stub = StubLaya(
        result={
            "answers": {
                "piece_selection": {"choice": "P", "probabilities": {"P": 0.3333, "N": 0.3333}}
            }
        }
    )
    agent = make_agent(stub)
    result = agent.predict_choice("state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight"})
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_all_zero_probabilities_fall_back_to_uniform():
    stub = StubLaya(
        result={"answers": {"piece_selection": {"choice": "P", "probabilities": {"P": 0.0, "N": 0.0}}}}
    )
    agent = make_agent(stub)
    result = agent.predict_choice("state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight"})
    assert result.probabilities == {"P": 0.5, "N": 0.5}


def test_a_choice_the_model_invents_is_replaced_by_the_best_offered_option():
    agent = make_agent(
        StubLaya(
            result={
                "answers": {
                    "piece_selection": {"choice": "Z", "probabilities": {"P": 0.3, "N": 0.7}}
                }
            }
        )
    )
    result = agent.predict_choice("state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight"})
    assert result.choice == "N"
    assert result.choice in {"P", "N"}


def test_token_budget_overflow_degrades_to_the_first_option():
    """Laya raises when a question's options overflow its token budget.

    Losing the move is worse than playing a fallback, so the wrapper absorbs it.
    """
    agent = make_agent(StubLaya(raises=ValueError("only 3 of its 40 option markers fit")))
    result = agent.predict_choice("state", "destination_selection", "Choose.", {"e4": "e4", "d4": "d4"})
    assert result.choice == "e4"
    assert result.probabilities == {"e4": 1.0, "d4": 0.0}


def test_other_model_errors_propagate():
    """Only the token-budget ValueError is absorbed; real failures must surface."""
    agent = make_agent(StubLaya(raises=RuntimeError("out of memory")))
    with pytest.raises(RuntimeError, match="out of memory"):
        agent.predict_choice("state", "piece_selection", "Choose.", {"P": "Pawn"})


def test_predict_rejects_an_empty_option_set():
    agent = make_agent(StubLaya())
    with pytest.raises(ValueError, match="no options"):
        agent.predict_choice("state", "piece_selection", "Choose.", {})


def test_predict_is_serialised_so_two_moves_never_overlap():
    """The ASGI threadpool can run two moves at once; one torch module must not."""
    overlap_sizes: list[int] = []
    state = {"active": 0}

    class OverlapStub(StubLaya):
        def predict(self, state_text, questions):
            state["active"] += 1
            overlap_sizes.append(state["active"])
            time.sleep(0.02)
            state["active"] -= 1
            return STUB_RESULT

    agent = make_agent(OverlapStub())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda _: agent.predict_choice("s", "piece_selection", "c", {"P": "Pawn"}),
                range(4),
            )
        )

    assert max(overlap_sizes) == 1, f"predict calls overlapped: {overlap_sizes}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `backend/`:
```bash
uv run pytest tests/test_agent.py
```

Expected: collection error, `ModuleNotFoundError: No module named 'chessbot.agent'`.

- [ ] **Step 3: Write `backend/chessbot/agent.py`**

```python
"""The seam between the move decider and `laya`.

Everything above this module talks to the `Agent` protocol, never to `laya` directly.
That keeps `laya` and `torch` out of the import path of the pure logic in
`decider.py`, and lets the test suite substitute a fake so no test has to load the
842 MB checkpoint.

`laya` is imported inside `LocalLayaAgent.__init__` rather than at module scope, so
that `import chessbot.agent` stays cheap for the pure-logic tests and for the web
layer's error paths.

`LocalLayaAgent` also enforces the invariant the decider relies on: a returned choice
is always one of the options that were offered.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class AgentLoadError(RuntimeError):
    """Raised when the Laya checkpoint cannot be loaded."""


@dataclass(frozen=True)
class ChoiceResult:
    """One multiple-choice answer from the model.

    Attributes:
        choice: The selected option key. Guaranteed to be one of the keys offered.
        probabilities: Probability per offered key, renormalised to sum to 1.
        latency_ms: Wall-clock duration of the underlying `predict` call.
    """

    choice: str
    probabilities: dict[str, float]
    latency_ms: float


@runtime_checkable
class Agent(Protocol):
    """The interface `decider.decide_move` depends on."""

    def predict_choice(
        self, state: str, question_id: str, instructions: str, options: dict[str, str]
    ) -> ChoiceResult:
        """Ask one multiple-choice question and return the model's answer.

        Args:
            state: The state text describing the position and the current stage.
            question_id: The key the answer comes back under.
            instructions: The instruction text for this stage.
            options: Mapping of option key to the label shown to the model. Must be
                non-empty; its keys are the only legal answers.

        Returns:
            A `ChoiceResult` whose `choice` is one of `options`.
        """
        ...


class LocalLayaAgent:
    """Runs the fine-tuned Laya checkpoint that lives on this machine.

    One instance is shared by the whole application. `predict` is serialised behind a
    lock because the ASGI threadpool may run two moves at once, and one torch module
    entered from two threads is not a safe assumption.
    """

    def __init__(
        self, model_dir: str | os.PathLike[str], device: str = "cpu", *, load: bool = True
    ) -> None:
        """Load the checkpoint.

        Args:
            model_dir: Path to the fine-tuned checkpoint directory.
            device: Torch device string. `"cpu"` unless a GPU is present.
            load: When `False`, skip loading entirely and leave the wrapper empty. Only
                the tests use it; it is how they substitute a stub without paying for the
                real load.

        Raises:
            AgentLoadError: If `laya` is not importable, or the checkpoint cannot be
                loaded. Callers are expected to catch this and continue in a degraded
                state rather than refuse to boot.
        """
        self.model_dir = os.fspath(model_dir)
        self.device = device
        self._lock = threading.Lock()
        self._laya: Any | None = None

        if not load:
            logger.debug("skipping checkpoint load for %s", self.model_dir)
            return

        started = time.perf_counter()
        logger.info("loading Laya checkpoint from %s on %s", self.model_dir, device)
        try:
            import laya
        except ImportError as exc:
            raise AgentLoadError(f"the laya package is not importable: {exc}") from exc
        try:
            self._laya = laya.Agent(self.model_dir, device=device)
        except Exception as exc:
            raise AgentLoadError(f"could not load {self.model_dir}: {exc}") from exc
        logger.info("Laya checkpoint loaded in %.1fs", time.perf_counter() - started)

    def predict_choice(
        self, state: str, question_id: str, instructions: str, options: dict[str, str]
    ) -> ChoiceResult:
        """Ask one multiple-choice question.

        Args:
            state: The state text describing the position and the current stage.
            question_id: The key the answer comes back under.
            instructions: The instruction text for this stage.
            options: Mapping of option key to label. Every key is a legal answer,
                because callers build it from `board.legal_moves`.

        Returns:
            A `ChoiceResult` whose `choice` is one of `options`.

        Raises:
            ValueError: If `options` is empty.
            RuntimeError: Any error from `laya.predict` other than the token-budget
                `ValueError`, which is absorbed into a fallback.
        """
        if not options:
            raise ValueError(f"cannot ask {question_id} with no options")

        questions = {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(options),
            }
        }

        started = time.perf_counter()
        try:
            with self._lock:
                raw = self._laya.predict(state, questions)
        except ValueError as exc:
            # Laya raises when a question's option markers do not fit the
            # checkpoint's max_len / head_max_len budget. Failing the whole move is
            # worse than playing the first option, so degrade instead.
            logger.warning(
                "laya rejected question %s (%s); falling back to the first option",
                question_id,
                exc,
            )
            first = next(iter(options))
            return ChoiceResult(
                choice=first,
                probabilities={key: 1.0 if key == first else 0.0 for key in options},
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        return self._coerce_answer(raw, question_id, options, (time.perf_counter() - started) * 1000.0)

    @staticmethod
    def _coerce_answer(
        raw: dict[str, Any], question_id: str, options: dict[str, str], latency_ms: float
    ) -> ChoiceResult:
        """Normalise laya's answer into a `ChoiceResult` over exactly `options`.

        Three defensive steps happen here. laya rounds probabilities to four decimals,
        so they are renormalised over the offered keys. An all-zero distribution, which
        should not happen, becomes uniform. And a choice the model names that was never
        offered is replaced by the best offered key: the decider filters candidates on
        the choice, so a key that does not exist would empty the candidate set and leave
        the bot unable to move.

        Args:
            raw: The dict returned by `laya.predict`.
            question_id: The key the answer came back under.
            options: The options that were offered.
            latency_ms: Measured duration of the call.

        Returns:
            A `ChoiceResult` whose `choice` is one of `options`.
        """
        answer = raw.get("answers", {}).get(question_id, {})
        raw_probs = dict(answer.get("probabilities") or {})
        choice = str(answer.get("choice", ""))

        probs = {key: max(0.0, float(raw_probs.get(key, 0.0))) for key in options}
        total = sum(probs.values())
        if total <= 0:
            probs = dict.fromkeys(options, 1.0 / len(options))
        else:
            probs = {key: value / total for key, value in probs.items()}

        if choice not in options:
            fallback = max(probs, key=lambda key: (probs[key], key))
            logger.warning(
                "model returned choice %r for %s, which was not offered; using %r",
                choice,
                question_id,
                fallback,
            )
            choice = fallback

        return ChoiceResult(choice=choice, probabilities=probs, latency_ms=latency_ms)
```

- [ ] **Step 4: Write `backend/tests/support.py`**

```python
"""Test doubles shared across the suite. Nothing here imports `laya` or `torch`."""

from __future__ import annotations

from dataclasses import dataclass, field

from chessbot.agent import ChoiceResult


@dataclass
class FakeAgent:
    """Scripted stand-in for `LocalLayaAgent`.

    Records every question asked, so tests can assert both the number of model calls
    and the exact prompt text.

    Attributes:
        script: Maps a question id to the option key that should win. A value may also be
            a list of keys, consumed one per call, which is how a test scripts a whole
            line such as fool's mate. A question id that is absent, exhausted, or
            present but naming an option that was not offered, falls back to the first
            offered option.
        probabilities: Maps a question id to an explicit probability dict, overriding
            `script` for that question. Use it to put all mass on one option.
        calls: Every `(question_id, state, instructions, options)` asked so far.
    """

    script: dict[str, str | list[str]] = field(default_factory=dict)
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    calls: list[tuple[str, str, str, dict[str, str]]] = field(default_factory=list)

    def predict_choice(
        self, state: str, question_id: str, instructions: str, options: dict[str, str]
    ) -> ChoiceResult:
        """Return the scripted answer and record the question.

        Args:
            state: The state text.
            question_id: The question key.
            instructions: The instruction text.
            options: The options offered.

        Returns:
            A `ChoiceResult` whose `choice` is always one of `options`, matching the
            guarantee `LocalLayaAgent` provides.
        """
        self.calls.append((question_id, state, instructions, dict(options)))

        if question_id in self.probabilities:
            raw = self.probabilities[question_id]
            chosen = max(options, key=lambda key: raw.get(key, 0.0))
            probs = {key: max(0.0, float(raw.get(key, 0.0))) for key in options}
        else:
            chosen = self._next_choice(question_id, options)
            probs = _peaked(options, chosen)

        total = sum(probs.values()) or 1.0
        return ChoiceResult(
            choice=chosen,
            probabilities={key: value / total for key, value in probs.items()},
            latency_ms=1.0,
        )

    def _next_choice(self, question_id: str, options: dict[str, str]) -> str:
        """Take the next scripted option for a question, falling back to the first.

        Args:
            question_id: The question key.
            options: The options offered.

        Returns:
            One key of `options`.
        """
        scripted = self.script.get(question_id)
        if isinstance(scripted, list):
            while scripted:
                candidate = scripted.pop(0)
                if candidate in options:
                    return candidate
        elif scripted in options:
            return str(scripted)
        return next(iter(options))

    @property
    def question_ids(self) -> list[str]:
        """The question ids asked so far, in order."""
        return [call[0] for call in self.calls]

    @property
    def call_count(self) -> int:
        """How many questions have been asked."""
        return len(self.calls)


def _peaked(options: dict[str, str], winner: str) -> dict[str, float]:
    """Return probabilities giving `winner` most of the mass.

    Args:
        options: The options offered.
        winner: The key that should be most likely.

    Returns:
        A probability per offered key, summing to 1.
    """
    losers = len(options) - 1
    return {key: (0.7 if key == winner else 0.3 / losers) for key in options} if losers else {winner: 1.0}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run from `backend/`:
```bash
uv run pytest tests/test_agent.py
```

Expected: all pass. `test_predict_is_serialised_so_two_moves_never_overlap` is the one to watch: it fails if the lock is removed.

- [ ] **Step 6: Lint**

Run from `backend/`:
```bash
uv run ruff check chessbot tests
uv run ruff format chessbot tests
```

---

### Task 5: `chessbot/decider.py`

**Files:**
- Create: `backend/chessbot/decider.py`
- Create: `backend/tests/test_decider.py`

**Interfaces:**
- Consumes: `chessbot.prompts` (`INSTRUCTIONS`, `QUESTION_PIECE`, `QUESTION_DIRECTION`, `QUESTION_DESTINATION`, `SLIDING_PIECES`, `piece_state`, `direction_state`, `destination_state`, `piece_options`, `direction_options`, `destination_options`), `chessbot.engine` (`destination_label`, `direction_of`, `piece_symbol_of`), `chessbot.agent.Agent`, `tests.support.FakeAgent`.
- Produces: `StageCall`, `Decision`, `decide_move(agent, board) -> Decision`.

Task 7 depends on `Decision` for the API response shape.

- [ ] **Step 1: Write the failing test `backend/tests/test_decider.py`**

```python
"""The three-stage decision chain.

Every test drives a `FakeAgent`, so none load the checkpoint. The tests assert three
things: the chosen move is always legal, the chain asks the right number of questions,
and the trace faithfully records what happened.
"""

from __future__ import annotations

import chess
import pytest

from chessbot.decider import decide_move
from chessbot.engine import board_from_moves, destination_label, piece_symbol_of
from tests.support import FakeAgent


def test_the_chosen_move_is_always_legal():
    decision = decide_move(FakeAgent(), chess.Board())
    assert decision.move in chess.Board().legal_moves


def test_pawn_path_asks_two_questions():
    """Pawns, knights and kings skip the direction stage, so two calls, not three."""
    board = board_from_moves(["e2e4", "d7d5"])
    agent = FakeAgent(script={"piece_selection": "P", "destination_selection": "d5"})
    decision = decide_move(agent, board)
    assert agent.question_ids == ["piece_selection", "destination_selection"]
    assert decision.move.uci() == "e4d5"
    assert [stage.stage for stage in decision.stages] == ["piece", "destination"]


def test_rook_path_asks_three_questions_in_order():
    board = board_from_moves(["e2e4", "d7d5", "g1f3", "d5e4", "f1e2"])
    agent = FakeAgent(script={"piece_selection": "R"})
    decision = decide_move(agent, board)
    assert agent.question_ids == [
        "piece_selection",
        "direction_selection",
        "destination_selection",
    ]
    assert [stage.stage for stage in decision.stages] == ["piece", "direction", "destination"]
    assert "Selected direction:" in decision.stages[2].state


def test_the_first_question_is_byte_exact():
    board = board_from_moves(["e2e4", "d7d5"])
    agent = FakeAgent()
    decide_move(agent, board)
    _, state, instructions, options = agent.calls[0]
    assert state == f"Chess position. FEN: {board.fen()}. Side to move: black. Stage: piece."
    assert instructions == "Choose the best chess piece for the move."
    assert options == {"P": "Pawn", "N": "Knight", "B": "Bishop", "R": "Rook"}


def test_every_offered_option_is_a_legal_move_of_the_chosen_piece():
    board = board_from_moves(["e2e4"])
    agent = FakeAgent(script={"piece_selection": "R"})
    decide_move(agent, board)
    _, _, _, options = agent.calls[-1]
    reachable = {
        chess.square_name(move.to_square)
        for move in board.legal_moves
        if piece_symbol_of(board, move) == "R"
    }
    assert reachable
    assert set(options) <= reachable


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
    for stage in decision.stages:
        assert set(stage.probabilities) == set(stage.options)
        assert sum(stage.probabilities.values()) == pytest.approx(1.0)


def test_each_stage_records_the_instructions_it_used():
    decision = decide_move(FakeAgent(), chess.Board())
    by_id = {stage.question_id: stage.instructions for stage in decision.stages}
    assert by_id["piece_selection"] == "Choose the best chess piece for the move."
    assert by_id["destination_selection"] == "Choose the best chess piece destination."


# --- forced-move short-circuit --------------------------------------------


def test_a_position_with_one_legal_move_asks_nothing():
    """With one legal move there is nothing to decide, so no model calls are made."""
    board, only_uci = find_single_legal_move_position()
    agent = FakeAgent()
    decision = decide_move(agent, board)

    assert decision.move.uci() == only_uci
    assert decision.forced is True
    assert agent.call_count == 0
    assert decision.stages, "the trace must still explain why nothing was asked"
    assert all(stage.skipped_reason for stage in decision.stages)


def test_more_than_one_legal_move_is_not_forced():
    decision = decide_move(FakeAgent(), chess.Board())
    assert decision.forced is False
    assert decision.stages[0].skipped_reason is None


def test_a_lone_candidate_piece_is_skipped():
    """One piece type, one move: the piece stage is not asked either."""
    board = board_from_moves(["e2e4", "d7d5", "g1f3", "d5e4", "b8c6", "f3e5"])
    agent = FakeAgent()
    decision = decide_move(agent, board)
    if len(list(board.legal_moves)) == 1:
        assert decision.forced is True
        assert agent.call_count == 0
        assert all(stage.skipped_reason for stage in decision.stages)


def test_a_position_with_no_legal_moves_is_rejected():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate
    assert not list(board.legal_moves)
    with pytest.raises(ValueError, match="no legal moves"):
        decide_move(FakeAgent(), board)


# --- tie-break ------------------------------------------------------------


def test_tie_break_picks_lowest_uci_when_two_rooks_share_a_destination():
    """Nothing in the question chain separates two rooks that reach the same square."""
    board = chess.Board("4k3/8/8/8/8/8/8/R3R2K w - - 0 1")
    agent = FakeAgent(script={"piece_selection": "R", "direction_selection": "horizontal"})
    decision = decide_move(agent, board)
    assert decision.tie_break is not None
    assert "lowest UCI" in decision.tie_break
    assert decision.move.uci() == "a1e1"


def test_tie_break_is_absent_when_the_choice_is_unambiguous():
    assert decide_move(FakeAgent(), chess.Board()).tie_break is None


# --- edge positions -------------------------------------------------------


def test_promotion_is_offered_with_the_promotion_piece():
    board = chess.Board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1")
    agent = FakeAgent(script={"piece_selection": "P", "destination_selection": "e8=Q"})
    decision = decide_move(agent, board)
    assert decision.move.promotion == chess.QUEEN
    assert decision.move.uci() == "e7e8q"
    assert "e8=Q" in decision.stages[-1].options


def test_castling_is_reachable():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    decision = decide_move(
        FakeAgent(script={"piece_selection": "R", "direction_selection": "horizontal"}), board
    )
    assert decision.move in board.legal_moves


def test_en_passant_is_reachable():
    board = board_from_moves(["e2e4", "a7a6", "e4e5", "d7d5"])
    agent = FakeAgent(script={"piece_selection": "P", "destination_selection": "d6"})
    decision = decide_move(agent, board)
    assert decision.move.uci() == "e5d6"


def test_a_slider_with_one_direction_still_asks_a_destination():
    board = board_from_moves(["e2e4"])
    agent = FakeAgent(script={"piece_selection": "R", "direction_selection": "vertical"})
    decision = decide_move(agent, board)
    assert [stage.stage for stage in decision.stages] == ["piece", "direction", "destination"]


def test_one_piece_type_is_offered_in_a_directed_position():
    """The direction stage is only skipped when a single *move* remains."""
    board = board_from_moves(["e2e4", "d7d5", "g1f3", "d5e4", "f1e2"])
    agent = FakeAgent(script={"piece_selection": "R"})
    decision = decide_move(agent, board)
    assert "direction_selection" in agent.question_ids


def find_single_legal_move_position(max_depth: int = 4) -> tuple[chess.Board, str]:
    """Find a position, at `board.turn`'s move, with exactly one legal move.

    Searching rather than hard-coding a FEN means the test cannot rot if the FEN gets
    mis-transcribed, and the assertion is about the invariant rather than about one
    memorised position.

    Args:
        max_depth: How many plies of the opening to explore before searching.

    Returns:
        A `(board, uci)` pair, where `uci` is the only legal move.

    Raises:
        AssertionError: If no such position exists within `max_depth` plies. Raise the
            depth rather than pasting a FEN that may be wrong.
    """

    def walk(board: chess.Board, depth: int):
        if depth == 0:
            yield board
            return
        for move in board.legal_moves:
            board.push(move)
            yield from walk(board, depth - 1)
            board.pop()

    for board in walk(chess.Board(), max_depth):
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return board, moves[0].uci()

    raise AssertionError(
        f"no position with exactly one legal move within {max_depth} plies; raise max_depth"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `backend/`:
```bash
uv run pytest tests/test_decider.py
```

Expected: collection error, `ModuleNotFoundError: No module named 'chessbot.decider'`.

- [ ] **Step 3: Write `backend/chessbot/decider.py`**

```python
"""Turns a position into one legal move, by asking the model a chain of questions.

The model cannot name a move. It can only answer a multiple-choice question about the
position. A move is therefore recovered by narrowing: which piece moves, then for
bishops, rooks and queens which direction, then which destination square.

Every option offered at every stage is derived from `board.legal_moves`. That is what
makes an illegal move structurally impossible rather than merely unlikely.

Selection is greedy -- the argmax at each stage -- so the bot is deterministic and its
decisions are reproducible. That is how the checkpoint was evaluated, and per-stage
argmax is not the same as maximising the product of the stage probabilities.
"""

from __future__ import annotations

import logging
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
        options: Mapping of option key to label, as offered.
        choice: The option the model selected.
        probabilities: Probability per offered option.
        latency_ms: How long the `predict` call took.
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
        move: The legal move to play.
        stages: One `StageCall` per stage, in the order they were considered, including
            stages that were skipped.
        total_latency_ms: Sum of the stage latencies.
        forced: True when a single legal move made every question unnecessary.
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
    questions are skipped, because the answer is already determined. At 1-8 seconds per
    call on CPU, skipping two calls in a forced position is worth the branch.

    Args:
        agent: The model wrapper to ask.
        board: The position to move in. `board.turn`'s side must be the one moving.

    Returns:
        A `Decision` whose `move` is a member of `board.legal_moves`.

    Raises:
        ValueError: If the position has no legal moves, or if the chain eliminated every
            candidate, which should be impossible because every option offered came from
            `board.legal_moves`.
    """
    candidates = list(board.legal_moves)
    if not candidates:
        raise ValueError(f"no legal moves in {board.fen()}")

    stages: list[StageCall] = []
    forced = len(candidates) == 1

    piece = decide_piece(agent, board, candidates, stages)
    candidates = [move for move in candidates if piece_symbol_of(board, move) == piece]
    if len(candidates) == 1:
        forced = True

    direction: str | None = None
    if piece in SLIDING_PIECES:
        if len(candidates) == 1:
            stages.append(
                skipped_stage("direction", QUESTION_DIRECTION, direction_state(board, piece))
            )
        else:
            call = ask(agent, "direction", QUESTION_DIRECTION, direction_state(board, piece),
                       direction_options(candidates))
            stages.append(call)
            if call.choice is not None:
                direction = call.choice
                candidates = [move for move in candidates if direction_of(move) == direction]
        if len(candidates) == 1:
            forced = True

    if len(candidates) == 1:
        stages.append(
            skipped_stage("destination", QUESTION_DESTINATION,
                          destination_state(board, piece, direction))
        )
        forced = True
    else:
        call = ask(agent, "destination", QUESTION_DESTINATION,
                   destination_state(board, piece, direction), destination_options(candidates))
        stages.append(call)
        if call.choice is not None:
            candidates = [move for move in candidates if destination_label(move) == call.choice]

    if not candidates:
        logger.error(
            "the decision chain eliminated every candidate in %s; this should be impossible "
            "because every option came from legal_moves",
            board.fen(),
        )
        raise ValueError(f"decision chain produced no legal move in {board.fen()}")

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
        The chosen piece symbol: `"P"`, `"N"`, `"B"`, `"R"`, `"Q"` or `"K"`.
    """
    if len(candidates) == 1:
        stages.append(skipped_stage("piece", QUESTION_PIECE, piece_state(board)))
        return piece_symbol_of(board, candidates[0])

    options = piece_options(board, candidates)
    call = ask(agent, "piece", QUESTION_PIECE, piece_state(board), options)
    stages.append(call)

    if call.choice is None or call.choice not in options:
        fallback = next(iter(options))
        logger.warning(
            "piece stage produced unusable choice %r; falling back to %r", call.choice, fallback
        )
        return fallback
    return call.choice


def ask(
    agent: Agent, stage: str, question_id: str, state: str, options: dict[str, str]
) -> StageCall:
    """Ask one stage's question and record the outcome.

    Args:
        agent: The model wrapper.
        stage: The stage name, for the trace.
        question_id: The question key.
        state: The state text.
        options: The options offered, always derived from legal moves.

    Returns:
        A `StageCall` with the answer filled in.
    """
    instructions = INSTRUCTIONS[question_id]
    result = agent.predict_choice(state, question_id, instructions, options)

    logger.debug(
        "%s stage\n  state: %s\n  options: %s\n  choice: %s\n  probabilities: %s\n  took %.0fms",
        stage,
        state,
        options,
        result.choice,
        result.probabilities,
        result.latency_ms,
    )

    return StageCall(
        stage=stage,
        question_id=question_id,
        state=state,
        instructions=instructions,
        options=options,
        choice=result.choice,
        probabilities=result.probabilities,
        latency_ms=result.latency_ms,
    )


def skipped_stage(stage: str, question_id: str, state: str) -> StageCall:
    """Build a `StageCall` for a stage that was not asked.

    The state text is still recorded, so the trace shows what the model would have been
    asked even when the answer was already determined.

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run from `backend/`:
```bash
uv run pytest tests/test_decider.py
```

Expected: all pass. If `find_single_legal_move_position` raises `AssertionError`, raise `max_depth` to 5 and re-run.

- [ ] **Step 5: Run the whole backend suite**

Run from `backend/`:
```bash
uv run pytest
```

Expected: everything passes. No test loads the checkpoint.

- [ ] **Step 6: Lint**

Run from `backend/`:
```bash
uv run ruff check chessbot tests
uv run ruff format chessbot tests
```

---

### Task 6: Persistence and schemas

**Files:**
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/db.py`
- Create: `backend/app/models.py`
- Create: `backend/app/engine_support.py`
- Create: `backend/app/schemas.py`
- Create: `backend/tests/test_schemas.py`

**Interfaces:**
- Consumes: `chessbot.engine` (`board_from_moves`, `game_status`, `legal_uci_moves`).
- Produces:
  - `app.config.settings` — module-level `Settings` with `laya_model_dir: Path`, `laya_device: str`, `log_level: str`, `database_url: str`.
  - `app.db.Base`, `app.db.engine`, `app.db.SessionLocal`, `app.db.make_engine(url)`, `app.db.init_db()`, `app.db.get_db()`.
  - `app.models.Game` with `id`, `human_side`, `status`, `result`, `termination`, `moves`, `last_trace`, `created_at`, `updated_at`.
  - `app.schemas.MoveRequest`, `StageTrace`, `BotTrace`, `GameSummary`, `GameState`, `HealthResponse`, `build_game_state(game, bot_trace=None)`.

Tasks 7 and 8 depend on all of the above.

- [ ] **Step 1: Write the failing test `backend/tests/test_schemas.py`**

```python
"""Response shaping. The move list is authoritative; no FEN is stored anywhere."""

from __future__ import annotations

import chess
import pytest
from pydantic import ValidationError

from app import schemas
from app.models import Game


def test_a_fresh_game_is_the_starting_position():
    game = Game(id=1, human_side="white", status="active", moves=[])
    state = schemas.build_game_state(game)
    assert state.fen == chess.STARTING_FEN
    assert state.turn == "white"
    assert state.moves == []
    assert state.last_move is None
    assert state.bot_trace is None
    assert state.result is None
    assert state.termination is None
    assert state.is_check is False
    assert state.status == "active"
    assert len(state.legal_moves) == 20


def test_the_fen_is_derived_from_the_move_list():
    game = Game(id=1, human_side="white", status="active", moves=["e2e4", "c7c5", "g1f3"])
    reference = chess.Board()
    for uci in ["e2e4", "c7c5", "g1f3"]:
        reference.push_uci(uci)

    state = schemas.build_game_state(game)
    assert state.fen == reference.fen()
    assert state.turn == "white"
    assert state.last_move == "g1f3"


def test_status_is_recomputed_from_the_position_not_trusted_from_the_row():
    """A stale `status` column must not survive into the response."""
    game = Game(id=1, human_side="white", status="active", moves=["f3", "e5", "g4", "Qh4"])
    state = schemas.build_game_state(game)

    assert state.status == "black_won"
    assert state.termination == "checkmate"
    assert state.is_check is True


def test_check_is_reported():
    game = Game(id=1, human_side="black", status="active", moves=["e4", "e5", "Qh5"])
    assert schemas.build_game_state(game).is_check is True


def test_a_stored_trace_is_returned():
    trace = {
        "move": "e7e5",
        "stages": [
            {
                "stage": "piece",
                "question_id": "piece_selection",
                "state": "Chess position. FEN: ... Stage: piece.",
                "instructions": "Choose the best chess piece for the move.",
                "options": {"P": "Pawn"},
                "choice": "P",
                "probabilities": {"P": 1.0},
                "latency_ms": 41.0,
                "skipped_reason": None,
            }
        ],
        "total_latency_ms": 41.0,
        "forced": False,
        "tie_break": None,
    }
    game = Game(id=1, human_side="white", status="active", moves=["e2e4"], last_trace=trace)
    state = schemas.build_game_state(game)

    assert state.bot_trace is not None
    assert state.bot_trace.move == "e7e5"
    assert state.bot_trace.stages[0].choice == "P"
    assert state.bot_trace.stages[0].probabilities == {"P": 1.0}


def test_an_explicit_trace_overrides_the_stored_one():
    game = Game(id=1, human_side="white", status="active", moves=["e2e4"])
    fresh = schemas.BotTrace(move="c7c5", stages=[], total_latency_ms=12.0)
    assert schemas.build_game_state(game, bot_trace=fresh).bot_trace.move == "c7c5"


def test_the_state_round_trips_through_json():
    game = Game(id=1, human_side="white", status="active", moves=["e2e4", "e7e5"])
    state = schemas.build_game_state(game)
    assert schemas.GameState.model_validate_json(state.model_dump_json()).fen == state.fen


def test_move_request_rejects_an_over_long_uci():
    with pytest.raises(ValidationError):
        schemas.MoveRequest(uci="a" * 40)
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `backend/`:
```bash
uv run pytest tests/test_schemas.py
```

Expected: `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Create `backend/app/__init__.py`**

Empty file.

- [ ] **Step 4: Write `backend/app/config.py`**

```python
"""Environment-backed settings.

Every value has a working default, so `uv run uvicorn app.main:app` starts with no
environment set up at all. Override any of them with an environment variable of the same
name, or a `.env` file in `backend/`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the backend.

    Attributes:
        laya_model_dir: Path to the fine-tuned Laya checkpoint.
        laya_device: Torch device. `"cpu"` here, since this machine has no GPU.
        log_level: Root log level. `DEBUG` narrates the whole decision chain, which is the
            fastest way to see why the bot played what it played.
        database_url: SQLAlchemy URL for the game store.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    laya_model_dir: Path = REPO_ROOT / "model" / "laya_finetuned_typed_decisions"
    laya_device: str = "cpu"
    log_level: str = "INFO"
    database_url: str = f"sqlite:///{REPO_ROOT / 'backend' / 'app' / 'chessbot.db'}"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, created once."""
    return Settings()


settings = get_settings()
```

- [ ] **Step 5: Write `backend/app/db.py`**

```python
"""SQLAlchemy engine, session factory and declarative base."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def make_engine(url: str):
    """Build a SQLAlchemy engine.

    Args:
        url: A SQLAlchemy database URL.

    Returns:
        A configured `Engine`. SQLite gets `check_same_thread=False` so a request handler
        and the threadpool can share connections.
    """
    return create_engine(
        url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}
    )


engine = make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create any tables that do not exist yet.

    There is one table and no schema history worth preserving, so Alembic would be
    ceremony. Revisit when the schema starts moving.
    """
    from app import models  # noqa: F401  importing registers the mappers

    Base.metadata.create_all(bind=engine)
    logger.info("database ready at %s", settings.database_url)


def get_db() -> Iterator[Session]:
    """Yield a request-scoped session and close it afterwards.

    Yields:
        An open `Session`.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

- [ ] **Step 6: Write `backend/app/models.py`**

```python
"""The one ORM model this app has."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Game(Base):
    """One game between a human and the bot.

    `moves` is the single source of truth for the position. There is deliberately no
    `fen` column: a stored FEN can drift from the move history, and a player seeing one
    position while the bot reasons about another is a genuinely nasty class of bug. The
    current FEN is derived on read.

    Attributes:
        id: Primary key.
        human_side: Which colour the human plays, `"white"` or `"black"`. Chosen at random
            when the game is created.
        status: `"active"`, `"white_won"`, `"black_won"` or `"draw"`.
        result: PGN result (`"1-0"`, `"0-1"`, `"1/2-1/2"`), or `None` while live.
        termination: Why the game ended, e.g. `"checkmate"`.
        moves: UCI move strings in order.
        last_trace: The most recent bot decision as a plain dict, for the UI.
        created_at: When the game was created.
        updated_at: When the game last changed.
    """

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    human_side: Mapped[str] = mapped_column(String(5), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    result: Mapped[str | None] = mapped_column(String(8), nullable=True)
    termination: Mapped[str | None] = mapped_column(String(32), nullable=True)
    moves: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    last_trace: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return (
            f"<Game id={self.id} human_side={self.human_side!r} "
            f"status={self.status!r} plies={len(self.moves or [])}>"
        )
```

- [ ] **Step 7: Write `backend/app/engine_support.py`**

```python
"""Narrow re-export of the `chessbot.engine` helpers the web layer needs.

Keeping this explicit means `app.*` never reaches into the library's internals, so the
library can grow or move without a scatter of import updates.
"""

from __future__ import annotations

from chessbot.engine import board_from_moves, game_status, legal_uci_moves

__all__ = ["board_from_moves", "game_status", "legal_uci_moves"]
```

- [ ] **Step 8: Write `backend/app/schemas.py`**

```python
"""Pydantic request and response models.

`GameState` is the single response shape for every game endpoint. It is built from the
move list on every read, so it always reflects the position the bot reasoned about.
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
        status: Current status.
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

    Args:
        game: The stored game.
        bot_trace: A freshly produced trace to attach, overriding the stored one. Used by
            the move endpoint, where the trace was just created.

    Returns:
        The `GameState` for the position the move list describes. `status` is recomputed
        from that position rather than read off the row, so a stale column cannot reach
        the client.
    """
    board = board_from_moves(game.moves or [])
    status, _, _ = game_status(board)

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
        result=game.result,
        termination=game.termination,
        is_check=board.is_check(),
        legal_moves=legal_uci_moves(board),
        last_move=game.moves[-1] if game.moves else None,
        bot_trace=trace,
    )
```

- [ ] **Step 9: Run the tests to verify they pass**

Run from `backend/`:
```bash
uv run pytest tests/test_schemas.py
```

Expected: all pass.

- [ ] **Step 10: Lint**

Run from `backend/`:
```bash
uv run ruff check app tests
uv run ruff format app tests
```

---

### Task 7: HTTP layer

**Files:**
- Create: `backend/app/state.py`
- Create: `backend/app/deps.py`
- Create: `backend/app/routers/__init__.py`
- Create: `backend/app/routers/games.py`
- Create: `backend/app/routers/health.py`
- Create: `backend/app/main.py`

**Interfaces:**
- Consumes: `chessbot.decider.decide_move`, `chessbot.agent.LocalLayaAgent`, `chessbot.engine.board_from_moves`, `app.config.settings`, `app.db.get_db`, `app.models.Game`, `app.schemas.*`.
- Produces:
  - `app.state.set_agent(agent)`, `app.state.get_agent() -> Agent | None`, `app.state.clear_agent()`
  - `app.deps.get_agent() -> Agent`
  - `app.routers.games.router` with `POST /api/games`, `GET /api/games`, `GET /api/games/{game_id}`, `POST /api/games/{game_id}/move`, `POST /api/games/{game_id}/undo`, `GET /api/games/{game_id}/pgn`
  - `app.routers.health.router` with `GET /api/health`
  - `app.main.create_app() -> FastAPI`, `app.main.app`

Task 8 overrides `app.deps.get_agent` with `FakeAgent`, which is why it must be a dependency rather than a module-level lookup.

- [ ] **Step 1: Write `backend/app/state.py`**

```python
"""Process-wide holder for the loaded model.

The agent is created once at startup and read from anywhere. It is a module-level holder
rather than something attached to `app.state`, so the routers do not need the app object
and tests can swap it without building an app.
"""

from __future__ import annotations

import logging

from chessbot.agent import Agent

logger = logging.getLogger(__name__)

_agent: Agent | None = None


def set_agent(agent: Agent | None) -> None:
    """Install the process-wide agent, or clear it.

    Args:
        agent: The loaded agent, or `None` to mark the service degraded.
    """
    global _agent
    _agent = agent
    logger.info("agent %s", "loaded" if agent is not None else "cleared")


def get_agent() -> Agent | None:
    """Return the process-wide agent.

    Returns:
        The `Agent`, or `None` when no model is loaded.
    """
    return _agent


def clear_agent() -> None:
    """Clear the process-wide agent."""
    set_agent(None)
```

- [ ] **Step 2: Write `backend/app/deps.py`**

```python
"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, status

from chessbot.agent import Agent
from app import state


def get_agent() -> Agent:
    """Return the loaded agent, or fail with `503`.

    Declared as a dependency rather than called directly inside the handlers, so a test
    can override it with `app.dependency_overrides` and never load the checkpoint.

    Returns:
        The process-wide `Agent`.

    Raises:
        HTTPException: `503` when no model is loaded. The message names the reason, so a
            player sees why the bot is not moving.
    """
    agent = state.get_agent()
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the model failed to load, so no move can be decided",
        )
    return agent
```

- [ ] **Step 3: Write `backend/app/routers/__init__.py`**

Empty file.

- [ ] **Step 4: Write `backend/app/routers/health.py`**

```python
"""The health endpoint. Answers whether the bot can move at all."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app import state
from app.config import settings
from app.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report model availability.

    Returns:
        `"ok"` when the checkpoint is in memory, `"degraded"` when it is not. Degraded is
        a normal, expected response on a machine where the model could not be loaded; it
        is not an error.
    """
    loaded = state.get_agent() is not None
    logger.debug("health check: model_loaded=%s", loaded)
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        device=settings.laya_device,
        model_dir=str(settings.laya_model_dir),
    )
```

- [ ] **Step 5: Write `backend/app/routers/games.py`**

```python
"""Game endpoints.

Every endpoint rebuilds the board by replaying `game.moves`, so the position the bot
reasons about is always the position the player sees.
"""

from __future__ import annotations

import logging
import random

import chess
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from chessbot.agent import Agent
from chessbot.decider import decide_move
from chessbot.engine import board_from_moves, game_status, uci_to_move
from app.deps import get_agent
from app.db import get_db
from app.models import Game
from app.schemas import BotTrace, GameState, GameSummary, MoveRequest, build_game_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/games", tags=["games"])


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

    Args:
        game: The stored game.

    Returns:
        The replayed `chess.Board`.

    Raises:
        HTTPException: `409` when the game is already finished.
    """
    board = board_from_moves(game.moves or [])
    game_status_value, _, _ = game_status(board)
    if game_status_value != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"game {game.id} is over ({game_status_value})",
        )
    return board


def bot_trace_of(decision) -> BotTrace:
    """Convert a `Decision` into the API trace shape.

    Args:
        decision: What `decide_move` returned.

    Returns:
        The `BotTrace` to store on the game and return to the client.
    """
    return BotTrace(
        move=decision.move.uci(),
        stages=[stage for stage in decision.stages],
        total_latency_ms=decision.total_latency_ms,
        forced=decision.forced,
        tie_break=decision.tie_break,
    )


def opponent_of(human_side: str) -> chess.Color:
    """Return the bot's colour.

    Args:
        human_side: `"white"` or `"black"`.

    Returns:
        The opposite colour.
    """
    return chess.BLACK if human_side == "white" else chess.WHITE


async def play_bot_move(agent: Agent, board: chess.Board) -> tuple[chess.Move, BotTrace]:
    """Ask the bot for a move without blocking the event loop.

    Args:
        agent: The loaded model wrapper.
        board: The position the bot is to move in.

    Returns:
        The chosen move and the trace explaining it.
    """
    decision = await run_in_threadpool(decide_move, agent, board)
    return decision.move, bot_trace_of(decision)


@router.post("", response_model=GameState, status_code=status.HTTP_201_CREATED)
async def create_game(
    db: Session = Depends(get_db), agent: Agent = Depends(get_agent)
) -> GameState:
    """Create a game, randomising which side the human plays.

    The human's side is chosen here rather than by the client so a player cannot
    accidentally always play the same colour, and so the first move -- whether the human
    or the bot -- is decided by one place.

    Returns:
        The new game's state, including the bot's opening move when the human is black.
    """
    human_side = random.choice(["white", "black"])  # noqa: S311 - a coin flip, not a secret
    game = Game(human_side=human_side, status="active", moves=[])
    db.add(game)
    db.commit()
    db.refresh(game)

    logger.info("game %s created, human plays %s", game.id, human_side)

    if human_side == "black":
        board = chess.Board()
        move, trace = await play_bot_move(agent, board)
        board.push(move)
        game.moves = [move.uci()]
        game.last_trace = trace.model_dump()
        game.updated_at = datetime_now()
        db.commit()
        db.refresh(game)
        logger.info("game %s: bot opened with %s", game.id, move.uci())

    return build_game_state(game)


@router.get("", response_model=list[GameSummary])
def list_games(db: Session = Depends(get_db)) -> list[GameSummary]:
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
def get_game(game_id: int, db: Session = Depends(get_db)) -> GameState:
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
    db: Session = Depends(get_db),
    agent: Agent = Depends(get_agent),
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

    bot_colour = opponent_of(game.human_side)
    if board.turn != bot_colour:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"it is not the human's turn in game {game.id}",
        )

    try:
        move = uci_to_move(board, payload.uci)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error

    board.push(move)
    game.moves = list(game.moves or []) + [move.uci()]
    logger.info("game %s: human played %s", game.id, move.uci())

    status_value, result, termination = game_status(board)
    if status_value != "active":
        finish(game, status_value, result, termination)
        db.commit()
        db.refresh(game)
        logger.info("game %s finished: %s (%s)", game.id, status_value, termination)
        return build_game_state(game)

    try:
        bot_move, trace = await play_bot_move(agent, board)
    except Exception as error:
        logger.exception("game %s: the bot failed to decide a move", game.id)
        # The human move is dropped rather than persisted, so retrying the same move
        # starts from the position the player was already looking at.
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the bot could not decide a move; nothing was saved",
        ) from error

    board.push(bot_move)
    game.moves = list(game.moves) + [bot_move.uci()]
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
def undo(game_id: int, db: Session = Depends(get_db)) -> GameState:
    """Take back the last human move and the bot's reply.

    Two plies are removed, not one: leaving the bot's reply in place would put the game
    back on the human's turn with the bot to move, which is not a position anyone wants.

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
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="there is nothing to undo"
        )

    board = board_from_moves(moves)
    if board.turn == opponent_of(game.human_side):
        moves.pop()

    game.moves = moves
    game.last_trace = None
    game.status = "active"
    game.result = None
    game.termination = None
    game.updated_at = datetime_now()
    db.commit()
    db.refresh(game)

    logger.info("game %s: undid back to ply %d", game.id, len(moves))
    return build_game_state(game)


@router.get("/{game_id}/pgn", response_class=Response)
def export_pgn(game_id: int, db: Session = Depends(get_db)) -> Response:
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
    headers = {
        "White": "Human" if game.human_side == "white" else "Laya",
        "Black": "Laya" if game.human_side == "white" else "Human",
        "Result": game.result or "*",
    }
    pgn = str(chess.pgn.Game(board, headers=headers))
    return Response(content=pgn, media_type="text/plain")


def finish(game: Game, status_value: str, result: str | None, termination: str | None) -> None:
    """Record a finished game on the row.

    Args:
        game: The game to update.
        status_value: `"active"`, `"white_won"`, `"black_won"` or `"draw"`.
        result: The PGN result, or `None` while live.
        termination: Why the game ended, or `None`.
    """
    game.status = status_value
    game.result = result if status_value != "active" else None
    game.termination = termination
    game.updated_at = datetime_now()


def datetime_now():
    """Return an aware UTC timestamp.

    Returns:
        The current time, for `updated_at`.
    """
    return datetime.now(UTC)
```

- [ ] **Step 6: Write `backend/app/main.py`**

```python
"""Application factory, lifespan and logging setup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from chessbot.agent import LocalLayaAgent
from app import state
from app.config import settings
from app.db import init_db
from app.routers import games, health

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up the two logger trees.

    `chessbot` and `app` are configured together at the same level because both are
    needed to understand a move: the library narrates the decision chain and the web
    layer narrates what happened to the game.
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    logging.getLogger("chessbot").setLevel(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    )
    logging.getLogger("app").setLevel(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    )


def load_model() -> None:
    """Load the checkpoint once.

    A failure is logged with its traceback and swallowed: the app still starts, reports
    `degraded` on `/api/health`, and returns `503` on the move endpoint. A crash on
    startup would take the whole UI down for a problem that only affects the bot.
    """
    try:
        agent = LocalLayaAgent(settings.laya_model_dir, device=settings.laya_device)
    except Exception:
        logger.exception(
            "could not load the model from %s; the app will serve in a degraded state",
            settings.laya_model_dir,
        )
        state.clear_agent()
        return
    state.set_agent(agent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the database and load the model.

    Args:
        app: The application being started.

    Yields:
        Control back to the server while it serves requests.
    """
    configure_logging()
    logger.info("starting; model dir %s on %s", settings.laya_model_dir, settings.laya_device)
    init_db()
    load_model()
    yield
    state.clear_agent()
    logger.info("stopped")


def create_app() -> FastAPI:
    """Build the FastAPI application.

    Returns:
        The configured `FastAPI` instance.
    """
    app = FastAPI(title="Laya Chess Bot", version="1.0.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(games.router)
    return app


app = create_app()
```

- [ ] **Step 7: Import-check the app without starting it**

Run from `backend/`:
```bash
uv run python -c "from app.main import create_app; print(len(create_app().routes))"
```

Expected: a route count greater than 6, printed on one line. Nothing is loaded and no database is touched, because `lifespan` has not run.

- [ ] **Step 8: Lint**

Run from `backend/`:
```bash
uv run ruff check app
uv run ruff format app
```

---

### Task 8: API tests

**Files:**
- Modify: `backend/tests/conftest.py`
- Create: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `app.main.create_app`, `app.deps.get_agent`, `app.db.get_db`, `app.db.Base`, `app.db.make_engine`, `app.db.SessionLocal`, `app.models.Game`, `tests.support.FakeAgent`.
- Produces: a `client` fixture yielding a `TestClient` wired to a throwaway SQLite file and a `FakeAgent`.

- [ ] **Step 1: Extend `backend/tests/conftest.py`**

Append these imports and fixtures to the file Task 1 created:

```python
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app import state
from app.db import Base, get_db, make_engine
from app.deps import get_agent
from app.main import create_app
from tests.support import FakeAgent


@pytest.fixture
def agent() -> Iterator[FakeAgent]:
    """Install a `FakeAgent` as the process agent for the duration of a test.

    Yielding it as its own fixture is what lets a test script a specific line, such as
    the two black moves that make fool's mate possible.

    Yields:
        The `FakeAgent` every request will be answered by.
    """
    fake = FakeAgent()
    state.set_agent(fake)
    yield fake
    state.clear_agent()


@pytest.fixture
def client(tmp_path, agent) -> Iterator[TestClient]:
    """Yield a `TestClient` with a temporary database and a fake model.

    Both dependencies the routers pull in are overridden. Overriding `get_agent` is what
    keeps the 842 MB checkpoint out of the test suite; overriding `get_db` keeps tests off
    the developer's real `chessbot.db`. The agent override delegates to `state.get_agent`
    rather than capturing the fake, so a test that clears the process agent exercises the
    degraded `503` path. The lifespan still runs, so `init_db` is exercised, but
    `load_model` finds the agent already installed.

    Args:
        tmp_path: pytest's temporary directory.
        agent: The `FakeAgent` fixture, installed in `state` before the app starts.

    Yields:
        A `TestClient` for the app.
    """
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db() -> Iterator[Session]:
        session = testing_session()
        try:
            yield session
        finally:
            session.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_agent] = state.get_agent

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


@pytest.fixture
def seed_game(tmp_path):
    """Return a callable that inserts a `Game` row straight into the test database.

    Some states are not reachable through the API. A game where it is not the human's turn
    is one of them: both plies always land in a single request, so the API can never leave
    the position on the wrong side. Inserting the row is the honest way to test the guard.

    Args:
        tmp_path: pytest's temporary directory, shared with the `client` fixture.

    Returns:
        A callable taking `Game` column values and returning the persisted row.
    """
    from app.models import Game as GameModel

    def seed(**fields) -> GameModel:
        engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
        Base.metadata.create_all(bind=engine)
        with sessionmaker(bind=engine)() as session:
            game = GameModel(**fields)
            session.add(game)
            session.commit()
            session.refresh(game)
            session.expunge(game)
        engine.dispose()
        return game

    return seed


@pytest.fixture
def new_game(client) -> dict:
    """Create a game and return its state.

    Args:
        client: The `TestClient` fixture.

    Returns:
        The `GameState` body of `POST /api/games`.
    """
    response = client.post("/api/games")
    assert response.status_code == 201
    return response.json()
```

- [ ] **Step 2: Write `backend/tests/test_api.py`**

```python
"""HTTP behaviour. Every test runs against a `FakeAgent` and a temporary database."""

from __future__ import annotations

import chess


def test_creating_a_game_randomises_the_human_side(client):
    sides = {client.post("/api/games").json()["human_side"] for _ in range(12)}
    assert sides <= {"white", "black"}
    assert len(sides) == 2, "the coin flip should not be stuck on one colour"


def test_a_new_game_starts_from_the_initial_position(client):
    state = client.post("/api/games").json()
    assert state["fen"] == chess.STARTING_FEN or len(state["moves"]) == 1
    assert state["status"] == "active"


def test_the_bot_opens_when_the_human_is_black(client):
    state = client.post("/api/games").json()
    if state["human_side"] == "black":
        assert len(state["moves"]) == 1, "the bot must have played first"
        assert state["bot_trace"] is not None
        assert state["turn"] == "black", "the bot opened, so it is the human's turn"
    else:
        assert state["moves"] == []


def test_playing_a_move_returns_both_plies_and_the_trace(client):
    state = client.post("/api/games").json()
    while state["human_side"] != "white":
        state = client.post("/api/games").json()

    response = client.post(f"/api/games/{state['id']}/move", json={"uci": "e2e4"})
    assert response.status_code == 200
    body = response.json()

    assert body["moves"] == ["e2e4", body["moves"][1]]
    assert len(body["moves"]) == 2
    assert body["last_move"] == body["moves"][1]
    assert body["bot_trace"]["move"] == body["moves"][1]
    assert body["bot_trace"]["stages"], "the trace must explain the reply"
    assert body["turn"] == "white"


def test_an_illegal_move_is_rejected_with_400(client):
    state = _white_human_game(client)
    response = client.post(f"/api/games/{state['id']}/move", json={"uci": "e2e5"})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_an_unparseable_move_is_rejected_with_400(client):
    state = _white_human_game(client)
    assert client.post(f"/api/games/{state['id']}/move", json={"uci": "zzzz"}).status_code == 400


def test_playing_out_of_turn_is_409(client, seed_game):
    # It is black to move, but the human is white -- so this is the bot's turn.
    game = seed_game(human_side="white", status="active", moves=["e2e4"])
    response = client.post(f"/api/games/{game.id}/move", json={"uci": "d2d4"})
    assert response.status_code == 409
    assert "turn" in response.json()["detail"]


def test_a_game_already_finished_is_409(client, seed_game):
    game = seed_game(human_side="white", status="white_won", moves=["f3", "e5", "g4", "Qh4"])
    response = client.post(f"/api/games/{game.id}/move", json={"uci": "e1f1"})
    assert response.status_code == 409


def test_an_unknown_game_is_404(client):
    assert client.get("/api/games/9999").status_code == 404
    assert client.post("/api/games/9999/move", json={"uci": "e2e4"}).status_code == 404
    assert client.post("/api/games/9999/undo").status_code == 404


def test_undo_takes_back_the_human_move_and_the_reply(client):
    state = _white_human_game(client)
    played = client.post(f"/api/games/{state['id']}/move", json={"uci": "e2e4"}).json()
    assert len(played["moves"]) == 2

    undone = client.post(f"/api/games/{state['id']}/undo").json()
    assert undone["moves"] == []
    assert undone["turn"] == "white"
    assert undone["bot_trace"] is None


def test_undo_on_a_fresh_game_is_409(client):
    state = _white_human_game(client)
    assert client.post(f"/api/games/{state['id']}/undo").status_code == 409


def test_a_finished_game_refuses_more_moves_with_409(client, agent):
    agent.script = {
        "piece_selection": ["P", "Q"],
        "direction_selection": ["diagonal_ne_sw"],
        "destination_selection": ["e5", "h4"],
    }
    game = _white_human_game(client)
    # Fool's mate: the human opens badly, and the bot's replies end it.
    for uci in ["f2f3", "g2g4"]:
        response = client.post(f"/api/games/{game['id']}/move", json={"uci": uci})
        assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "black_won"
    assert body["termination"] == "checkmate"
    assert body["result"] == "0-1"
    assert body["is_check"] is True
    assert client.post(f"/api/games/{game['id']}/move", json={"uci": "e1f1"}).status_code == 409


def test_pgn_export_names_both_players(client):
    state = _white_human_game(client)
    client.post(f"/api/games/{state['id']}/move", json={"uci": "e2e4"})
    response = client.get(f"/api/games/{state['id']}/pgn")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert '[White "Human"]' in response.text
    assert '[Black "Laya"]' in response.text
    assert "1. e4" in response.text


def test_listing_games_is_newest_first(client):
    first = client.post("/api/games").json()
    second = client.post("/api/games").json()
    listed = client.get("/api/games").json()
    assert [row["id"] for row in listed[:2]] == [second["id"], first["id"]]
    assert "move_count" in listed[0]


def test_health_is_ok_when_a_model_is_installed(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_reports_degraded_when_no_model_is_loaded(client):
    from app import state

    state.clear_agent()
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_a_move_without_a_model_is_503(client):
    from app import state

    game = _white_human_game(client)
    state.clear_agent()
    response = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e4"})
    assert response.status_code == 503


def test_a_bot_failure_leaves_the_history_untouched(client, monkeypatch):
    game = _white_human_game(client)
    before = game["moves"]

    import app.routers.games as games_module

    def explode(*args, **kwargs):
        raise RuntimeError("torch exploded")

    monkeypatch.setattr(games_module, "decide_move", explode)
    response = client.post(f"/api/games/{game['id']}/move", json={"uci": "e2e4"})

    assert response.status_code == 503
    after = client.get(f"/api/games/{game['id']}").json()
    assert after["moves"] == before, "the human move must be rolled back"


def _white_human_game(client) -> dict:
    """Return a game where the human plays white and nothing has been played yet.

    Args:
        client: The `TestClient` fixture.

    Returns:
        The `GameState` body, retried until the coin flip lands on white.
    """
    for _ in range(20):
        state = client.post("/api/games").json()
        if state["human_side"] == "white" and not state["moves"]:
            return state
    raise AssertionError("the coin flip never produced a fresh white game")
```

- [ ] **Step 3: Run the API tests**

Run from `backend/`:
```bash
uv run pytest tests/test_api.py
```

Expected: all pass.

- [ ] **Step 4: Run the full suite and lint**

Run from `backend/`:
```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

Expected: everything passes and no test took more than a couple of seconds.

---

### Task 9: Frontend scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/next.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/app/layout.tsx`
- Create: `frontend/app/globals.css`
- Create: `frontend/lib/types.ts`
- Create: `frontend/lib/api.ts`

**Interfaces:**
- Consumes: nothing from Python. The shapes here mirror `app.schemas`.
- Produces: `StageTrace`, `BotTrace`, `GameState`, `GameSummary`, `HealthState` types; `createGame()`, `fetchGame(id)`, `listGames()`, `playMove(id, uci)`, `undoMove(id)`, `fetchPgn(id)`, `fetchHealth()`.

Tasks 10 and 11 import from `lib/`, which is why this task is in Wave B.

- [ ] **Step 1: Scaffold with `create-next-app`, then overwrite the generated files**

Run from the repo root:
```bash
npx --yes create-next-app@latest frontend --ts --app --no-src-dir --no-eslint --no-tailwind --import-alias "@/*" --use-npm --skip-install
```

The flags keep the layout flat (`frontend/app`, not `frontend/src/app`) to match the plan's file table, and skip the install because Step 3 does it once, with the extra dependency.

- [ ] **Step 2: Write `frontend/package.json`**

Keep whatever `create-next-app` generated for `next`, `react` and `react-dom`, and add `react-chessboard`:

```json
{
  "name": "laya-chessbot-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "next": "<the version create-next-app generated>",
    "react": "<the version create-next-app generated>",
    "react-chessboard": "^4.7.3",
    "react-dom": "<the version create-next-app generated>"
  },
  "devDependencies": {
    "@types/node": "<as generated>",
    "@types/react": "<as generated>",
    "@types/react-dom": "<as generated>",
    "typescript": "<as generated>"
  }
}
```

Do not guess the caret ranges; copy the ones the generator wrote and change nothing else.

- [ ] **Step 3: Install**

Run from `frontend/`:
```bash
npm install
```

- [ ] **Step 4: Write `frontend/next.config.ts`**

```ts
import type { NextConfig } from "next";

/**
 * Rewrites `/api/*` to the FastAPI backend so the browser sees a single origin.
 *
 * Without this, every fetch from the page would need an absolute backend URL and the
 * backend would need CORS configured. One origin in dev and production is worth the
 * rewrite.
 */
const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
```

- [ ] **Step 5: Write `frontend/tsconfig.json`**

Keep the generated file, which already has the `@/*` path alias `create-next-app` sets up. Confirm `"strict": true` and that `paths` maps `@/*` to `./*`.

- [ ] **Step 6: Write `frontend/lib/types.ts`**

```ts
/**
 * TypeScript mirror of the FastAPI response shapes.
 *
 * Kept by hand rather than generated: there are six types, and a generated file would add
 * a build step to keep in sync for less clarity than this.
 */

/** One question asked of the model, and what it answered. */
export interface StageTrace {
  stage: "piece" | "direction" | "destination";
  questionId: string;
  state: string;
  instructions: string;
  options: Record<string, string>;
  choice: string | null;
  probabilities: Record<string, number>;
  latencyMs: number;
  skippedReason: string | null;
}

/** Everything the model did to produce one move. */
export interface BotTrace {
  move: string;
  stages: StageTrace[];
  totalLatencyMs: number;
  forced: boolean;
  tieBreak: string | null;
}

/** The full state of a game. Returned by every game endpoint. */
export interface GameState {
  id: number;
  fen: string;
  moves: string[];
  turn: "white" | "black";
  humanSide: "white" | "black";
  status: "active" | "white_won" | "black_won" | "draw";
  result: string | null;
  termination: string | null;
  isCheck: boolean;
  legalMoves: string[];
  lastMove: string | null;
  botTrace: BotTrace | null;
}

/** A row in the game list. */
export interface GameSummary {
  id: number;
  humanSide: string;
  status: string;
  moveCount: number;
  createdAt: string;
}

/** Whether the bot can decide a move. */
export interface HealthState {
  status: "ok" | "degraded";
  modelLoaded: boolean;
  device: string;
  modelDir: string;
}

/** An error body from FastAPI. */
export interface ApiError {
  detail: string;
}
```

Pydantic serialises snake_case, so `question_id` arrives as `question_id`. The field names above use camelCase; therefore `lib/api.ts` must convert. Rather than duplicating the mapping by hand in every component, `lib/api.ts` owns the whole snake_case → camelCase conversion with one recursive function.

- [ ] **Step 7: Write `frontend/lib/api.ts`**

```ts
import type {
  ApiError,
  BotTrace,
  GameState,
  GameSummary,
  HealthState,
  StageTrace,
} from "./types";

/**
 * Typed fetch wrappers for the backend.
 *
 * Every response goes through `toCamel`, one recursive key rename. Pydantic emits
 * snake_case and TypeScript reads better in camelCase; one function is a smaller cost
 * than keeping the two vocabularies aligned by hand in every component.
 */

function toCamel(key: string): string {
  return key.replace(/_([a-z0-9])/g, (_match, char: string) => char.toUpperCase());
}

function convert(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(convert);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, inner]) => [
        toCamel(key),
        convert(inner),
      ]),
    );
  }
  return value;
}

/** An error carrying the HTTP status and FastAPI's `detail`. */
export class ApiRequestError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as ApiError;
      if (body.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // A non-JSON error body is not worth a second failure; the status line will do.
    }
    throw new ApiRequestError(response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return convert(await response.json()) as T;
}

/** Create a game. The server randomises the human's side. */
export function createGame(): Promise<GameState> {
  return request<GameState>("/api/games", { method: "POST" });
}

/** Fetch one game's full state. */
export function fetchGame(id: number): Promise<GameState> {
  return request<GameState>(`/api/games/${id}`);
}

/** List games, newest first. */
export function listGames(): Promise<GameSummary[]> {
  return request<GameSummary[]>("/api/games");
}

/** Play the human's move and receive both plies plus the bot's trace. */
export function playMove(id: number, uci: string): Promise<GameState> {
  return request<GameState>(`/api/games/${id}/move`, {
    method: "POST",
    body: JSON.stringify({ uci }),
  });
}

/** Take back the last human move and the bot's reply. */
export function undoMove(id: number): Promise<GameState> {
  return request<GameState>(`/api/games/${id}/undo`, { method: "POST" });
}

/** Fetch the game as PGN. */
export async function fetchPgn(id: number): Promise<string> {
  const response = await fetch(`/api/games/${id}/pgn`);
  if (!response.ok) {
    throw new ApiRequestError(response.status, `${response.status} ${response.statusText}`);
  }
  return response.text();
}

/** Report whether the model is loaded. */
export function fetchHealth(): Promise<HealthState> {
  return request<HealthState>("/api/health");
}

export type { BotTrace, GameState, GameSummary, HealthState, StageTrace };
```

- [ ] **Step 8: Write `frontend/app/layout.tsx`**

```tsx
import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Laya Chess Bot",
  description: "Play chess against the fine-tuned Laya model and see how it decided.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```

- [ ] **Step 9: Replace `frontend/app/globals.css`**

Delete everything `create-next-app` generated and write:

```css
:root {
  --bg: #14161a;
  --panel: #1c1f26;
  --line: #2b303b;
  --text: #e6e8ec;
  --muted: #9aa1ae;
  --accent: #7aa2f7;
  --good: #7dcf94;
  --bad: #f2777a;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5;
}

a {
  color: var(--accent);
}

.shell {
  max-width: 1100px;
  margin: 0 auto;
  padding: 32px 24px 64px;
}

.topbar {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 24px;
}

.topbar h1 {
  font-size: 20px;
  font-weight: 650;
  letter-spacing: -0.01em;
  margin: 0;
}

.badge {
  font-size: 12px;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 10px;
}

.badge[data-state="ok"] {
  color: var(--good);
  border-color: color-mix(in srgb, var(--good) 40%, var(--line));
}

.badge[data-state="degraded"] {
  color: var(--bad);
  border-color: color-mix(in srgb, var(--bad) 40%, var(--line));
}

.layout {
  display: grid;
  grid-template-columns: minmax(320px, 1fr) minmax(320px, 420px);
  gap: 24px;
  align-items: start;
}

@media (max-width: 900px) {
  .layout {
    grid-template-columns: 1fr;
  }
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px;
}

.panel h2 {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  font-weight: 600;
  margin: 0 0 12px;
}

.toolbar {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 12px;
}

button {
  font: inherit;
  color: var(--text);
  background: #262a33;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 6px 14px;
  cursor: pointer;
}

button:hover:not(:disabled) {
  border-color: var(--accent);
}

button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.error {
  color: var(--bad);
  font-size: 13px;
  margin-top: 12px;
}

.movelist {
  font-family: ui-monospace, "Cascadia Code", Menlo, monospace;
  font-size: 13px;
  display: grid;
  grid-template-columns: 2.2rem 1fr 1fr;
  gap: 2px 8px;
}

.movelist .num {
  color: var(--muted);
  text-align: right;
}

.trace-stage {
  border-top: 1px solid var(--line);
  padding: 12px 0;
}

.trace-stage:first-of-type {
  border-top: 0;
}

.trace-stage header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  font-weight: 600;
}

.trace-stage .state {
  font-family: ui-monospace, "Cascadia Code", Menlo, monospace;
  font-size: 12px;
  color: var(--muted);
  white-space: pre-wrap;
  word-break: break-word;
  margin: 8px 0;
}

.options {
  display: grid;
  gap: 4px;
}

.option {
  display: grid;
  grid-template-columns: 1fr 90px 52px;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}

.option[data-chosen="true"] {
  color: var(--good);
  font-weight: 600;
}

.bar {
  height: 4px;
  background: var(--line);
  border-radius: 999px;
  overflow: hidden;
}

.bar span {
  display: block;
  height: 100%;
  background: var(--accent);
}

.spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid var(--line);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  vertical-align: -2px;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
```

- [ ] **Step 10: Typecheck**

Run from `frontend/`:
```bash
npm run typecheck
```

Expected: no errors. `app/page.tsx` is still the generated placeholder, which typechecks.

---

### Task 10: `Board` and the game page

**Files:**
- Create: `frontend/components/Board.tsx`
- Create: `frontend/app/games/[id]/page.tsx`

**Interfaces:**
- Consumes: `react-chessboard`'s `Chessboard` component, `lib/api`'s `fetchGame`, `playMove`, `undoMove`, `fetchPgn`, and the `GameState` type.
- Produces: `Board({ fen, legalMoves, lastMove, orientation, disabled, onMove, thinking })` where `onMove(square: string): void` receives the destination square; `app/games/[id]/page.tsx` as a client page.

Task 11 owns the other components and `app/page.tsx`, so this task does not touch them.

- [ ] **Step 1: Write `frontend/components/Board.tsx`**

```tsx
"use client";

import { Chessboard } from "react-chessboard";
import type { Square } from "react-chessboard";

/** Props for `Board`. */
interface BoardProps {
  /** Current position in FEN. */
  fen: string;
  /** Legal moves in UCI, used to highlight the squares a piece may reach. */
  legalMoves: string[];
  /** The most recent move in UCI, highlighted on the board. */
  lastMove: string | null;
  /** Which colour is at the bottom. */
  orientation: "white" | "black";
  /** Reject input, e.g. while the bot is thinking. */
  disabled: boolean;
  /** Called with the destination square once a legal move is completed. */
  onMove: (square: string) => void;
  /** Whether the bot is currently deciding. */
  thinking: boolean;
}

/**
 * The board, plus the small amount of state that dragging a piece needs.
 *
 * `react-chessboard` reports a completed drag as a destination square. Translating that
 * into a UCI move means finding the legal move that ends there, which is why `legalMoves`
 * is a prop rather than something this component recomputes from the FEN.
 */
export default function Board({
  fen,
  legalMoves,
  lastMove,
  orientation,
  disabled,
  onMove,
  thinking,
}: BoardProps) {
  const lastMoveSquares = lastMove
    ? ([lastMove.slice(0, 2), lastMove.slice(2, 4)] as Square[])
    : undefined;

  return (
    <div style={{ opacity: disabled ? 0.75 : 1, transition: "opacity 120ms" }}>
      <Chessboard
        options={{
          position: fen,
          orientation,
          squareStyles: highlightStyles(legalMoves),
          lastMove: lastMoveSquares,
          allowDragging: !disabled,
          enableResizing: true,
          darkSquareStyle: { backgroundColor: "#4a5568" },
          lightSquareStyle: { backgroundColor: "#cbd5e0" },
          boardStyle: { borderRadius: "8px", overflow: "hidden" },
        }}
        onPieceDrop={(squares: Square[]) => onMove(squares[1])}
      />
      {thinking ? (
        <p className="thinking">
          <span className="spinner" /> Laya is choosing a move…
        </p>
      ) : null}
    </div>
  );
}

/**
 * Map legal moves onto square styles, so a player can see where a piece may go.
 *
 * @param legalMoves Legal moves in UCI.
 * @returns A style map keyed by square name.
 */
function highlightStyles(legalMoves: string[]): Record<string, Record<string, string>> {
  const styles: Record<string, Record<string, string>> = {};
  for (const move of legalMoves) {
    styles[move.slice(2, 4)] = { backgroundColor: "rgba(122, 162, 247, 0.35)" };
  }
  return styles;
}
```

`react-chessboard` v4 uses the `options` prop. If the installed version still uses the v3 top-level props, `npm run typecheck` will fail and the fix is to spread `options` at the top level instead; do not rewrite the component around it.

- [ ] **Step 2: Write `frontend/app/games/[id]/page.tsx`**

```tsx
"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import Board from "@/components/Board";
import DecisionTrace from "@/components/DecisionTrace";
import MoveList from "@/components/MoveList";
import { ApiRequestError, fetchGame, fetchPgn, playMove, undoMove } from "@/lib/api";
import type { GameState } from "@/lib/types";

/**
 * One game: the board, the move list, the bot's reasoning, and the controls.
 *
 * The server's response replaces local state wholesale after every action. Nothing is
 * advanced optimistically, because at 1-8 s per bot move an optimistic board would sit
 * there being wrong.
 */
export default function GamePage() {
  const params = useParams<{ id: string }>();
  const gameId = Number(params.id);

  const [game, setGame] = useState<GameState | null>(null);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGame(gameId)
      .then(setGame)
      .catch((cause: unknown) => setError(describe(cause)));
  }, [gameId]);

  async function run(action: () => Promise<GameState>) {
    setThinking(true);
    setError(null);
    try {
      setGame(await action());
    } catch (cause: unknown) {
      setError(describe(cause));
    } finally {
      setThinking(false);
    }
  }

  function handleMove(destination: string) {
    if (!game) {
      return;
    }
    // react-chessboard has already visually moved the piece; the server decides whether
    // the move was legal and what the position becomes.
    void run(() => playMove(game.id, `${originOf(game, destination)}${destination}`));
  }

  if (!game) {
    return (
      <main className="shell">
        <p className="error">{error ?? "Loading…"}</p>
        <p>
          <Link href="/">Back</Link>
        </p>
      </main>
    );
  }

  return (
    <main className="shell">
      <div className="topbar">
        <h1>Game {game.id}</h1>
        <span className="badge">{describeStatus(game)}</span>
      </div>

      <div className="layout">
        <section>
          <Board
            fen={game.fen}
            legalMoves={game.legalMoves}
            lastMove={game.lastMove}
            orientation={game.humanSide}
            disabled={thinking || game.status !== "active"}
            onMove={handleMove}
            thinking={thinking}
          />
          <div className="toolbar">
            <button onClick={() => void run(() => undoMove(game.id))} disabled={thinking || game.moves.length === 0}>
              Undo move
            </button>
            <a href={`/api/games/${game.id}/pgn`}>
              <button type="button">Export PGN</button>
            </a>
            <Link href="/">
              <button type="button">New game</button>
            </Link>
          </div>
          {error ? <p className="error">{error}</p> : null}
        </section>

        <aside className="panel">
          <h2>Moves</h2>
          <MoveList moves={game.moves} />
        </aside>

        <section className="panel">
          <h2>How Laya decided</h2>
          <DecisionTrace trace={game.botTrace} />
        </section>
      </div>
    </main>
  );
}

/**
 * Find the origin square of the piece that can reach a destination.
 *
 * @param game The current game state.
 * @param destination The square the piece was dropped on.
 * @returns The origin square, or an empty string when no legal move ends there.
 */
function originOf(game: GameState, destination: string): string {
  const match = game.legalMoves.find((move) => move.slice(2, 4) === destination);
  return match ? match.slice(0, 2) : "";
}

/**
 * Turn an unknown throw into a sentence.
 *
 * @param cause Whatever was thrown.
 * @returns A message for the user.
 */
function describe(cause: unknown): string {
  return cause instanceof ApiRequestError ? cause.message : String(cause);
}

/**
 * Summarise the game status for the header badge.
 *
 * @param game The current game state.
 * @returns A short human-readable status.
 */
function describeStatus(game: GameState): string {
  if (game.status === "active") {
    return game.isCheck ? "check" : "in play";
  }
  const winner = game.status === "white_won" ? "White" : game.status === "black_won" ? "Black" : "Nobody";
  const how = game.termination ? ` by ${game.termination.replace("_", " ")}` : "";
  return `${winner} wins${how}`;
}
```

`fetchPgn` is imported but unused because the toolbar links straight at the URL; drop it from the import list if `tsc` or the linter objects.

- [ ] **Step 3: Typecheck**

Run from `frontend/`:
```bash
npm run typecheck
```

Expected: errors about the missing `DecisionTrace` and `MoveList` components. They are Task 11; create nothing here. Move to Task 11 and re-run the check at the end.

---

### Task 11: `DecisionTrace`, `MoveList`, and the home page

**Files:**
- Create: `frontend/components/DecisionTrace.tsx`
- Create: `frontend/components/MoveList.tsx`
- Create: `frontend/app/page.tsx`

**Interfaces:**
- Consumes: `BotTrace` from `lib/types`, `createGame`/`fetchHealth`/`listGames` from `lib/api`.
- Produces: `DecisionTrace({ trace })`, `MoveList({ moves })`, and a home page that creates games and links to `/games/[id]`.

Task 11 owns `app/globals.css` only if it needs a class Task 10 did not define; Task 9 already defined every class used, so it should not need to.

- [ ] **Step 1: Write `frontend/components/DecisionTrace.tsx`**

```tsx
"use client";

import type { BotTrace, StageTrace } from "@/lib/types";

/** Props for `DecisionTrace`. */
interface DecisionTraceProps {
  /** The most recent decision, or `null` before the bot has moved. */
  trace: BotTrace | null;
}

/**
 * Show how the bot reached its move: every stage, every option, every probability.
 *
 * With 49.7% per-stage accuracy this is the useful part of the app. Seeing that the bot
 * put 0.71 on a square it did not play explains a blunder; showing only the move does not.
 */
export default function DecisionTrace({ trace }: DecisionTraceProps) {
  if (!trace) {
    return <p className="state">The bot has not moved yet.</p>;
  }

  return (
    <div>
      {trace.stages.map((stage) => (
        <Stage key={stage.questionId + stage.latencyMs} stage={stage} />
      ))}
      <p className="state">
        Chose <strong>{trace.move}</strong> in {(trace.totalLatencyMs / 1000).toFixed(1)}s
        {trace.forced ? " — only one move was available" : ""}
        {trace.tieBreak ? ` — ${trace.tieBreak}` : ""}
      </p>
    </div>
  );
}

/** Props for `Stage`. */
interface StageProps {
  /** The stage to render. */
  stage: StageTrace;
}

/**
 * Render one question: its state text, its options with probabilities, and its latency.
 *
 * @param stage The stage to render.
 * @returns The stage block.
 */
function Stage({ stage }: StageProps) {
  const entries = Object.entries(stage.probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="trace-stage">
      <header>
        <span>
          {stage.stage}
          {stage.skippedReason ? " — skipped" : stage.choice ? ` — ${stage.choice}` : ""}
        </span>
        <span className="badge">{stage.skippedReason ?? `${(stage.latencyMs / 1000).toFixed(1)}s`}</span>
      </header>

      {!stage.skippedReason ? <p className="state">{stage.instructions}</p> : null}
      {stage.skippedReason ? <p className="state">{stage.skippedReason}</p> : null}

      <div className="options">
        {entries.map(([key, probability]) => (
          <div className="option" data-chosen={key === stage.choice} key={key}>
            <span>
              {key} {stage.options[key] ? `· ${stage.options[key]}` : ""}
            </span>
            <span className="bar">
              <span style={{ width: `${Math.max(2, Math.round(probability * 100))}%` }} />
            </span>
            <span>{probability.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Write `frontend/components/MoveList.tsx`**

```tsx
"use client";

/** Props for `MoveList`. */
interface MoveListProps {
  /** The move list in UCI. */
  moves: string[];
}

/**
 * Show the moves as numbered UCI pairs.
 *
 * UCI rather than SAN because the backend stores UCI and adding a chess library to the
 * frontend to re-derive SAN would duplicate rules the server already owns.
 *
 * @param props The component props.
 * @returns A three-column grid of move numbers and moves.
 */
export default function MoveList({ moves }: MoveListProps) {
  if (moves.length === 0) {
    return <p className="state">No moves yet.</p>;
  }

  const rows = [];
  for (let index = 0; index < moves.length; index += 2) {
    rows.push(
      <div key={index} style={{ display: "contents" }}>
        <span className="num">{index / 2 + 1}.</span>
        <span>{moves[index]}</span>
        <span>{moves[index + 1] ?? ""}</span>
      </div>,
    );
  }

  return <div className="movelist">{rows}</div>;
}
```

- [ ] **Step 3: Write `frontend/app/page.tsx`**

```tsx
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ApiRequestError, createGame, fetchHealth, listGames } from "@/lib/api";
import type { GameSummary, HealthState } from "@/lib/types";

/**
 * The landing page: start a new game, or resume an old one.
 *
 * The human's colour is decided by the server, so this page cannot offer a colour choice
 * even if it wanted to; that is deliberate, and it is why a new game is one button.
 */
export default function HomePage() {
  const [health, setHealth] = useState<HealthState | null>(null);
  const [games, setGames] = useState<GameSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
    listGames().then(setGames).catch(() => setGames([]));
  }, []);

  async function newGame() {
    setBusy(true);
    setError(null);
    try {
      const game = await createGame();
      window.location.href = `/games/${game.id}`;
    } catch (cause: unknown) {
      setError(cause instanceof ApiRequestError ? cause.message : String(cause));
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <div className="topbar">
        <h1>Laya Chess Bot</h1>
        {health ? (
          <span className="badge" data-state={health.status}>
            {health.modelLoaded ? "model loaded" : "model unavailable"}
          </span>
        ) : null}
      </div>

      <div className="layout">
        <section className="panel">
          <h2>Play</h2>
          <p className="state">
            You will be given a colour at random. The bot decides its move by answering
            three questions about the position, and shows you its reasoning afterwards.
          </p>
          <div className="toolbar">
            <button onClick={newGame} disabled={busy || health?.modelLoaded === false}>
              New game
            </button>
          </div>
          {error ? <p className="error">{error}</p> : null}
        </section>

        <section className="panel">
          <h2>Recent games</h2>
          {games.length === 0 ? (
            <p className="state">Nothing yet.</p>
          ) : (
            <ul>
              {games.slice(0, 10).map((game) => (
                <li key={game.id}>
                  <Link href={`/games/${game.id}`}>
                    Game {game.id} — you played {game.humanSide}, {game.moveCount} plies,{" "}
                    {game.status.replace("_", " ")}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </main>
  );
}
```

- [ ] **Step 4: Typecheck the whole frontend**

Run from `frontend/`:
```bash
npm run typecheck
```

Expected: no errors. Task 10's expected errors are gone.

- [ ] **Step 5: Build**

Run from `frontend/`:
```bash
npm run build
```

Expected: a successful production build. This is the check that catches server/client boundary mistakes, which `tsc` alone does not.

---

### Task 12: Correct the guide and write the README

**Files:**
- Modify: `laya_chess_bot_move_guide.md`
- Create: `README.md`

**Interfaces:**
- Consumes: the prompt contract from `backend/chessbot/prompts.py`, which is the authority.
- Produces: a guide that matches the trained model, and a README that says how to run it.

This task has no code dependency, so it can run in Wave B alongside the others.

- [ ] **Step 1: Correct `laya_chess_bot_move_guide.md`**

Fix exactly these errors, and change nothing else in the file:

| Section | Wrong | Right |
| --- | --- | --- |
| Directions | Four directions listed with an extra `anti-diagonal` framing | Exactly `horizontal`, `vertical`, `diagonal_nw_se`, `diagonal_ne_sw` |
| Direction labels | ASCII hyphen | `Diagonal NW–SE` and `Diagonal NE–SW`, en dash U+2013 |
| Stage B instructions | Anything other than the single line | `Choose the best direction for the chess piece.` |
| Stage C state | No suffix or a different suffix | Append ` Selected direction: <direction>.` |
| FEN source | Invented example FENs | Real FEN from `model/laya_finetuned_1000_after_finetuning.json` |

State at the top of the file that the authority is the prediction JSON and `backend/chessbot/prompts.py`, and that the strings are byte-exact including the en dash. Add the `test_prompts.py` pointer so a future retrain has a failing test to look at.

- [ ] **Step 2: Write `README.md`**

```markdown
# Laya Chess Bot

Play chess against the fine-tuned [Laya](https://github.com/rohanrhu/laya) checkpoint in
`model/laya_finetuned_typed_decisions`, and see how it decided every move.

The model cannot name a move. It answers multiple-choice questions about a position, so a
move is recovered by narrowing: which piece, then which direction for a bishop, rook or
queen, then which destination square. Every option offered at every stage is derived from
the position's legal moves, which is what makes an illegal move structurally impossible
rather than merely unlikely.

## Requirements

- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- Node 20 or newer

## Run it

Two terminals.

```bash
# terminal 1 — backend, on :8000
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

```bash
# terminal 2 — frontend, on :3000
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. `next.config.ts` rewrites `/api/*` to the backend, so the
browser only ever talks to one origin.

## Tests

```bash
cd backend
uv run pytest
```

No test loads the 842 MB checkpoint. `app.deps.get_agent` is overridden with a
`FakeAgent`, so the suite runs in seconds.

## How it is put together

```
backend/chessbot/prompts.py   the trained prompt contract, byte-exact
backend/chessbot/engine.py    pure python-chess helpers
backend/chessbot/agent.py     the only module that imports laya
backend/chessbot/decider.py   the three-stage chain
backend/app/                  FastAPI, SQLite, one ORM model
frontend/                     Next.js App Router, react-chessboard
```

## The prompt contract is load-bearing

`model/laya_finetuned_1000_after_finetuning.json` is the record of what the model was
actually trained on, and it is the authority for every string in `chessbot/prompts.py` —
including `Diagonal NW–SE`, whose dash is an en dash (U+2013) and not a hyphen.
`tests/test_prompts.py` reads that JSON and asserts the strings match byte-for-byte, so a
retrain with a new format fails a test instead of quietly degrading the bot.

## Configuration

Environment variables, all optional:

| Variable | Default | Notes |
| --- | --- | --- |
| `LAYA_MODEL_DIR` | `../model/laya_finetuned_typed_decisions` | Path to the checkpoint |
| `LAYA_DEVICE` | `cpu` | Set `cuda` if a GPU appears |
| `LOG_LEVEL` | `INFO` | `DEBUG` narrates the whole decision chain |
| `DATABASE_URL` | `sqlite:///app/chessbot.db` | |

## Performance

Inference is CPU-only on an Intel UHD machine with no GPU, so expect roughly 1-8 seconds
per bot move, two or three model calls per move. Each move records its per-stage latencies,
so the real number is visible from the first game onwards. Positions with a single legal
move skip the questions entirely.

If the checkpoint fails to load, the app still starts, `/api/health` reports `degraded`,
and the move endpoint returns `503` with the history untouched.
```

- [ ] **Step 3: Check the guide no longer contradicts the code**

Run from the repo root:
```bash
rg -n "anti-diagonal|NW-SE|NE-SW" laya_chess_bot_move_guide.md
```

Expected: no `NW-SE` or `NE-SW` with an ASCII hyphen, and no `anti-diagonal`. Any hit is a missed correction.

---

### Task 13: End-to-end verification

**Files:**
- None. This task only runs things.

- [ ] **Step 1: Full backend suite, no model loaded**

Run from `backend/`:
```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Expected: all tests pass; ruff reports no findings.

- [ ] **Step 2: Frontend build**

Run from `frontend/`:
```bash
npm run typecheck
npm run build
```

Expected: no type errors, successful build.

- [ ] **Step 3: Start the backend with the real checkpoint and check health**

Run from `backend/`:
```bash
uv run uvicorn app.main:app --port 8000
```

Expected: startup logs the model dir and device, then `agent loaded`. In another terminal:

```bash
curl -s http://localhost:8000/api/health
```

Expected: `{"status":"ok","model_loaded":true,"device":"cpu","model_dir":"..."}`.

- [ ] **Step 4: One real game, end to end**

```bash
curl -s -X POST http://localhost:8000/api/games
```

Play a move if the human is white:

```bash
curl -s -X POST http://localhost:8000/api/games/1/move -H "Content-Type: application/json" -d '{"uci":"e2e4"}'
```

Expected: `moves` has two entries, `bot_trace.stages` has two or three entries each with
real probabilities, and `totalLatencyMs` is in the thousands. Record the number — it is the
real per-move cost, replacing the 1-8 s estimate in the README.

- [ ] **Step 5: Undo and PGN**

```bash
curl -s -X POST http://localhost:8000/api/games/1/undo
curl -s http://localhost:8000/api/games/1/pgn
```

Expected: `moves` is back to `[]`; the PGN names both players.

- [ ] **Step 6: Frontend against the running backend**

Run from `frontend/`:
```bash
npm run dev
```

Open <http://localhost:3000>, start a game, play a move, and confirm: the board does not
move until the response arrives, the spinner shows while thinking, the trace panel lists
every stage with probabilities, undo takes back both plies, and the PGN link downloads.

- [ ] **Step 7: Confirm nothing was committed**

Run from the repo root:
```bash
git status --short
git log --oneline -1
```

Expected: the working tree shows the new files, and `HEAD` is still the initial commit.
Nothing is staged, nothing is committed.

