# Laya Chess Bot — Design Spec

- **Date:** 2026-10-06
- **Status:** Draft for review
- **Stack:** FastAPI backend (own uv project in `backend/`) + Next.js frontend

---

## 1. Goal

Build a web application that lets a human play chess against a bot whose moves come from the
fine-tuned Laya model in `model/laya_finetuned_typed_decisions/`. The human plays one side; the
model plays the other. The application must show, after every bot move, the stage-by-stage
decision the model made.

The model does not emit moves. It emits a probability distribution over the options of a
multiple-choice question. A move is produced by chaining two or three such questions, where every
option offered is drawn from the current position's legal moves. Therefore the bot can only ever
return a legal move.

## 2. What already exists

| Asset | Detail |
| --- | --- |
| `model/laya_finetuned_typed_decisions/` | Fine-tuned checkpoint, 842 MB safetensors, ModernBERT-large encoder, tokenizer present. Gitignored. |
| `model/laya_finetuned_1000_after_finetuning.json` | 1,000 held-out predictions **with the exact prompt text for each row**. Authoritative source for the prompt contract. |
| `model/laya_baseline_1000_before_finetuning.json` | Pre-finetuning predictions, for comparison only. |
| `finetuning_notebook/laya-finetuning-on-chess.ipynb` | Training and evaluation notebook. |
| `laya_chess_bot_move_guide.md` | Human-written guide. **Stage B is wrong** — see §4.2. |

### 2.1 Measured facts about the environment

These constrain the design and were verified, not assumed.

- **No GPU.** `nvidia-smi` is absent; the only display adapter is Intel UHD Graphics. Inference runs
  on CPU. The notebook's 40 ms per call was measured on 2×T4 Kaggle GPUs; expect roughly
  **1–8 seconds per bot move** here, plus a slow first load of the checkpoint.
- **Python 3.14.** `.python-version` specified 3.14; `torch==2.14.1` and `laya==0.3.28` both
  resolve for it.
- **Node 24.12.0 / npm 11.6.2** available. No pnpm, yarn, or bun.

### 2.2 Model performance

From the notebook, on 1,000 held-out rows. Each row is **one stage question**, not one full move.

| Metric | Base Laya | Fine-tuned |
| --- | --- | --- |
| Accuracy | 26.1% | 49.7% |
| Soft accuracy | 0.291 | 0.372 |
| Brier score | 0.411 | 0.268 |

The bot's move accuracy is therefore meaningfully below 50%, and per-stage error compounds across a
two- or three-call chain. The application must treat this as a weak-but-legal opponent, which is
why undo (§7.3) and the decision trace (§6.5) are first-class features rather than extras.

---

## 3. Goals and non-goals

### Goals

1. A playable chess UI where a human can make a move and receive a legal Laya-chosen reply.
2. The bot's decision is fully legible: every stage, every option, every probability, every latency.
3. The trained prompt contract is encoded in exactly one place and is protected by a test.
4. Move selection is testable without loading the model.

### Non-goals

- Bot versus bot, or any form of self-play.
- PGN import, opening books, or an engine (Stockfish) in the loop.
- Move selection by sampling. Greedy only — see §6.4.
- Alembic or any schema migration tooling. One table, no history to preserve.
- Frontend component test suites.

---

## 4. The trained prompt contract

### 4.1 Authority

`model/laya_finetuned_1000_after_finetuning.json` is the authority, **not** the guide. Every string
in §4.3 was extracted from that file and cross-checked across all 1,000 rows: 31 distinct state
shapes, 3 distinct instructions, 4 direction labels, 3 question ids.

The guide is wrong in three ways that would each silently degrade the bot:

| Guide says | Actually trained | Consequence if followed |
| --- | --- | --- |
| Three directions: `horizontal` / `vertical` / `diagonal` | **Four:** `horizontal`, `vertical`, `diagonal_nw_se`, `diagonal_ne_sw` | Every rook, bishop and queen move mis-classified |
| Direction instructions are `"<copy from dataset generator>"` | `Choose the best direction for the chess piece.` | Untrained prompt text |
| Destination state has no direction clause | `... Selected piece: R. Selected direction: horizontal.` | Wrong input format for every slider destination |

Stage A and Stage C wording in the guide is correct and matches the notebook.

### 4.2 Exact strings

```python
STATE_PREFIX = "Chess position. FEN: {fen}. Side to move: {side}."

INSTRUCTIONS = {
    "piece_selection":      "Choose the best chess piece for the move.",
    "direction_selection":  "Choose the best direction for the chess piece.",
    "destination_selection": "Choose the best chess piece destination.",
}

PIECE_LABELS = {"P": "Pawn", "N": "Knight", "B": "Bishop",
                "R": "Rook", "Q": "Queen", "K": "King"}

DIRECTIONS = (
    ("horizontal",     "Horizontal"),
    ("vertical",       "Vertical"),
    ("diagonal_nw_se", "Diagonal NW–SE"),
    ("diagonal_ne_sw", "Diagonal NE–SW"),
)

SLIDING_PIECES = frozenset("BRQ")
```

`Diagonal NW–SE` and `Diagonal NE–SW` use **U+2013 EN DASH**, not ASCII hyphen. Replacing it with
a hyphen is a silent accuracy loss. `test_prompts.py` pins the codepoint.

### 4.3 The four state shapes

Built as `STATE_PREFIX` plus the following suffix.

| Stage | Suffix |
| --- | --- |
| Piece | `Stage: piece.` |
| Direction | `Stage: direction. Selected piece: {piece}.` |
| Destination, P/N/K | `Stage: destination. Selected piece: {piece}.` |
| Destination, B/R/Q | `Stage: destination. Selected piece: {piece}. Selected direction: {direction}.` |

`{side}` is the literal `white` or `black`. `{fen}` is `board.fen()` verbatim, including the
halfmove and fullmove counters.

### 4.4 Direction classification

Four classes. With `df = to_file − from_file` and `dr = to_rank − from_rank`:

| Condition | Direction | Check |
| --- | --- | --- |
| `dr == 0` | `horizontal` | a1 → a4 |
| `df == 0` | `vertical` | a1 → a5 |
| `df == dr` | `diagonal_ne_sw` | a1 → b2 goes up and to the right |
| `df == -dr` | `diagonal_nw_se` | h1 → a8 goes up and to the left |

### 4.5 Destination labels

`chess.square_name(move.to_square)`, with `=Q`, `=R`, `=B` or `=N` appended when
`move.promotion` is set. Matches the `e8=Q` form present in the dataset.

### 4.6 Option ordering

Options are deduplicated and emitted in ascending square index, so a given position always produces
the same `criteria` dict. The dataset's option order is not sorted, which indicates it was shuffled
upstream; Laya scores each option independently, so order does not affect the result, but a
deterministic order makes traces diffable across runs.

---

## 5. Project structure

```
Laya-ChessBot/
├── backend/                        # own uv project
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version             # 3.14
│   ├── .venv/
│   ├── chessbot/                   # pure Python — no FastAPI import anywhere
│   │   ├── __init__.py
│   │   ├── prompts.py              # §4. The trained contract. Single source of truth.
│   │   ├── agent.py                # LayaAgent wrapper
│   │   ├── engine.py               # python-chess helpers
│   │   └── decider.py              # (agent, board) -> Decision
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # create_app() + lifespan
│   │   ├── config.py               # Settings
│   │   ├── db.py                   # SQLAlchemy engine, session, Base
│   │   ├── models.py               # Game
│   │   ├── schemas.py              # pydantic models
│   │   ├── deps.py                 # get_agent, get_db
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── games.py
│   │       └── health.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_prompts.py
│   │   ├── test_directions.py
│   │   ├── test_engine.py
│   │   ├── test_decider.py
│   │   └── test_api.py
│   └── app/chessbot.db             # SQLite, gitignored
├── frontend/                       # Next.js
│   ├── package.json
│   ├── next.config.ts
│   ├── tsconfig.json
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                # new game + recent games
│   │   └── games/[id]/page.tsx
│   ├── components/
│   │   ├── Board.tsx
│   │   ├── MoveList.tsx
│   │   └── DecisionTrace.tsx
│   └── lib/
│       ├── api.ts
│       └── types.ts
├── model/                          # weights, gitignored
├── finetuning_notebook/
├── laya_chess_bot_move_guide.md    # Stage B needs correcting
└── docs/
```

---

## 6. The `chessbot` library

Django-free by construction, so move selection is unit-testable without the web framework and
could be lifted into a CLI harness later without modification.

### 6.1 `prompts.py`

Holds §4 verbatim. Exposes:

- `piece_state(board) -> str`
- `direction_state(board, piece) -> str`
- `destination_state(board, piece, direction | None) -> str`
- `piece_options(board, moves) -> dict[str, str]`
- `direction_options(moves) -> dict[str, str]`
- `destination_options(moves) -> dict[str, str]`

Every option builder derives its keys exclusively from the moves it is given, which are always a
subset of `board.legal_moves`. This is what makes illegal moves structurally impossible rather
than merely unlikely.

The module docstring records the §4.1 evidence, so a future reader knows why the strings look odd
and where to re-verify them.

### 6.2 `agent.py`

```python
class LayaAgent(Protocol):
    def predict_choice(self, state: str, question_id: str,
                       instructions: str, options: dict[str, str]) -> ChoiceResult: ...
```

`ChoiceResult` carries `choice: str`, `probabilities: dict[str, float]`, `latency_ms: float`.

The concrete `LocalLayaAgent` wraps `laya.Agent(model_dir, device=...)`. It holds a
`threading.Lock` held across `predict()`: the ASGI threadpool may run two moves concurrently, and a
single torch module shared across threads is not a safe assumption. `predict_choice` normalises the
returned probabilities over exactly the option keys it was given.

Agents are created once during application startup (§7.1) and injected via `Depends(get_agent)`.
No module-level global. This is what allows the test suite to substitute a fake and never load the
842 MB checkpoint.

### 6.3 `engine.py`

Pure helpers over `python-chess`: rebuild a board from a UCI move list by replay, report
side-to-move text, compute game status and termination, produce destination labels (§4.5), classify
direction (§4.4), and convert UCI to `chess.Move` with a parse error on failure.

### 6.4 `decider.py`

```python
def decide_move(agent: Agent, board: chess.Board) -> Decision
```

`Decision` is a frozen dataclass: `move: chess.Move`, `stages: list[StageCall]`,
`total_latency_ms: float`, `forced: bool`, `tie_break: str | None`.

`StageCall` is a frozen dataclass: `stage`, `question_id`, `state`, `instructions`,
`options: dict[str, str]`, `choice: str`, `probabilities: dict[str, float]`, `latency_ms: float`,
`skipped_reason: str | None`.

Chain:

1. **Piece.** Distinct piece types across all legal moves, ordered `P N B R Q K`. Predict. Filter
   candidates to the chosen type.
2. **Direction**, only if the piece is in `SLIDING_PIECES`. Distinct directions present among the
   candidates, in `DIRECTIONS` order. Predict. Filter.
3. **Destination.** Deduplicated destination labels of the remaining candidates, ordered by square
   index. Predict. Filter.

Resolve to one `chess.Move`.

**Greedy, not sampled.** `ChoiceResult.choice` is the argmax at every stage, making the bot
deterministic and its output reproducible. Sampling from `probabilities` is possible with the same
code but is not enabled: the notebook's 49.7% accuracy was measured under greedy decoding, and
stepwise argmax is not the same as maximising the product of stage probabilities. That alternative
is a deliberate non-goal (§3).

**Forced-move short-circuit.** If at any point the candidate list holds exactly one move, the
remaining questions are answered by that move, and the stage records
`skipped_reason="only one candidate move"`. `Decision.forced` is `True`. This is provably correct
and saves two model calls, which matters given the 1–8 s per-move budget.

**Fixed tie-break.** Two pieces of the same type can reach the same square — two rooks to e4 — and
no question in the chain distinguishes them. Rule: lowest UCI wins. When it fires, `tie_break`
records how many candidates matched and which was chosen, so it is visible in the UI rather than
guessed at.

**Never raise on a bad answer.** If `choice` is not among the candidates — it should not be,
because every option came from `legal_moves`, but the model is a neural net — fall back to the
highest-probability candidate, then to the first legal move in UCI order. Each fallback logs a
`WARNING` with the offending choice. An exception here would leave the bot unable to move and the
game hung with no diagnostic.

### 6.5 Trace construction

`decide_move` builds `list[StageCall]` as it goes. The same objects feed the API response and the
`DEBUG` log lines. They are never derived independently — a trace that disagrees with what actually
ran is worse than no trace.

---

## 7. Backend

### 7.1 Application and lifespan

`create_app()` builds the FastAPI app. Its lifespan handler:

- Opens the SQLite engine and runs `Base.metadata.create_all()`.
- Attempts to load the Laya agent once. **A failure here does not stop startup**; it logs the
  traceback at `ERROR`, leaves `model_loaded = False`, and the app serves normally. `/api/health`
  reports the degraded state and `POST /move` returns `503`. Booting degraded is preferable to an
  import crash, and it makes the degraded path testable.

### 7.2 Data model

```python
class Game(Base):
    __tablename__ = "games"
    id: int                # primary key
    human_side: str        # "white" | "black" — chosen at random, see §12.1
    status: str            # "active" | "white_won" | "black_won" | "draw"
    result: str | None     # "1-0" | "0-1" | "1/2-1/2"
    termination: str | None # "checkmate" | "stalemate" | "insufficient_material" | ...
    moves: JSON            # ["e2e4", "e7e5", ...] — the single source of truth
    last_trace: JSON       # most recent Decision trace, for the UI
    created_at: datetime
    updated_at: datetime
```

`moves` is authoritative and the board is rebuilt by replaying it on every request. There is
deliberately **no `fen` column**: a stored FEN can drift from the move history, and a silent
divergence between what is displayed and what the bot reasoned about is a genuinely nasty class of
bug. The current FEN is derived per request and sent in the response.

### 7.3 Endpoints

All under `/api`. All JSON except the PGN export, which is `text/plain`.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/api/health` | — | `{status, model_loaded, device, model_dir}` |
| `POST` | `/api/games` | — | `GameState` |
| `GET` | `/api/games` | — | `[GameSummary, ...]`, newest first |
| `GET` | `/api/games/{id}` | — | `GameState` |
| `POST` | `/api/games/{id}/move` | `{uci}` | `GameState` including `bot_trace` |
| `POST` | `/api/games/{id}/undo` | — | `GameState` |
| `GET` | `/api/games/{id}/pgn` | — | `text/plain` PGN |

`GameState` is the single response shape for every game endpoint: `id`, `fen`, `moves`, `turn`,
`human_side`, `status`, `result`, `termination`, `is_check`, `legal_moves`, `last_move`,
`bot_trace`.

The server sends `legal_moves` so client and server agree on legality by construction. The client
uses them for square highlighting and turn gating; **every move is still re-validated server-side**,
because a browser is not a trust boundary.

`POST /undo` pops **two** plies, not one. Undoing only the bot's reply would leave it on the human's
turn with nothing to do. It is a single slice operation, and with a weak model it is a feature
players reach for immediately.

### 7.4 `POST /move` flow

1. Replay `moves` to build the board.
2. Validate: game exists (`404`), status active (`409`), `board.turn` matches the non-human side
   (`409`), UCI parses (`400`), move is legal (`400`).
3. Push the human move.
4. If the game is over, record `status`, `result`, `termination`, persist, return. No model call.
5. Otherwise `await run_in_threadpool(decide_move, agent, board)`. This is the 1–8 s blocking
   section; it must not run on the event loop or every other request stalls behind it.
6. Push the bot move, recompute status and termination.
7. Persist `moves`, `last_trace`, `status`, `updated_at`.
8. Return the full `GameState`.

### 7.5 Configuration

`backend/app/config.py`, all overridable by environment variable:

| Setting | Default | Notes |
| --- | --- | --- |
| `LAYA_MODEL_DIR` | `../model/laya_finetuned_typed_decisions` | Resolved relative to the repo root |
| `LAYA_DEVICE` | `cpu` | No GPU on this machine; set `cuda` if one appears |
| `LOG_LEVEL` | `INFO` | `DEBUG` narrates the full decision chain |
| `DATABASE_URL` | `sqlite:///app/chessbot.db` | |

The root `.gitignore` covers `.venv/` at any depth but does not yet exclude the SQLite file.
Implementation adds `*.db` and `*.db-journal` to it.

### 7.6 Logging

Two logger trees, separated so the decision chain can be enabled without drowning in access logs.

- **`chessbot.*`** — the library. `DEBUG`: each stage's full state text, instructions, options,
  probabilities, choice, latency. `INFO`: the resolved move and total latency. `WARNING`: forced
  short-circuit, tie-break applied, invalid-choice fallback.
- **`app.*`** — the web layer. `INFO`: game created, human move accepted, bot reply with latency,
  game finished. `ERROR`: model load failure and mid-move exceptions, both with traceback.

Configured in `config.py`; uvicorn access logs remain at their default level.

### 7.7 Error handling

| Situation | Response | Rationale |
| --- | --- | --- |
| Model absent or failed to load | `503`, `GameState` in body | Nothing can be decided; the UI shows the degraded state |
| Model raises mid-move | Roll back the human move, `503`, `GameState` in body | History is unchanged, so the same move retries cleanly. The drag is not lost |
| Unknown game id | `404` | |
| Game already finished | `409` | |
| Not the human's turn | `409` | |
| Unparseable or illegal UCI | `400` | |

Rolling back is chosen over persisting the human move and telling the player to undo: it is less
code and a better experience. Every one of these logs at `INFO` or `ERROR`.

### 7.8 Comment convention

- Docstrings on every module, public class and public function, **Google style** (`Args:`,
  `Returns:`, `Raises:`) to match `laya`'s own convention.
- Comments only for reasoning the code cannot express: why the en-dash in `Diagonal NW–SE` is
  load-bearing, why `df == dr` maps to `ne_sw`, why lowest-UCI wins the tie, why the human move is
  rolled back. No narration of the following line. No commented-out code.
- The `prompts.py` docstring records the §4.1 evidence trail, so the strings are auditable.

### 7.9 Dependencies

```
fastapi
uvicorn[standard]
sqlalchemy
pydantic
pydantic-settings
python-chess
laya
torch
pytest
httpx          # FastAPI TestClient
ruff
```

---

## 8. Frontend

Next.js App Router, TypeScript, React. No separate state-management library; the board component
owns a small amount of local state.

### 8.1 API access

`next.config.ts` rewrites `/api/*` to `http://localhost:8000/api/*`. The browser therefore sees a
single origin in both dev and production, and no CORS configuration exists anywhere in the project.

### 8.2 Board

`react-chessboard` — drag-and-drop and click-to-move out of the box, last-move and check
highlighting, and built-in SVG pieces in v4, so there are no image assets to vendor and no npm
`chess.js` duplication.

On a completed move the component posts to `/api/games/{id}/move`, then replaces its position with
the returned `fen`. **The server's response is authoritative; the client never advances its own
state optimistically.** At 1–8 s per round trip an optimistic update would also be visibly wrong
every time the bot's reply differs from what was guessed.

A `thinking` flag disables the board and shows a spinner on the bot's square while the request is
in flight. One request in, one `GameState` out — no polling, matching the synchronous move
decision.

### 8.3 Components

| Component | Responsibility |
| --- | --- |
| `Board` | Board rendering, input handling, POST of the human move, `thinking` state |
| `MoveList` | SAN or UCI move pairs from `moves` |
| `DecisionTrace` | Collapsible panel: per stage, the state text, every option with its probability, the choice, and per-call latency |

`DecisionTrace` is the feature that makes this project worth having: with 49.7% per-stage accuracy,
seeing *why* the bot blundered is more useful than seeing that it did.

### 8.4 Frontend verification

`tsc --noEmit` plus manual play. A component test suite for two components is not worth the tooling
cost yet.

---

## 9. Testing

All backend tests run without the model. `get_agent` is overridden with a `FakeAgent`, so the
842 MB checkpoint is never loaded.

### 9.1 `test_prompts.py` — the guard that matters

Reads `model/laya_finetuned_1000_after_finetuning.json`, extracts the real state strings,
instructions, question ids and direction labels, and asserts that `prompts.py` reproduces them
byte-for-byte, including the en-dash codepoint.

This is the test that makes the guide's Stage B errors unrepeatable. If the model is ever retrained
with a new prompt format, this test fails and points at exactly what changed, instead of the bot
silently degrading.

### 9.2 Remaining suites

| File | Covers |
| --- | --- |
| `test_directions.py` | All four classes, parameterized over positions. Pins `h1→a8 = nw_se` and `a1→b2 = ne_sw` |
| `test_engine.py` | Destination labels, promotion `e8=Q`, history replay, status and termination detection, UCI parse failure |
| `test_decider.py` | Scripted `FakeAgent`. P/N/K path (2 calls), B/R/Q path (3 calls), forced short-circuit, tie-break, invalid-choice fallback, promotion, castling, en passant |
| `test_api.py` | `TestClient` with the agent overridden. Create game, legal move, illegal `400`, out-of-turn `409`, finished `409`, unknown `404`, undo, PGN export |

---

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| CPU inference is far slower than expected, making play unpleasant | The design's per-stage latencies are recorded from the first game onward, so the real number is known immediately rather than estimated. Forced-move short-circuit trims some calls |
| Model load takes a long time on startup | Lifespan loads once; `/api/health` distinguishes "loading" from "failed" |
| A retrain changes the prompt format | `test_prompts.py` fails loudly against the real prediction file |
| Move choice degrades because a string drifted | Same test, plus the en-dash codepoint assertion |
| Torch on Python 3.14 has gaps | Dependencies resolve for 3.14 today; `uv sync` is the first implementation step and will surface any problem before code is written |

---

## 11. Out of scope, deliberately

Bot versus bot and self-play evaluation; sampling-based move selection; sampling temperature
tuning; opening books; PGN import; Alembic; frontend component tests; deployment.

---

## 12. Resolved decisions

1. **Human side is random.** `POST /api/games` takes no `human_side`; the server picks white or
   black with `random.choice` at creation and the UI displays which side the player has. This
   removes a decision the player has to make before playing and makes both colours exercised
   equally during casual use.
2. **`laya_chess_bot_move_guide.md` is corrected in place.** A document that is known to contain a
   wrong prompt contract is an active hazard in the repository, so §4.2 and §4.3 of the guide are
   replaced with the verified strings, and the three errors are listed at the top of the file so a
   reader knows the guide once said something different.

