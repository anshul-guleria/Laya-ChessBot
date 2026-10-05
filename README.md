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

Inference is CPU-only on an Intel UHD machine with no GPU. Measured on this machine, not
estimated:

| Step | Time |
| --- | --- |
| Checkpoint load, once at startup | ~11 s |
| One multiple-choice call | 0.4-0.8 s |
| A whole bot move, two or three calls | 1.1-1.4 s |

Each move records its per-stage latencies, so the real number is visible from the first
game onwards. Positions with a single legal move skip the questions entirely.

If the checkpoint fails to load, the app still starts, `/api/health` reports `degraded`,
and the move endpoint returns `503` with the history untouched.
