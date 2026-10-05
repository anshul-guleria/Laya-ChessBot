# How the Laya Chess Bot Makes One Move

This document explains, step by step, how the fine-tuned Laya model is used as a chess bot: what goes **in**, what comes **out**, and how those outputs become one legal move on the board.

> ### Provenance — read this before trusting any string below
>
> Every prompt string in this guide was verified against
> `model/laya_finetuned_1000_after_finetuning.json`: 1,000 held-out predictions, each
> carrying the exact state text, instructions, question id and option labels that produced
> it. **That file is the authority** for the prompt contract, and
> `backend/chessbot/prompts.py` is its single implementation.
>
> The strings are **byte-exact**:
>
> - `Diagonal NW–SE` and `Diagonal NE–SW` contain U+2013 EN DASH, not an ASCII hyphen.
> - The destination-stage direction clause is exactly ` Selected direction: <direction>.` —
>   one leading space, one trailing period, and the direction **key**, not the display label.
>
> This document was wrong about three things and has been corrected: it described **three**
> directions instead of **four**, it left the direction instruction as an unfilled
> placeholder, and it omitted the destination-stage direction clause. Re-verify against the
> JSON, not against this prose.
>
> `backend/tests/test_prompts.py` reads that JSON and asserts every string byte-for-byte,
> including the en-dash codepoint. A retrain with a new prompt format fails that test
> instead of silently degrading the bot.

---

## 1. The core idea (read this first)

Laya is **not** a chess engine and does **not** output a move like `e2e4` directly.

Laya is a **multiple-choice decision model**. You give it:

1. a **state** (text describing the position), and
2. a **question** with a fixed list of options,

and it returns a **probability for every option**.

A chess move is therefore built from **a short chain of multiple-choice questions**. Each answer narrows the move down until exactly one legal move remains.

| Piece chosen | Questions asked per move | Chain |
|---|---|---|
| Pawn (P), Knight (N), King (K) | **2** | piece → destination |
| Bishop (B), Rook (R), Queen (Q) | **3** | piece → direction → destination |

> Every option offered to the model comes from `board.legal_moves`. The model can only choose among legal options, so it can never output an illegal move.

---

## 2. What you need

| Item | Detail |
|---|---|
| Fine-tuned model folder | `laya_finetuned_typed_decisions/` (output of the training notebook, Step 9) |
| Tokenizer layout | The folder needs a `tokenizer/` sub-folder. Notebook Step 10 creates it by copying the tokenizer files. |
| Libraries | `laya`, `python-chess`, `torch` (GPU used in the notebook: `device="cuda"`) |
| Model call | `agent.predict(state, questions)` (the same call used in the notebook's `evaluate_agent`) |

Load the agent **once** at startup, not once per move:

```python
import laya
agent = laya.Agent("/kaggle/working/laya_finetuned_typed_decisions", device="cuda")
```

---

## 3. The model call: exact input and output

### Input

```python
result = agent.predict(state, questions)
```

- `state` is a **plain string**.
- `questions` is a dict with exactly **one** question of type `"choice"`:

```python
questions = {
    "<question_id>": {
        "type": "choice",
        "instructions": "<instruction text>",
        "criteria": {"<option_key>": "<option label>", ...}   # legal options only
    }
}
```

### Output

```python
answer = result["answers"]["<question_id>"]
answer["choice"]          # the option key with the highest probability
answer["probabilities"]   # {option_key: probability, ...}
```

---

## 4. The question types (copy these strings exactly)

The model was trained on these exact formats. The bot must produce **identical wording**, otherwise the model sees text it was not trained on.

### Stage A: Piece (confirmed against the prediction JSON)

```text
state:
  Chess position. FEN: <fen>. Side to move: <white|black>. Stage: piece.

question id:    piece_selection
instructions:   Choose the best chess piece for the move.
criteria:       only piece types that have at least one legal move
                {"P": "Pawn", "N": "Knight", "B": "Bishop", "R": "Rook", "Q": "Queen", "K": "King"}
```

### Stage B: Direction (only for B, R, Q)

Verified in the prediction JSON, not inferred.

```text
state:
  Chess position. FEN: <fen>. Side to move: <white|black>. Stage: direction. Selected piece: <B|R|Q>.

question id:    direction_selection
instructions:   Choose the best direction for the chess piece.
criteria:       only the directions that have at least one legal move for that piece
                  horizontal     -> Horizontal
                  vertical       -> Vertical
                  diagonal_nw_se -> Diagonal NW–SE
                  diagonal_ne_sw -> Diagonal NE–SW
```

There are **four** direction classes, not three. The two diagonals are separate classes rather
than one class split by handedness, so every bishop, rook and queen move is classified into
one of exactly four buckets.

The two diagonal labels use U+2013 EN DASH. Substituting an ASCII hyphen produces a string
the model was never trained on, which costs accuracy silently.

### Stage C: Destination (confirmed against the prediction JSON)

```text
state (P, N, K):
  Chess position. FEN: <fen>. Side to move: <white|black>. Stage: destination. Selected piece: <P|N|K>.

state (B, R, Q), after the direction was chosen:
  Chess position. FEN: <fen>. Side to move: <white|black>. Stage: destination. Selected piece: <B|R|Q>. Selected direction: <direction>.

question id:    destination_selection
instructions:   Choose the best chess piece destination.
criteria:       destination squares of the legal moves that match the choices made so far
                {"e4": "e4", "d4": "d4", ...}
```

The direction clause is appended for B, R and Q **only**. It is exactly
` Selected direction: <direction>.` — one leading space, one trailing period — and
`<direction>` is the direction **key** (`horizontal`, `vertical`, `diagonal_nw_se` or
`diagonal_ne_sw`), not the display label.

A real destination state from the corpus, for reference:

```text
Chess position. FEN: 2kr3r/1pp4p/p5p1/1q1pPbQ1/8/NPP5/P4PPP/R3K2R b KQ - 2 21. Side to move: black. Stage: destination. Selected piece: Q. Selected direction: diagonal_nw_se.
```

Promotions are written with the promotion piece (for example `e8=Q`, `e8=R`, `e8=B`, `e8=N`), as described in the approach document.

---

## 5. Making one move: step by step

Starting point: a `python-chess` board where it is the bot's turn.

### Step 1: Read the position

```python
fen  = board.fen()
side = "white" if board.turn == chess.WHITE else "black"
moves = list(board.legal_moves)
```

### Step 2: Ask "which piece?"

1. For every legal move, find the piece type that moves (`P N B R Q K`).
2. Keep each distinct type once. These are the `criteria`.
3. Build the Stage A state and question, call `agent.predict`.
4. Take `answer["choice"]` as the **selected piece**.

### Step 3: If the piece is B, R or Q, ask "which direction?"

1. Keep only the legal moves of the selected piece.
2. Label each move by direction from its start square to its end square. With
   `df = to_file − from_file` and `dr = to_rank − from_rank`:
   - `dr == 0` (same rank) → `horizontal`
   - `df == 0` (same file) → `vertical`
   - `df == dr` (up-and-right or down-and-left, as in a1 → b2) → `diagonal_ne_sw`
   - `df == -dr` (up-and-left or down-and-right, as in h1 → a8) → `diagonal_nw_se`
3. Build the Stage B question with the directions that occur, call `agent.predict`.
4. Take `answer["choice"]` as the **selected direction**.

Anything that is neither a straight line nor a 45-degree diagonal is not a slider move and
must not reach this step.

For P, N and K, skip this step.

### Step 4: Ask "which destination?"

1. Keep only the legal moves that match the piece (and the direction, for B/R/Q).
2. The destination squares of those moves are the `criteria`.
3. Build the Stage C state and question, call `agent.predict`.
4. Take `answer["choice"]` as the **selected destination**.

### Step 5: Turn the choices into one legal move

Find the legal move whose piece type, direction (if used) and destination all match the three choices.

- This will almost always be exactly one move.
- **Edge case:** two pieces of the same type can reach the same square (for example two knights). The hierarchy has no question that separates them, so more than one move can match. Decide a fixed rule for this (for example, pick the first match in sorted UCI order) and state it when you present the agent.

### Step 6: Play it

```python
board.push(chosen_move)
```

The opponent moves, and the cycle repeats from Step 1.

---

## 6. Worked example

A real position from `model/laya_finetuned_1000_after_finetuning.json`, black to move after
`1.e4 d5 2.c4`. Both stages below are the states that file actually recorded, with the
probabilities the model actually returned.

```text
Chess position. FEN: rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2. Side to move: black. Stage: piece.
```

| Stage | Question | Options offered | Model answer |
|---|---|---|---|
| A. Piece | `piece_selection` | Pawn, Knight, Bishop, Queen, King | **Pawn** (0.6036) |
| C. Destination | `destination_selection` | c4, a5 … c6, e6 … h6 | **e6** (0.2192) |

Only the pawn was chosen, so there is no direction stage. The destination `e6` identifies
exactly one legal move: `d5e6`, the d5 pawn capturing the pawn on e4.

Total model calls for this move: **2**.

A queen move takes **3**. The destination stage for one of them is the `Selected direction:
diagonal_nw_se` state quoted in §4, which was offered `c4`, `d3`, `e2` and `f1`, and the model
answered **f1** (0.2683).

---

## 7. Reference code: one bot move

This is a sketch that follows the steps above. For the real implementation, and for the strings that
must be byte-exact, read `backend/chessbot/prompts.py`.

```python
import chess

PIECE_NAMES = {"P": "Pawn", "N": "Knight", "B": "Bishop",
               "R": "Rook", "Q": "Queen", "K": "King"}
# Directions are labelled, not echoed. The dash in the diagonal labels is an en dash,
# U+2013, and must not be typed as an ASCII hyphen.
DIRECTION_NAMES = {"horizontal": "Horizontal", "vertical": "Vertical",
                   "diagonal_nw_se": "Diagonal NW–SE",
                   "diagonal_ne_sw": "Diagonal NE–SW"}

def ask(agent, state, qid, instructions, options):
    """One multiple-choice call. `options` is a list of option keys."""
    labels = DIRECTION_NAMES if qid == "direction_selection" else PIECE_NAMES
    questions = {qid: {"type": "choice",
                       "instructions": instructions,
                       "criteria": {o: labels.get(o, o) for o in options}}}
    return agent.predict(state, questions)["answers"][qid]["choice"]

def piece_of(board, move):
    return chess.piece_symbol(board.piece_type_at(move.from_square)).upper()

def direction_of(move):
    f1, r1 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    f2, r2 = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    df, dr = f2 - f1, r2 - r1
    if dr == 0: return "horizontal"
    if df == 0: return "vertical"
    if df == dr: return "diagonal_ne_sw"     # up-and-right, or down-and-left
    if df == -dr: return "diagonal_nw_se"    # up-and-left, or down-and-right
    raise ValueError(f"{move.uci()} is neither a straight line nor a 45-degree diagonal")

def dest_label(move):
    name = chess.square_name(move.to_square)
    return name + "=" + chess.piece_symbol(move.promotion).upper() if move.promotion else name

def bot_move(agent, board):
    side = "white" if board.turn == chess.WHITE else "black"
    base = f"Chess position. FEN: {board.fen()}. Side to move: {side}."
    moves = list(board.legal_moves)

    # Stage A: piece
    pieces = [p for p in "PNBRQK" if any(piece_of(board, m) == p for m in moves)]
    piece = ask(agent, f"{base} Stage: piece.", "piece_selection",
                "Choose the best chess piece for the move.", pieces)
    moves = [m for m in moves if piece_of(board, m) == piece]

    # Stage B: direction (sliding pieces only)
    state_dest = f"{base} Stage: destination. Selected piece: {piece}."
    if piece in "BRQ":
        dirs = [d for d in ("horizontal", "vertical", "diagonal_nw_se", "diagonal_ne_sw")
                if any(direction_of(m) == d for m in moves)]
        direction = ask(agent, f"{base} Stage: direction. Selected piece: {piece}.",
                        "direction_selection",
                        "Choose the best direction for the chess piece.", dirs)
        moves = [m for m in moves if direction_of(m) == direction]
        state_dest += f" Selected direction: {direction}."   # exactly this suffix

    # Stage C: destination
    dests = list(dict.fromkeys(dest_label(m) for m in moves))
    dest = ask(agent, state_dest, "destination_selection",
               "Choose the best chess piece destination.", dests)

    # Resolve to one legal move
    matches = [m for m in moves if dest_label(m) == dest]
    return sorted(matches, key=lambda m: m.uci())[0]      # fixed tie-break rule
```

Game loop:

```python
board = chess.Board()
while not board.is_game_over():
    board.push(bot_move(agent, board))   # replace with the human/opponent move on their turn
```

---

## 8. Choosing the answer: greedy or sampled

`answer["choice"]` is the most probable option (greedy). This is the default and makes the bot deterministic.

If you want varied play, sample from `answer["probabilities"]` instead of taking the maximum. This is a design choice to state explicitly when you present the agent.

Note that picking the best option at each stage is not always the same as picking the single most probable complete move, because the move probability is the product of the stage probabilities (`P(piece) × P(direction | piece) × P(destination | piece, direction)`).

---

## 9. Cost per move

- 2 model calls for P, N, K; 3 for B, R, Q.
- Notebook latency, on the 2×T4 Kaggle setup: about **34 ms per call**, roughly **70 to 100 ms per bot move**. That is a GPU number and does not apply to the local CPU app.
- This machine, measured through the running backend: **~11 s** to load the checkpoint once at startup, **0.4–0.8 s per call**, **1.1–1.4 s per bot move**. Each move's per-stage latencies are stored on the game and shown in the UI.

---

## 10. What the evaluation shows

Measured in the notebook on 1,000 held-out test rows, where each row is **one stage question** (not a full move):

The fine-tuned column is read from `model/laya_finetuned_1000_after_finetuning.json` and the base column from `model/laya_baseline_1000_before_finetuning.json`, so the figures are the ones the app treats as authoritative rather than the notebook's printed rounding.

| Metric | Base Laya | Fine-tuned | Better |
|---|---|---|---|
| Accuracy | 26.1% | 49.7% | higher |
| Soft accuracy | 0.291 | 0.373 | higher |
| Brier score | 0.4111 | 0.2667 | lower |
| KL divergence | 0.9245 | 0.5382 | lower |
| TV distance | 0.5208 | 0.3938 | lower |

Accuracy here means the model's top option equals the top option of the gold distribution for that stage. The gold distribution is built from the human move plus Stockfish scores, so the target is "human-like, engine-informed", not "engine-best". These numbers are per stage, so they do not directly give the accuracy of a full move.

---

## 11. One-paragraph summary for presenting the agent

> The bot encodes the board as a FEN text state and asks the fine-tuned Laya model a short chain of multiple-choice questions: first which piece type to move, then (for bishops, rooks and queens) which direction, then which destination square. Every option comes from the legal moves of the current position, so the model can only pick legal moves. The three answers identify one legal move, which is played on the board with python-chess. Each move takes 2 or 3 model calls, a few hundred milliseconds each on a CPU.
