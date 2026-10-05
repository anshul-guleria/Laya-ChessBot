# How the Laya Chess Bot Makes One Move

This document explains, step by step, how the fine-tuned Laya model is used as a chess bot: what goes **in**, what comes **out**, and how those outputs become one legal move on the board.

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

### Stage A: Piece (confirmed in the notebook)

```text
state:
  Chess position. FEN: <fen>. Side to move: <white|black>. Stage: piece.

question id:    piece_selection
instructions:   Choose the best chess piece for the move.
criteria:       only piece types that have at least one legal move
                {"P": "Pawn", "N": "Knight", "B": "Bishop", "R": "Rook", "Q": "Queen", "K": "King"}
```

### Stage B: Direction (only for B, R, Q)

The notebook's printed samples only show the piece and destination stages, so the direction wording below follows the same pattern and the design in `chess_laya_dataset_approach.md`. **Check the state and instruction strings against your dataset generator before running.**

```text
state:
  Chess position. FEN: <fen>. Side to move: <white|black>. Stage: direction. Selected piece: <B|R|Q>.

criteria:       only directions that have at least one legal move for that piece
                horizontal / vertical / diagonal
```

### Stage C: Destination (confirmed in the notebook for the pawn case)

```text
state (P, N, K):
  Chess position. FEN: <fen>. Side to move: <white|black>. Stage: destination. Selected piece: <P|N|K>.

state (B, R, Q), after the direction was chosen:
  same as above, plus the selected direction (copy the exact wording from your dataset generator)

question id:    destination_selection
instructions:   Choose the best chess piece destination.
criteria:       destination squares of the legal moves that match the choices made so far
                {"e4": "e4", "d4": "d4", ...}
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
2. Label each move by direction from its start square to its end square:
   - same rank → `horizontal`
   - same file → `vertical`
   - otherwise → `diagonal`
3. Build the Stage B question with the directions that occur, call `agent.predict`.
4. Take `answer["choice"]` as the **selected direction**.

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

Position: starting position, white to move.

| Stage | State ends with | Options offered | Model answer |
|---|---|---|---|
| A. Piece | `Stage: piece.` | P, N | P (e.g. 0.84) |
| C. Destination | `Stage: destination. Selected piece: P.` | a3 … h3, a4 … h4 | e4 (e.g. 0.47) |

Only the pawn was chosen, so no direction stage. Matching legal move: pawn to e4, which is `e2e4`.

Total model calls for this move: **2**.

For a queen move the chain would be piece (Q) → direction (e.g. diagonal) → destination (e.g. h5), which is **3** calls.

---

## 7. Reference code: one bot move

This is a sketch that follows the steps above. The direction and destination wording must be matched to your dataset generator.

```python
import chess

PIECE_NAMES = {"P": "Pawn", "N": "Knight", "B": "Bishop",
               "R": "Rook", "Q": "Queen", "K": "King"}

def ask(agent, state, qid, instructions, options):
    """One multiple-choice call. `options` is a list of option keys."""
    questions = {qid: {"type": "choice",
                       "instructions": instructions,
                       "criteria": {o: PIECE_NAMES.get(o, o) for o in options}}}
    return agent.predict(state, questions)["answers"][qid]["choice"]

def piece_of(board, move):
    return chess.piece_symbol(board.piece_type_at(move.from_square)).upper()

def direction_of(move):
    f1, r1 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    f2, r2 = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    if r1 == r2: return "horizontal"
    if f1 == f2: return "vertical"
    return "diagonal"

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
        dirs = sorted({direction_of(m) for m in moves})
        direction = ask(agent, f"{base} Stage: direction. Selected piece: {piece}.",
                        "direction_selection", "<copy from dataset generator>", dirs)
        moves = [m for m in moves if direction_of(m) == direction]
        state_dest += f" Selected direction: {direction}."   # match dataset wording

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
- Measured latency in the notebook: about **34 ms per call** on the 2×T4 Kaggle setup, so roughly **70 to 100 ms per bot move**.

---

## 10. What the evaluation shows

Measured in the notebook on 1,000 held-out test rows, where each row is **one stage question** (not a full move):

| Metric | Base Laya | Fine-tuned | Better |
|---|---|---|---|
| Accuracy | 26.1% | 49.7% | higher |
| Soft accuracy | 0.291 | 0.372 | higher |
| Brier score | 0.411 | 0.268 | lower |
| KL divergence | 0.925 | 0.544 | lower |
| TV distance | 0.521 | 0.396 | lower |

Accuracy here means the model's top option equals the top option of the gold distribution for that stage. The gold distribution is built from the human move plus Stockfish scores, so the target is "human-like, engine-informed", not "engine-best". These numbers are per stage, so they do not directly give the accuracy of a full move.

---

## 11. One-paragraph summary for presenting the agent

> The bot encodes the board as a FEN text state and asks the fine-tuned Laya model a short chain of multiple-choice questions: first which piece type to move, then (for bishops, rooks and queens) which direction, then which destination square. Every option comes from the legal moves of the current position, so the model can only pick legal moves. The three answers identify one legal move, which is played on the board with python-chess. Each move takes 2 or 3 model calls, about 35 ms each.
