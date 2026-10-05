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
