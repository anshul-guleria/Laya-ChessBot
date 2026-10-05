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
