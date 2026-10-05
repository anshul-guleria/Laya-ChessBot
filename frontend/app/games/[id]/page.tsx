"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import Board from "@/components/Board";
import DecisionTrace from "@/components/DecisionTrace";
import MoveList from "@/components/MoveList";
import { ApiRequestError, fetchGame, playMove, undoMove } from "@/lib/api";
import type { GameState } from "@/lib/types";

/**
 * One game: the board, the move list, the bot's reasoning, and the controls.
 *
 * The server's response replaces local state wholesale after every action. Nothing is
 * advanced optimistically, because at 1-8 s per bot move an optimistic board would sit
 * there being wrong -- and because the bot's reply, the decision trace and the promoted
 * position all only exist in that response.
 *
 * That also means `fetchPgn` is not called: PGN export is a plain link at the backend URL,
 * so the browser downloads the file without a round trip through React.
 */
export default function GamePage() {
  const params = useParams<{ id: string }>();
  const gameId = Number(params.id);

  const [game, setGame] = useState<GameState | null>(null);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isInteger(gameId)) {
      setError(`"${String(params.id)}" is not a game id.`);
      return;
    }
    let live = true;
    fetchGame(gameId)
      .then((state) => {
        if (live) {
          setGame(state);
        }
      })
      .catch((cause: unknown) => {
        if (live) {
          setError(describe(cause));
        }
      });
    return () => {
      live = false;
    };
  }, [gameId, params.id]);

  /**
   * Run one server action, holding the `thinking` flag for its whole duration.
   *
   * The flag spans the request rather than just the bot's half of it, so the board is also
   * locked while the server validates the human's move.
   *
   * @param action The request to make.
   */
  const run = useCallback(async (action: () => Promise<GameState>) => {
    setThinking(true);
    setError(null);
    try {
      setGame(await action());
    } catch (cause: unknown) {
      setError(describe(cause));
    } finally {
      setThinking(false);
    }
  }, []);

  /**
   * Send a completed drag to the server.
   *
   * The origin square comes from the board's own drop report, but it is not trusted: the
   * move has to appear in the `legalMoves` the server sent for this exact position, so an
   * origin that is not one of them never becomes a request.
   *
   * @param uci The dropped move in UCI, promotion suffix included.
   */
  function handleMove(uci: string) {
    if (!game || thinking || game.status !== "active" || !game.legalMoves.includes(uci)) {
      return;
    }
    void run(() => playMove(game.id, uci));
  }

  if (!game) {
    return (
      <main className="shell">
        {error ? (
          <>
            <p className="error">{error}</p>
            <p>
              <Link href="/">Back to the game list</Link>
            </p>
          </>
        ) : (
          <p className="state">Loading game {gameId}…</p>
        )}
      </main>
    );
  }

  const live = game.status === "active";

  return (
    <main className="shell">
      <div className="topbar">
        <h1>Game {game.id}</h1>
        <span className="badge" data-state={live ? "ok" : "degraded"}>
          {describeStatus(game)}
        </span>
      </div>

      <div className="layout">
        <section>
          <Board
            fen={game.fen}
            legalMoves={game.legalMoves}
            lastMove={game.lastMove}
            orientation={game.humanSide}
            disabled={thinking || !live}
            onMove={handleMove}
            thinking={thinking}
          />
          <div className="toolbar">
            <button
              type="button"
              onClick={() => void run(() => undoMove(game.id))}
              disabled={thinking || game.moves.length === 0}
            >
              Undo move
            </button>
            <a href={`/api/games/${game.id}/pgn`} download={`game-${game.id}.pgn`}>
              Export PGN
            </a>
            <Link href="/">New game</Link>
          </div>
          {error ? <p className="error">{error}</p> : null}
        </section>

        <div>
          <section className="panel">
            <h2>Moves</h2>
            <MoveList moves={game.moves} />
          </section>
          <section className="panel" style={{ marginTop: 24 }}>
            <h2>How Laya decided</h2>
            <DecisionTrace trace={game.botTrace} />
          </section>
        </div>
      </div>
    </main>
  );
}

/**
 * Turn an unknown throw into a sentence the player can act on.
 *
 * Each status the backend can return here gets its own wording, because the three failure
 * modes mean very different things: a degraded model is worth retrying, an illegal move is
 * worth explaining, and a finished game is worth showing the result of.
 *
 * @param cause Whatever was thrown, almost always an `ApiRequestError`.
 * @returns A message for the user.
 */
function describe(cause: unknown): string {
  if (!(cause instanceof ApiRequestError)) {
    return cause instanceof Error ? cause.message : String(cause);
  }

  switch (cause.status) {
    case 400:
      return `The server rejected that move as illegal. ${cause.message}`;
    case 404:
      return `That game does not exist. ${cause.message}`;
    case 409:
      return `This game is already over, so the move was not played. ${cause.message}`;
    case 503:
      return `The Laya model could not decide a move, so the turn was passed back to you. ${cause.message}`;
    default:
      return `The server returned ${cause.status}. ${cause.message}`;
  }
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
  const winner =
    game.status === "white_won" ? "White" : game.status === "black_won" ? "Black" : "Nobody";
  const how = game.termination ? ` by ${game.termination.replace("_", " ")}` : "";
  return `${winner} wins${how}`;
}