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
  const [problems, setProblems] = useState<string[]>([]);

  useEffect(() => {
    // `window.location.href` in `newGame` tears this component down mid-flight, so every
    // late resolution has to be able to tell that nobody is left to render it.
    let live = true;

    fetchHealth()
      .then((state) => {
        if (live) {
          setHealth(state);
        }
      })
      .catch((cause: unknown) => {
        if (live) {
          setProblems((current) => [...current, `Could not read model status: ${describe(cause)}`]);
        }
      });

    listGames()
      .then((summaries) => {
        if (live) {
          setGames(summaries);
        }
      })
      .catch((cause: unknown) => {
        if (live) {
          setProblems((current) => [...current, `Could not load recent games: ${describe(cause)}`]);
        }
      });

    return () => {
      live = false;
    };
  }, []);

  async function newGame() {
    setBusy(true);
    setError(null);
    try {
      const game = await createGame();
      window.location.href = `/games/${game.id}`;
    } catch (cause: unknown) {
      setError(describe(cause));
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <div className="topbar">
        <h1>Laya Chess Bot</h1>
        <span className="badge" data-state={health ? health.status : "degraded"}>
          {health ? (health.modelLoaded ? "model loaded" : "model unavailable") : "model status unknown"}
        </span>
      </div>

      {problems.map((problem) => (
        <p className="error" key={problem}>
          {problem}
        </p>
      ))}

      <div className="layout">
        <section className="panel">
          <h2>Play</h2>
          <p className="state">
            You will be given a colour at random. The bot decides its move by answering
            three questions about the position, and shows you its reasoning afterwards.
          </p>
          <div className="toolbar">
            <button onClick={newGame} disabled={busy || health?.modelLoaded === false}>
              {busy ? "Creating…" : "New game"}
              {busy ? <span className="spinner" /> : null}
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

/**
 * Turn an unknown throw into a sentence.
 *
 * @param cause Whatever was thrown.
 * @returns A message for the user.
 */
function describe(cause: unknown): string {
  return cause instanceof ApiRequestError ? cause.message : String(cause);
}