"use client";

import type { BotTrace, StageTrace } from "@/lib/types";

/** Props for `DecisionTrace`. */
export interface DecisionTraceProps {
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
        <Stage key={stage.questionId} stage={stage} />
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
 * Render one question: its instruction, its options with probabilities, and its latency.
 *
 * A stage the model was never asked renders as skipped, with the reason and nothing else.
 * Drawing a probability list for a question that was never put to the model would be a
 * fabrication, and this panel exists precisely because it is trusted as fact.
 *
 * @param stage The stage to render.
 * @returns The stage block.
 */
function Stage({ stage }: StageProps) {
  const latency = <span className="badge">{(stage.latencyMs / 1000).toFixed(1)}s</span>;

  if (stage.skippedReason) {
    return (
      <div className="trace-stage">
        <header>
          <span>{stage.stage} — skipped</span>
          {latency}
        </header>
        <p className="state">{stage.skippedReason}</p>
      </div>
    );
  }

  const entries = Object.entries(stage.probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="trace-stage">
      <header>
        <span>
          {stage.stage}
          {stage.choice ? ` — ${stage.choice}` : ""}
        </span>
        {latency}
      </header>

      <p className="state">{stage.instructions}</p>

      <div className="options">
        {entries.map(([key, probability]) => {
          const percent = Math.round(probability * 100);
          return (
            <div className="option" data-chosen={key === stage.choice} key={key}>
              <span>
                {key} {stage.options[key] ? `· ${stage.options[key]}` : ""}
              </span>
              <span className="bar">
                <span style={{ width: `${Math.max(2, percent)}%` }} />
              </span>
              <span>{percent}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}