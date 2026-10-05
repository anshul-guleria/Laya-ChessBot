"use client";

/** Props for `MoveList`. */
export interface MoveListProps {
  /** The move list in UCI. */
  moves: string[];
}

/**
 * Show the moves as numbered UCI pairs.
 *
 * UCI rather than SAN because the backend stores UCI and adding a chess library to the
 * frontend to re-derive SAN would duplicate rules the server already owns.
 *
 * The row wrapper is `display: contents`, so its three spans become direct grid items of
 * `.movelist` and the number, white move and black move line up in one column each. The
 * wrapper is a plain `div` with no implicit role, which is the case where `display:
 * contents` costs nothing in accessibility.
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