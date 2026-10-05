"use client";

import { useCallback, useMemo } from "react";
import type { ComponentProps } from "react";
import { Chessboard } from "react-chessboard";

/**
 * `react-chessboard` v4 does not re-export its `Square` and `Piece` unions from the package
 * root -- `dist/index.d.ts` only forwards `Chessboard`, `ChessboardDnDProvider`,
 * `SparePiece` and `ClearPremoves`. Both unions are therefore derived from the component's
 * own prop types. Deriving rather than re-declaring keeps this file honest: a library
 * signature change breaks the build instead of silently mistyping a square name.
 */
type ChessboardProps = ComponentProps<typeof Chessboard>;
type PieceDropHandler = NonNullable<ChessboardProps["onPieceDrop"]>;
type CustomSquareStyles = NonNullable<ChessboardProps["customSquareStyles"]>;
type Square = Parameters<PieceDropHandler>[0];
type Piece = Parameters<PieceDropHandler>[2];

/** A two-character UCI fragment is a square name only if it looks like one. */
const SQUARE_PATTERN = /^[a-h][1-8]$/;

/** Props for `Board`. */
export interface BoardProps {
  /** Current position in FEN. */
  fen: string;
  /** Legal moves in UCI, used to highlight the squares a piece may reach. */
  legalMoves: string[];
  /** The most recent move in UCI, highlighted on the board. */
  lastMove: string | null;
  /** Which colour is at the bottom. */
  orientation: "white" | "black";
  /** Reject input, e.g. while the bot is thinking. */
  disabled: boolean;
  /**
   * Called with a legal move in UCI once a drag completes.
   *
   * The whole move is passed rather than just the destination square, because the origin is
   * only recoverable from the drop itself -- guessing it from the destination cannot tell
   * `e2e4` from `d2e4`, and for a promotion it cannot tell `e7e8=Q` from `e7e8=R`.
   */
  onMove: (uci: string) => void;
  /** Whether the bot is currently deciding. */
  thinking: boolean;
}

/**
 * The board, plus the small amount of state that dragging a piece needs.
 *
 * The component never advances the position itself. `react-chessboard` is given an
 * `onPieceDrop`, and its own source comment is explicit that when one is supplied
 * "position must be updated externally and captured by useEffect above for this move to
 * show on board" -- the library leaves `currentPosition` untouched and waits for the next
 * `position` prop, which arrives only once the server has answered. Returning `true` from
 * the handler therefore means "an authoritative position is coming", not "this move is
 * confirmed", and it is what stops the drag snapping back mid-thought. At 1-8 s per bot
 * move the board holds the last server-confirmed position and shows a spinner, rather than
 * showing a piece the server has not accepted.
 */
export default function Board({
  fen,
  legalMoves,
  lastMove,
  orientation,
  disabled,
  onMove,
  thinking,
}: BoardProps) {
  const squareStyles = useMemo(
    () => buildSquareStyles(legalMoves, lastMove),
    [legalMoves, lastMove],
  );

  /**
   * Resolve a completed drag into UCI, and refuse anything the server did not offer.
   *
   * @param sourceSquare Where the piece was picked up.
   * @param targetSquare Where it was dropped.
   * @param piece The piece as the library spells it, e.g. `"wQ"`.
   * @returns Whether the parent will supply a new authoritative position.
   */
  const handleDrop = useCallback(
    (sourceSquare: Square, targetSquare: Square, piece: Piece): boolean => {
      // A promotion arrives here already swapped for the chosen piece, because the library
      // routes it through the promotion dialog before it gets here. A pawn therefore means
      // the move was not a promotion, and needs no suffix.
      const uci = `${sourceSquare}${targetSquare}${piece[1] === "P" ? "" : piece[1].toLowerCase()}`;

      if (!legalMoves.includes(uci)) {
        return false;
      }

      onMove(uci);
      return true;
    },
    [legalMoves, onMove],
  );

  return (
    <div style={{ opacity: disabled ? 0.7 : 1, transition: "opacity 120ms" }}>
      <Chessboard
        // Re-keying on the position rebuilds the library's internal state from the FEN the
        // server sent. It is what makes the board authoritative: nothing survives locally
        // between moves. It also re-arms the promotion dialog, which is otherwise a one-shot
        // -- the library only opens it again when `promotionToSquare` changes.
        key={fen}
        position={fen}
        boardOrientation={orientation}
        arePiecesDraggable={!disabled}
        showPromotionDialog
        customSquareStyles={squareStyles}
        customDarkSquareStyle={{ backgroundColor: "#4a5568" }}
        customLightSquareStyle={{ backgroundColor: "#cbd5e0" }}
        customBoardStyle={{ borderRadius: "8px", overflow: "hidden" }}
        onPieceDrop={handleDrop}
      />
      {thinking ? (
        <p className="state">
          <span className="spinner" /> Laya is choosing a move, 1-8 seconds on CPU…
        </p>
      ) : null}
    </div>
  );
}

/**
 * Shade every square the side to move can reach, then shade the last move's two squares
 * more strongly so the most recent ply reads through the legal-move hints.
 *
 * @param legalMoves Legal moves in UCI, as the server computed them.
 * @param lastMove The most recent move in UCI, or `null` before the first ply.
 * @returns A style map keyed by square name, for `customSquareStyles`.
 */
function buildSquareStyles(legalMoves: string[], lastMove: string | null): CustomSquareStyles {
  const styles: CustomSquareStyles = {};

  for (const move of legalMoves) {
    const target = toSquare(move.slice(2, 4));
    if (target) {
      styles[target] = { backgroundColor: "rgba(122, 162, 247, 0.28)" };
    }
  }

  if (lastMove) {
    for (const name of [lastMove.slice(0, 2), lastMove.slice(2, 4)]) {
      const square = toSquare(name);
      if (square) {
        styles[square] = { backgroundColor: "rgba(122, 162, 247, 0.55)" };
      }
    }
  }

  return styles;
}

/**
 * Narrow a UCI fragment to a square name.
 *
 * @param name A two-character fragment taken from a UCI move.
 * @returns The square, or `null` when the fragment is not one of the 64.
 */
function toSquare(name: string): Square | null {
  return SQUARE_PATTERN.test(name) ? (name as Square) : null;
}