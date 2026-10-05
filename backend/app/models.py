"""The one ORM model this app has."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Returns:
        `datetime.now(UTC)`. SQLite drops the offset on the way to disk, so the value is
        made aware here rather than parsed back with a naive assumption.
    """
    return datetime.now(UTC)


class Game(Base):
    """One game between a human and the bot.

    `moves` is the single source of truth for the position. There is deliberately no
    `fen` column: a stored FEN can drift from the move history, and a player seeing one
    position while the bot reasons about another is a genuinely nasty class of bug. The
    current FEN is derived on read, and so is the status.

    `status`, `result` and `termination` are therefore a cache of a replay, kept so the
    game list can be answered without replaying every game in the table. Nothing is ever
    read back out of them into a response: `schemas.build_game_state` recomputes all three.

    Attributes:
        id: Primary key.
        human_side: Which colour the human plays, `"white"` or `"black"`. Chosen at random
            when the game is created.
        status: `"active"`, `"white_won"`, `"black_won"` or `"draw"`.
        result: PGN result (`"1-0"`, `"0-1"`, `"1/2-1/2"`), or `None` while live.
        termination: Why the game ended, e.g. `"checkmate"`.
        moves: UCI move strings in order.
        last_trace: The most recent bot decision as a plain dict, for the UI.
        created_at: When the game was created.
        updated_at: When the game last changed.
    """

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    human_side: Mapped[str] = mapped_column(String(5), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    result: Mapped[str | None] = mapped_column(String(8), nullable=True)
    termination: Mapped[str | None] = mapped_column(String(32), nullable=True)
    moves: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    last_trace: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    def __repr__(self) -> str:
        """Return a debugging representation.

        Returns:
            A short string naming the id, the human's colour, the cached status and the
            ply count. The board itself is not rendered; `build_game_state` is how you see
            the position.
        """
        return (
            f"<Game id={self.id} human_side={self.human_side!r} "
            f"status={self.status!r} plies={len(self.moves or [])}>"
        )
