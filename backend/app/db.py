"""SQLAlchemy engine, session factory and declarative base."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def make_engine(url: str) -> Engine:
    """Build a SQLAlchemy engine.

    Named `make_engine` rather than `create_engine` so that the wrapper is never mistaken
    for SQLAlchemy's own factory at a call site.

    Args:
        url: A SQLAlchemy database URL.

    Returns:
        A configured `Engine`. SQLite gets `check_same_thread=False`, because FastAPI
        serves sync handlers from the threadpool and a connection opened on one worker
        thread would otherwise be rejected on the next.
    """
    return create_engine(
        url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}
    )


engine = make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create any tables that do not exist yet.

    There is one table and no schema history worth preserving, so Alembic would be
    ceremony. Revisit when the schema starts moving.
    """
    from app import models  # noqa: F401  importing registers the mappers

    Base.metadata.create_all(bind=engine)
    logger.info("database ready at %s", settings.database_url)


def get_db() -> Iterator[Session]:
    """Yield a request-scoped session and close it afterwards.

    Yields:
        An open `Session`.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
