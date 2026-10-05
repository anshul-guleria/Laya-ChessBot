"""Application factory, lifespan and logging setup."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import state
from app.config import settings
from app.db import init_db
from app.routers import games, health
from chessbot.agent import LocalLayaAgent

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up the two logger trees.

    `chessbot` and `app` are configured together at the same level because both are
    needed to understand a move: the library narrates the decision chain and the web
    layer narrates what happened to the game.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    logging.getLogger("chessbot").setLevel(level)
    logging.getLogger("app").setLevel(level)


def load_model() -> None:
    """Load the checkpoint once.

    An agent already installed in `app.state` wins: loading a second copy of an 842 MB
    model over the top of one that is already answering questions would cost a minute of
    startup for nothing, and the test suite installs its own agent before the app starts.

    A failure is logged with its traceback and swallowed: the app still starts, reports
    `degraded` on `/api/health`, and returns `503` on the move endpoint. A crash on
    startup would take the whole UI down for a problem that only affects the bot.

    Raises:
        Nothing. `LocalLayaAgent` raises `AgentLoadError`, and anything else it might raise
            on the way out is treated the same way.
    """
    if state.get_agent() is not None:
        logger.info("a model is already installed; not loading the checkpoint again")
        return
    try:
        agent = LocalLayaAgent(settings.laya_model_dir, device=settings.laya_device)
    except Exception:
        logger.exception(
            "could not load the model from %s; the app will serve in a degraded state",
            settings.laya_model_dir,
        )
        state.clear_agent()
        return
    state.set_agent(agent)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the database and load the model.

    Args:
        app: The application being started.

    Yields:
        Control back to the server while it serves requests.
    """
    configure_logging()
    logger.info("starting; model dir %s on %s", settings.laya_model_dir, settings.laya_device)
    init_db()
    load_model()
    yield
    state.clear_agent()
    logger.info("stopped")


def create_app() -> FastAPI:
    """Build the FastAPI application.

    Returns:
        The configured `FastAPI` instance.
    """
    app = FastAPI(title="Laya Chess Bot", version="1.0.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(games.router)
    return app


app = create_app()

__all__ = ["app", "create_app"]
