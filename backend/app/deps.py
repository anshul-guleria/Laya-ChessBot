"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, status

from app import state
from chessbot.agent import Agent


def get_agent() -> Agent:
    """Return the loaded agent, or fail with `503`.

    Declared as a dependency rather than called directly inside the handlers, so a test
    can override it with `app.dependency_overrides` and never load the checkpoint.

    Returns:
        The process-wide `Agent`.

    Raises:
        HTTPException: `503` when no model is loaded. The message names the reason, so a
            player sees why the bot is not moving.
    """
    agent = state.get_agent()
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the model failed to load, so no move can be decided",
        )
    return agent


__all__ = ["get_agent"]
