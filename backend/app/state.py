"""Process-wide holder for the loaded model.

The agent is created once at startup and read from anywhere. It is a module-level holder
rather than something attached to `app.state`, so the routers do not need the app object
and tests can swap it without building an app.
"""

from __future__ import annotations

import logging

from chessbot.agent import Agent

logger = logging.getLogger(__name__)

_agent: Agent | None = None


def set_agent(agent: Agent | None) -> None:
    """Install the process-wide agent, or clear it.

    Args:
        agent: The loaded agent, or `None` to mark the service degraded.
    """
    global _agent
    _agent = agent
    logger.info("agent %s", "loaded" if agent is not None else "cleared")


def get_agent() -> Agent | None:
    """Return the process-wide agent.

    Returns:
        The `Agent`, or `None` when no model is loaded.
    """
    return _agent


def clear_agent() -> None:
    """Clear the process-wide agent."""
    set_agent(None)


__all__ = ["clear_agent", "get_agent", "set_agent"]
