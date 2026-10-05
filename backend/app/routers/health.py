"""The health endpoint. Answers whether the bot can move at all."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app import state
from app.config import settings
from app.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report model availability.

    Returns:
        `"ok"` when the checkpoint is in memory, `"degraded"` when it is not. Degraded is
        a normal, expected response on a machine where the model could not be loaded; it
        is not an error.
    """
    loaded = state.get_agent() is not None
    logger.debug("health check: model_loaded=%s", loaded)
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        device=settings.laya_device,
        model_dir=str(settings.laya_model_dir),
    )


__all__ = ["health", "router"]
