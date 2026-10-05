"""Environment-backed settings.

Every value has a working default, so `uv run uvicorn app.main:app` starts with no
environment set up at all. Override any of them with an environment variable of the same
name, or a `.env` file in `backend/`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the backend.

    Attributes:
        laya_model_dir: Path to the fine-tuned Laya checkpoint.
        laya_device: Torch device. `"cpu"` here, since this machine has no GPU.
        log_level: Root log level. `DEBUG` narrates the whole decision chain, which is the
            fastest way to see why the bot played what it played.
        database_url: SQLAlchemy URL for the game store.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    laya_model_dir: Path = REPO_ROOT / "model" / "laya_finetuned_typed_decisions"
    laya_device: str = "cpu"
    log_level: str = "INFO"
    # Forward slashes: a Windows path with backslashes reads ambiguously to a SQLite URL.
    database_url: str = f"sqlite:///{REPO_ROOT / 'backend' / 'app' / 'chessbot.db'}"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, created once.

    Returns:
        The cached `Settings` instance. `lru_cache` keeps import cost off every request and
        makes the override in a test a single `cache_clear`.
    """
    return Settings()


settings = get_settings()
