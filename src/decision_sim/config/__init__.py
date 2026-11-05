"""Configuration and environment loading."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SimConfig(BaseModel):
    """Runtime configuration for a simulation run."""

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=20, ge=1, description="Hard cap on total rounds executed.")
    recent_posts_window: int = Field(
        default=20, ge=1, description="How many recent readable posts an agent sees."
    )
    memory_recall_limit: int = Field(
        default=5, ge=1, description="How many of an agent's own past notes to recall."
    )
    seed: int = Field(default=7, description="Base seed for deterministic agent ordering.")
    model: str | None = Field(
        default=None, description="LiteLLM model override; falls back to LLM_MODEL env var."
    )


__all__ = ["SimConfig"]
