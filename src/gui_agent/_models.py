"""Shared Pydantic model policies used by persisted and runtime schemas."""

from pydantic import BaseModel, ConfigDict


class StrictFrozenModel(BaseModel):
    """Reject coercion and unknown fields while keeping validated values immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
