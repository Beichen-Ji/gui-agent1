from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gui_agent.agent.types import AgentAction
from gui_agent.datasets.schema import DatasetSource

Sha256 = str


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TrainingExample(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    sample_id: str = Field(min_length=1, max_length=500)
    source: DatasetSource
    episode_id: str = Field(min_length=1, max_length=500)
    instruction: str = Field(min_length=1, max_length=4000)
    image_path: str = Field(min_length=1, max_length=2000)
    text_observation: str | None = Field(default=None, min_length=1, max_length=8000)
    target_action: AgentAction
    source_revision: str = Field(min_length=1, max_length=500)


class SourceSplitCounts(_StrictFrozenModel):
    seen: int = Field(ge=0)
    accepted: int = Field(ge=0)
    skipped: int = Field(ge=0)
    train: int = Field(ge=0)
    validation: int = Field(ge=0)


class TrainingSplit(_StrictFrozenModel):
    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]
    seed: int = Field(ge=0)
    validation_ratio: float = Field(gt=0.0, lt=1.0)
    records_seen: int = Field(ge=0)
    records_skipped: int = Field(ge=0)
    source_counts: dict[DatasetSource, SourceSplitCounts]
    skip_reasons: dict[str, int]
    input_sha256: dict[DatasetSource, Sha256]
    source_revisions: dict[DatasetSource, tuple[str, ...]]
    source_licenses: dict[DatasetSource, str]


class TrainingManifest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["gui-agent-week5-training"] = "gui-agent-week5-training"
    seed: int = Field(ge=0)
    validation_ratio: float = Field(gt=0.0, lt=1.0)
    records_seen: int = Field(ge=0)
    records_accepted: int = Field(ge=0)
    records_skipped: int = Field(ge=0)
    train_examples: int = Field(ge=0)
    validation_examples: int = Field(ge=0)
    source_counts: dict[DatasetSource, SourceSplitCounts]
    skip_reasons: dict[str, int]
    input_sha256: dict[DatasetSource, Sha256]
    source_revisions: dict[DatasetSource, tuple[str, ...]]
    source_licenses: dict[DatasetSource, str]
    output_sha256: dict[str, Sha256]

    @field_validator("input_sha256", "output_sha256")
    @classmethod
    def _valid_sha256_values(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for digest in value.values()
        ):
            raise ValueError("input_sha256 and output_sha256 values must be lowercase SHA-256")
        return value


__all__ = [
    "SourceSplitCounts",
    "TrainingExample",
    "TrainingManifest",
    "TrainingSplit",
]
