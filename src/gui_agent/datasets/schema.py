from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gui_agent.agent.types import AgentAction

DatasetSource = Literal["screenagent", "mind2web", "webarena"]
RecordType = Literal["trajectory_step", "task"]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NormalizedGUIRecord(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    source: DatasetSource
    record_type: RecordType
    split: str = Field(min_length=1, max_length=64)
    episode_id: str = Field(min_length=1, max_length=500)
    step_index: int = Field(ge=0)
    instruction: str = Field(min_length=1, max_length=4000)
    image_path: str | None = Field(default=None, min_length=1, max_length=1000)
    text_observation: str | None = Field(default=None, min_length=1, max_length=8000)
    action: AgentAction | None = None
    success_criteria: str | None = Field(default=None, min_length=1, max_length=4000)
    source_revision: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_record_shape(self) -> Self:
        if self.record_type == "trajectory_step":
            if self.action is None:
                raise ValueError("trajectory_step requires an action")
            if self.image_path is None and self.text_observation is None:
                raise ValueError("trajectory_step requires an image or text observation")
        if self.record_type == "task" and self.success_criteria is None:
            raise ValueError("task requires success_criteria")
        return self


class DatasetManifest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    source: DatasetSource
    source_url: str = Field(min_length=1, max_length=1000)
    source_revision: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=1000)
    records_written: int = Field(ge=0)
    records_skipped: int = Field(ge=0)
    output_file: str = Field(min_length=1, max_length=1000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


__all__ = ["DatasetManifest", "DatasetSource", "NormalizedGUIRecord", "RecordType"]
