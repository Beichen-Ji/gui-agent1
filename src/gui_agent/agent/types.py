from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from gui_agent.types import OCRDetection, ScreenshotResult


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ClickAction(_StrictFrozenModel):
    kind: Literal["click"] = "click"
    x: int
    y: int
    button: Literal["left", "middle", "right"] = "left"
    clicks: int = Field(default=1, ge=1, le=2)


class TypeTextAction(_StrictFrozenModel):
    kind: Literal["type_text"] = "type_text"
    text: str = Field(min_length=1, max_length=500)


class HotkeyAction(_StrictFrozenModel):
    kind: Literal["hotkey"] = "hotkey"
    keys: tuple[str, ...] = Field(min_length=1, max_length=4)


class ScrollAction(_StrictFrozenModel):
    kind: Literal["scroll"] = "scroll"
    clicks: int = Field(ge=-20, le=20)
    x: int | None = None
    y: int | None = None


class DragAction(_StrictFrozenModel):
    kind: Literal["drag"] = "drag"
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    duration: float = Field(default=0.5, ge=0.0, le=5.0)


class WaitAction(_StrictFrozenModel):
    kind: Literal["wait"] = "wait"
    seconds: float = Field(ge=0.0, le=5.0)


class FinishAction(_StrictFrozenModel):
    kind: Literal["finish"] = "finish"
    success: bool
    summary: str = Field(min_length=1, max_length=500)


AgentAction: TypeAlias = Annotated[
    ClickAction
    | TypeTextAction
    | HotkeyAction
    | ScrollAction
    | DragAction
    | WaitAction
    | FinishAction,
    Field(discriminator="kind"),
]


class TaskStep(_StrictFrozenModel):
    id: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=500)


class TaskPlan(_StrictFrozenModel):
    goal: str = Field(min_length=1, max_length=1000)
    steps: tuple[TaskStep, ...] = Field(min_length=1, max_length=20)


class AgentDecision(_StrictFrozenModel):
    current_step_id: str = Field(min_length=1, max_length=64)
    rationale_summary: str = Field(min_length=1, max_length=500)
    action: AgentAction
    expected_outcome: str = Field(min_length=1, max_length=500)


class StepResult(_StrictFrozenModel):
    step_index: int = Field(ge=0)
    action: AgentAction
    status: Literal["dry_run", "executed", "denied", "failed"]
    message: str = Field(min_length=1, max_length=500)


@dataclass(frozen=True, slots=True)
class Observation:
    screenshot: ScreenshotResult
    detections: tuple[OCRDetection, ...]
    step_index: int

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")


@dataclass(frozen=True, slots=True)
class AgentState:
    goal: str
    plan: TaskPlan
    observation: Observation
    decisions: tuple[AgentDecision, ...]
    results: tuple[StepResult, ...]

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal must not be blank")


__all__ = [
    "AgentAction",
    "AgentDecision",
    "AgentState",
    "ClickAction",
    "DragAction",
    "FinishAction",
    "HotkeyAction",
    "Observation",
    "ScrollAction",
    "StepResult",
    "TaskPlan",
    "TaskStep",
    "TypeTextAction",
    "WaitAction",
]
