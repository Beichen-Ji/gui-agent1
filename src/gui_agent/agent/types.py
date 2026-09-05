from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

FailureReason: TypeAlias = Literal[
    "execution_error",
    "observation_error",
    "no_visual_change",
    "expected_text_missing",
    "planner_output_invalid",
    "policy_denied",
    "confirmation_rejected",
    "repeated_action",
    "retry_exhausted",
]


class TaskStep(_StrictFrozenModel):
    id: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=500)


class TaskPlan(_StrictFrozenModel):
    goal: str = Field(min_length=1, max_length=1000)
    steps: tuple[TaskStep, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "TaskPlan":
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("task plan step IDs must be unique")
        return self


class StepProgress(_StrictFrozenModel):
    step_id: str = Field(min_length=1, max_length=64)
    status: Literal["pending", "active", "completed", "failed"]
    attempts: int = Field(default=0, ge=0)


class PlanProgress(_StrictFrozenModel):
    steps: tuple[StepProgress, ...] = Field(min_length=1, max_length=20)
    active_step_id: str = Field(min_length=1, max_length=64)
    replan_count: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def _valid_active_step(self) -> "PlanProgress":
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("progress step IDs must be unique")
        if self.active_step_id not in ids:
            raise ValueError("active step must be part of progress")
        active = [step.step_id for step in self.steps if step.status == "active"]
        if len(active) > 1:
            raise ValueError("progress may contain at most one active step")
        if active and active != [self.active_step_id]:
            raise ValueError("active_step_id must identify the active progress step")
        if not active:
            terminal = next(
                step for step in self.steps if step.step_id == self.active_step_id
            )
            if terminal.status not in {"completed", "failed"}:
                raise ValueError("progress without an active step must be terminal")
        return self

    @classmethod
    def from_plan(cls, plan: TaskPlan) -> "PlanProgress":
        return cls(
            steps=tuple(
                StepProgress(
                    step_id=step.id,
                    status="active" if index == 0 else "pending",
                )
                for index, step in enumerate(plan.steps)
            ),
            active_step_id=plan.steps[0].id,
        )

    @property
    def is_complete(self) -> bool:
        return all(step.status == "completed" for step in self.steps)

    @property
    def completed_step_ids(self) -> tuple[str, ...]:
        return tuple(
            step.step_id for step in self.steps if step.status == "completed"
        )

    def record_attempt(self, step_id: str) -> "PlanProgress":
        index = self._step_index(step_id)
        step = self.steps[index]
        if step.status != "active" or step_id != self.active_step_id:
            raise ValueError("attempts may only be recorded for the active step")
        return self._replace(
            index,
            step.model_copy(update={"attempts": step.attempts + 1}),
        )

    def select_step(self, step_id: str) -> "PlanProgress":
        requested_index = self._step_index(step_id)
        requested = self.steps[requested_index]
        if requested.status == "completed":
            raise ValueError("a completed step cannot become active again")
        if requested.status == "failed":
            raise ValueError("a failed step requires a revised plan")
        if requested.status == "active":
            if step_id != self.active_step_id:
                raise ValueError("progress contains an inconsistent active step")
            return self

        active_index = self._step_index(self.active_step_id)
        active = self.steps[active_index]
        if active.status != "active" or requested_index != active_index + 1:
            raise ValueError("a decision may select only the active or next pending step")
        updated = list(self.steps)
        updated[active_index] = active.model_copy(update={"status": "completed"})
        updated[requested_index] = requested.model_copy(update={"status": "active"})
        return self.model_copy(
            update={"steps": tuple(updated), "active_step_id": step_id}
        )

    def complete_active(self) -> "PlanProgress":
        active_index = self._step_index(self.active_step_id)
        active = self.steps[active_index]
        if active.status != "active":
            raise ValueError("progress is already complete or failed")
        updated = list(self.steps)
        updated[active_index] = active.model_copy(update={"status": "completed"})
        next_index = next(
            (
                index
                for index in range(active_index + 1, len(updated))
                if updated[index].status == "pending"
            ),
            None,
        )
        active_step_id = self.active_step_id
        if next_index is not None:
            updated[next_index] = updated[next_index].model_copy(
                update={"status": "active"}
            )
            active_step_id = updated[next_index].step_id
        return self.model_copy(
            update={"steps": tuple(updated), "active_step_id": active_step_id}
        )

    def fail_active(self) -> "PlanProgress":
        index = self._step_index(self.active_step_id)
        active = self.steps[index]
        if active.status != "active":
            return self
        return self._replace(index, active.model_copy(update={"status": "failed"}))

    def _step_index(self, step_id: str) -> int:
        for index, step in enumerate(self.steps):
            if step.step_id == step_id:
                return index
        raise ValueError(f"step {step_id!r} is not part of the plan")

    def _replace(self, index: int, step: StepProgress) -> "PlanProgress":
        updated = list(self.steps)
        updated[index] = step
        return self.model_copy(update={"steps": tuple(updated)})


class ReplanContext(_StrictFrozenModel):
    reason_code: str = Field(min_length=1, max_length=64)
    summary: str = Field(min_length=1, max_length=500)


def reconcile_revised_plan(
    current_plan: TaskPlan,
    progress: PlanProgress,
    proposed_plan: TaskPlan,
) -> tuple[TaskPlan, PlanProgress]:
    """Preserve completed facts while accepting one bounded revised plan."""
    if progress.replan_count >= 1:
        raise ValueError("replan limit has already been reached")
    current_ids = tuple(step.id for step in current_plan.steps)
    progress_ids = tuple(step.step_id for step in progress.steps)
    if current_ids != progress_ids:
        raise ValueError("progress steps do not match the current plan")

    completed_ids = set(progress.completed_step_ids)
    completed_steps = tuple(
        step for step in current_plan.steps if step.id in completed_ids
    )
    remaining_steps = tuple(
        step for step in proposed_plan.steps if step.id not in completed_ids
    )
    revised = TaskPlan(
        goal=current_plan.goal,
        steps=completed_steps + remaining_steps,
    )
    previous = {step.step_id: step for step in progress.steps}
    first_remaining = len(completed_steps)
    revised_steps: list[StepProgress] = []
    for index, step in enumerate(revised.steps):
        prior = previous.get(step.id)
        attempts = prior.attempts if prior is not None else 0
        if index < first_remaining:
            status: Literal["pending", "active", "completed", "failed"] = "completed"
        elif index == first_remaining:
            status = "active"
        else:
            status = "pending"
        revised_steps.append(
            StepProgress(step_id=step.id, status=status, attempts=attempts)
        )
    active_step_id = (
        revised.steps[first_remaining].id
        if first_remaining < len(revised.steps)
        else revised.steps[-1].id
    )
    revised_progress = PlanProgress(
        steps=tuple(revised_steps),
        active_step_id=active_step_id,
        replan_count=progress.replan_count + 1,
    )
    return revised, revised_progress


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


class VerificationResult(_StrictFrozenModel):
    passed: bool
    summary: str = Field(min_length=1, max_length=500)
    reason_code: FailureReason | None = None
    retryable: bool = False
    evidence: tuple[str, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def _consistent_result(self) -> "VerificationResult":
        if self.passed and self.reason_code is not None:
            raise ValueError("a passed verification cannot have a failure reason")
        if self.passed and self.retryable:
            raise ValueError("a passed verification cannot be retryable")
        if not self.passed and self.reason_code is None:
            raise ValueError("a failed verification requires a reason code")
        return self


class RetryDecision(_StrictFrozenModel):
    retry: bool
    delay_seconds: float = Field(default=0.0, ge=0.0, le=5.0)
    reason_code: FailureReason

    @model_validator(mode="after")
    def _no_delay_without_retry(self) -> "RetryDecision":
        if not self.retry and self.delay_seconds != 0.0:
            raise ValueError("a stopped retry decision cannot have a delay")
        return self


@dataclass(frozen=True, slots=True)
class Observation:
    screenshot: ScreenshotResult
    detections: tuple[OCRDetection, ...]
    step_index: int

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")


@dataclass(frozen=True, slots=True, init=False)
class AgentState:
    goal: str
    plan: TaskPlan
    progress: PlanProgress
    observation: Observation
    decisions: tuple[AgentDecision, ...]
    results: tuple[StepResult, ...]
    replan_context: ReplanContext | None

    def __init__(
        self,
        goal: str,
        plan: TaskPlan,
        observation: Observation,
        decisions: tuple[AgentDecision, ...],
        results: tuple[StepResult, ...],
        progress: PlanProgress | None = None,
        replan_context: ReplanContext | None = None,
    ) -> None:
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "progress", progress or PlanProgress.from_plan(plan))
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "replan_context", replan_context)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal must not be blank")
        plan_ids = tuple(step.id for step in self.plan.steps)
        progress_ids = tuple(step.step_id for step in self.progress.steps)
        if plan_ids != progress_ids:
            raise ValueError("progress steps must match plan steps in order")


__all__ = [
    "AgentAction",
    "AgentDecision",
    "AgentState",
    "ClickAction",
    "DragAction",
    "FailureReason",
    "FinishAction",
    "HotkeyAction",
    "Observation",
    "PlanProgress",
    "ReplanContext",
    "RetryDecision",
    "ScrollAction",
    "StepResult",
    "StepProgress",
    "TaskPlan",
    "TaskStep",
    "TypeTextAction",
    "VerificationResult",
    "WaitAction",
    "reconcile_revised_plan",
]
