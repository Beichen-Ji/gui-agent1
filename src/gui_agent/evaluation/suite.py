"""Validate the fixed task suite and execute deterministic references."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from gui_agent._models import StrictFrozenModel
from gui_agent.agent.planner import FakePlanner
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    ClickAction,
    DragAction,
    FinishAction,
    ScrollAction,
    TaskPlan,
    TaskStep,
)
from gui_agent.agent.verification import RuleBasedOutcomeVerifier
from gui_agent.simulation.apps import AppId
from gui_agent.simulation.harness import (
    SimulatedActionExecutor,
    SimulatedDesktop,
    SimulatedObservationSource,
    SimulationPolicy,
)
from gui_agent.simulation.state import FaultProfile, TestbedState

Difficulty: TypeAlias = Literal["easy", "medium", "hard"]
_QUOTED_TEXT = re.compile(r"'([^']+)'")


class EvaluationTask(StrictFrozenModel):
    """Define a simulated task with a bounded reference solution."""

    id: str = Field(min_length=1, max_length=80)
    app: AppId
    difficulty: Difficulty
    instruction: str = Field(min_length=1, max_length=1000)
    success_criteria: str = Field(min_length=1, max_length=1000)
    initial_state: dict[str, object]
    fault_profile: FaultProfile = "none"
    max_steps: int = Field(ge=1, le=20)
    reference_actions: tuple[AgentAction, ...] = Field(min_length=1, max_length=20)
    tags: tuple[str, ...] = ()

    @field_validator("success_criteria")
    @classmethod
    def _requires_quoted_text(cls, value: str) -> str:
        if not _QUOTED_TEXT.search(value):
            raise ValueError("success criteria must contain single-quoted text")
        return value

    @field_validator("tags")
    @classmethod
    def _unique_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(tag.strip().casefold() for tag in value)
        if any(not tag for tag in normalized):
            raise ValueError("task tags must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("task tags must be unique")
        return normalized

    @model_validator(mode="after")
    def _valid_reference_solution(self) -> "EvaluationTask":
        if len(self.reference_actions) > self.max_steps:
            raise ValueError("reference actions must fit inside max_steps")
        if not isinstance(self.reference_actions[-1], FinishAction):
            raise ValueError("reference actions must end with a finish action")
        if not self.reference_actions[-1].success:
            raise ValueError("reference finish action must report success")
        unknown = set(self.initial_state) - {"active_app"}
        if unknown:
            raise ValueError(f"unsupported initial state fields: {sorted(unknown)}")
        active_app = self.initial_state.get("active_app", self.app)
        if active_app not in {"browser", "files", "messages", "settings", "editor"}:
            raise ValueError("initial_state active_app is invalid")
        return self


class EvaluationTaskSuite(StrictFrozenModel):
    """Store the fixed version-one suite of exactly twenty unique tasks."""

    schema_version: Literal[1] = 1
    kind: Literal["gui-agent-week7-task-suite"] = "gui-agent-week7-task-suite"
    tasks: tuple[EvaluationTask, ...]

    @field_validator("tasks")
    @classmethod
    def _valid_tasks(cls, value: tuple[EvaluationTask, ...]) -> tuple[EvaluationTask, ...]:
        ids = [task.id for task in value]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation task IDs must be unique")
        if len(value) != 20:
            raise ValueError("the Week 7 task suite must contain exactly 20 tasks")
        return value


@dataclass(frozen=True, slots=True)
class ReferenceSolutionResult:
    """Summarize deterministic reference execution for one task."""

    task_id: str
    succeeded: bool
    action_count: int
    message: str


def load_task_suite(path: Path) -> tuple[EvaluationTask, ...]:
    """Load and strictly validate the fixed Week 7 task suite."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"could not read evaluation task suite: {path}") from error
    try:
        suite = EvaluationTaskSuite.model_validate_json(raw)
    except ValueError as error:
        raise ValueError(f"invalid evaluation task suite: {path}") from error
    return suite.tasks


def run_reference_solution(
    task: EvaluationTask,
    *,
    root: Path,
    canvas: tuple[int, int] = (1280, 720),
) -> ReferenceSolutionResult:
    """Run one reference solution entirely inside the simulated desktop."""
    state = TestbedState(root, fault_profile=task.fault_profile)
    state.activate_app(task.initial_state.get("active_app", task.app))  # type: ignore[arg-type]
    desktop = SimulatedDesktop(state, canvas=canvas)
    observer = SimulatedObservationSource(desktop, mode="oracle")
    executor = SimulatedActionExecutor(desktop)
    policy = SimulationPolicy()
    verifier = RuleBasedOutcomeVerifier(success_criteria=task.success_criteria)

    for step_index, action in enumerate(task.reference_actions):
        before = observer.observe(step_index)
        try:
            policy.authorize(action, before, expected_outcome=task.success_criteria)
        except Exception as error:
            return ReferenceSolutionResult(
                task.id,
                False,
                step_index + 1,
                f"reference action was denied: {error}",
            )
        execution = executor.execute(action, step_index=step_index)
        if execution.status != "executed":
            return ReferenceSolutionResult(task.id, False, step_index + 1, execution.message)
        after = observer.observe(step_index + 1)
        if isinstance(action, FinishAction):
            decision = AgentDecision(
                current_step_id="reference",
                rationale_summary="Execute the deterministic reference solution.",
                action=action,
                expected_outcome=task.success_criteria,
            )
            verification = verifier.verify(before, decision, execution, after)
            return ReferenceSolutionResult(
                task.id,
                verification.passed,
                step_index + 1,
                verification.summary,
            )
    return ReferenceSolutionResult(
        task.id,
        False,
        len(task.reference_actions),
        "reference solution did not finish",
    )


def _scaled_reference_action(
    action: AgentAction,
    canvas: tuple[int, int],
) -> AgentAction:
    scale_x = canvas[0] / 1280
    scale_y = canvas[1] / 720
    if isinstance(action, ClickAction):
        return action.model_copy(
            update={"x": round(action.x * scale_x), "y": round(action.y * scale_y)}
        )
    if isinstance(action, ScrollAction) and action.x is not None and action.y is not None:
        return action.model_copy(
            update={"x": round(action.x * scale_x), "y": round(action.y * scale_y)}
        )
    if isinstance(action, DragAction):
        return action.model_copy(
            update={
                "start_x": round(action.start_x * scale_x),
                "start_y": round(action.start_y * scale_y),
                "end_x": round(action.end_x * scale_x),
                "end_y": round(action.end_y * scale_y),
            }
        )
    return action


def build_reference_planner(
    task: EvaluationTask,
    *,
    canvas: tuple[int, int] = (1280, 720),
) -> FakePlanner:
    """Build a deterministic planner whose actions scale to the canvas."""
    actions = tuple(
        _scaled_reference_action(action, canvas) for action in task.reference_actions
    )
    steps = tuple(
        TaskStep(
            id=f"reference-{index + 1}",
            description=f"Execute reference {action.kind} action",
        )
        for index, action in enumerate(actions)
    )
    decisions = tuple(
        AgentDecision(
            current_step_id=steps[index].id,
            rationale_summary="Follow the validated deterministic reference solution.",
            action=action,
            expected_outcome=(
                task.success_criteria
                if isinstance(action, FinishAction)
                else f"The simulated desktop reflects the {action.kind} action."
            ),
        )
        for index, action in enumerate(actions)
    )
    return FakePlanner(
        plan=TaskPlan(goal=task.instruction, steps=steps),
        decisions=decisions,
    )


__all__ = [
    "Difficulty",
    "EvaluationTask",
    "EvaluationTaskSuite",
    "ReferenceSolutionResult",
    "build_reference_planner",
    "load_task_suite",
    "run_reference_solution",
]
