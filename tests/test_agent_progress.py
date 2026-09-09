from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from gui_agent.agent.planner import FakePlanner, MultimodalPlanner, PlannerError
from gui_agent.agent.types import (
    AgentState,
    Observation,
    PlanProgress,
    ReplanContext,
    StepProgress,
    TaskPlan,
    TaskStep,
    reconcile_revised_plan,
)
from gui_agent.types import Point, ScreenshotResult


def plan(*step_ids: str) -> TaskPlan:
    return TaskPlan(
        goal="Complete the local workflow",
        steps=tuple(
            TaskStep(id=step_id, description=f"Do {step_id}")
            for step_id in step_ids
        ),
    )


def observation() -> Observation:
    return Observation(
        screenshot=ScreenshotResult(
            image=np.zeros((20, 30, 3), dtype=np.uint8),
            monitor_index=1,
            captured_at=datetime(2026, 9, 5, tzinfo=UTC),
            origin=Point(0, 0),
        ),
        detections=(),
        step_index=0,
    )


def state(task_plan: TaskPlan, progress: PlanProgress) -> AgentState:
    return AgentState(
        goal=task_plan.goal,
        plan=task_plan,
        progress=progress,
        observation=observation(),
        decisions=(),
        results=(),
    )


def test_task_plan_requires_unique_stable_step_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        plan("step-1", "step-1")


def test_progress_initializes_one_active_step_and_is_immutable() -> None:
    progress = PlanProgress.from_plan(plan("step-1", "step-2"))

    assert progress.active_step_id == "step-1"
    assert [step.status for step in progress.steps] == ["active", "pending"]
    assert [step.attempts for step in progress.steps] == [0, 0]
    assert progress.replan_count == 0
    with pytest.raises(ValidationError, match="frozen"):
        progress.active_step_id = "step-2"


def test_progress_records_only_the_active_step_attempt() -> None:
    progress = PlanProgress.from_plan(plan("step-1", "step-2"))

    attempted = progress.record_attempt("step-1")

    assert attempted.steps[0].attempts == 1
    assert progress.steps[0].attempts == 0
    with pytest.raises(ValueError, match="active"):
        progress.record_attempt("step-2")


def test_selecting_next_step_requires_verified_current_step() -> None:
    progress = PlanProgress.from_plan(plan("step-1", "step-2", "step-3"))

    with pytest.raises(ValueError, match="successful verification"):
        progress.select_step("step-2")

    advanced = progress.select_step("step-2", verified_step_id="step-1")

    assert advanced.active_step_id == "step-2"
    assert [step.status for step in advanced.steps] == [
        "completed",
        "active",
        "pending",
    ]
    with pytest.raises(ValueError, match="completed"):
        advanced.select_step("step-1")
    with pytest.raises(ValueError, match="next pending"):
        progress.select_step("step-3", verified_step_id="step-1")
    with pytest.raises(ValueError, match="not part"):
        progress.select_step("invented-step")


def test_completing_last_step_has_terminal_progress() -> None:
    progress = PlanProgress.from_plan(plan("step-1")).record_attempt("step-1")

    completed = progress.complete_active()

    assert completed.is_complete is True
    assert completed.active_step_id == "step-1"
    assert completed.steps == (
        StepProgress(step_id="step-1", status="completed", attempts=1),
    )
    with pytest.raises(ValueError, match="already complete"):
        completed.complete_active()


def test_replan_preserves_completed_facts_and_uses_stable_ids() -> None:
    original = plan("step-1", "step-2")
    progress = PlanProgress.from_plan(original).select_step(
        "step-2",
        verified_step_id="step-1",
    )
    proposed = TaskPlan(
        goal="A model tried to rewrite the goal",
        steps=(
            TaskStep(id="recovery-1", description="Recover the correct tab"),
            TaskStep(id="step-2", description="Finish the original task"),
        ),
    )

    revised, revised_progress = reconcile_revised_plan(original, progress, proposed)

    assert revised.goal == original.goal
    assert revised.steps[0] == original.steps[0]
    assert [step.id for step in revised.steps] == [
        "step-1",
        "recovery-1",
        "step-2",
    ]
    assert [step.status for step in revised_progress.steps] == [
        "completed",
        "active",
        "pending",
    ]
    assert revised_progress.replan_count == 1

    with pytest.raises(ValueError, match="replan limit"):
        reconcile_revised_plan(revised, revised_progress, proposed)


def test_agent_state_rejects_progress_from_a_different_plan() -> None:
    with pytest.raises(ValueError, match="progress steps"):
        state(plan("step-1"), PlanProgress.from_plan(plan("other-step")))


def test_fake_planner_returns_revised_plans_deterministically() -> None:
    original = plan("step-1")
    revised = plan("recovery-1")
    progress = PlanProgress.from_plan(original)
    agent_state = state(original, progress)
    failure = ReplanContext(reason_code="no_visual_change", summary="Nothing changed")
    planner = FakePlanner(
        plan=original,
        decisions=(),
        revised_plans=(revised,),
    )

    assert isinstance(planner, MultimodalPlanner)
    assert planner.revise_plan(agent_state, failure) == revised
    with pytest.raises(PlannerError, match="revised plan"):
        planner.revise_plan(agent_state, failure)
