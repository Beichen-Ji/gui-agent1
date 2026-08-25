from datetime import UTC, datetime

import numpy as np
import pytest

from gui_agent.agent.planner import FakePlanner, MultimodalPlanner, PlannerError
from gui_agent.agent.prompts import build_action_prompt, build_plan_prompt
from gui_agent.agent.types import (
    AgentDecision,
    AgentState,
    ClickAction,
    FinishAction,
    Observation,
    StepResult,
    TaskPlan,
    TaskStep,
)
from gui_agent.types import BoundingBox, OCRDetection, Point, ScreenshotResult


def observation() -> Observation:
    screenshot = ScreenshotResult(
        image=np.zeros((600, 800, 3), dtype=np.uint8),
        monitor_index=1,
        captured_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        origin=Point(-100, 20),
    )
    detections = (
        OCRDetection("Save", 0.95, BoundingBox(10, 20, 60, 50)),
        OCRDetection(
            r"C:\Users\student\secret.txt sk-example123456",
            0.80,
            BoundingBox(70, 20, 300, 50),
        ),
    )
    return Observation(screenshot=screenshot, detections=detections, step_index=0)


def task_plan() -> TaskPlan:
    return TaskPlan(
        goal="Open the browser",
        steps=(
            TaskStep(id="step-1", description="Click the browser icon"),
            TaskStep(id="step-2", description="Confirm the window opened"),
        ),
    )


def test_plan_prompt_contains_goal_screen_ocr_and_action_allowlist() -> None:
    prompt = build_plan_prompt("Open the browser", observation())

    assert "Open the browser" in prompt
    assert "800x600" in prompt
    assert "origin=(-100,20)" in prompt
    assert 'text="Save"' in prompt
    for action_kind in ("click", "type_text", "hotkey", "scroll", "drag", "wait", "finish"):
        assert action_kind in prompt


def test_prompt_redacts_absolute_paths_and_likely_api_keys() -> None:
    prompt = build_plan_prompt("Inspect the visible screen", observation())

    assert r"C:\Users\student\secret.txt" not in prompt
    assert "sk-example123456" not in prompt
    assert "[local-path]" in prompt
    assert "[secret]" in prompt


def test_action_prompt_includes_plan_step_and_recent_result() -> None:
    first_decision = AgentDecision(
        current_step_id="step-1",
        rationale_summary="The icon is visible",
        action=ClickAction(x=20, y=30),
        expected_outcome="The browser window appears",
    )
    result = StepResult(
        step_index=0,
        action=first_decision.action,
        status="dry_run",
        message="Dry-run click recorded",
    )
    state = AgentState(
        goal="Open the browser",
        plan=task_plan(),
        observation=observation(),
        decisions=(first_decision,),
        results=(result,),
    )

    prompt = build_action_prompt(state)

    assert "step-1" in prompt
    assert "Click the browser icon" in prompt
    assert "dry_run" in prompt
    assert "Dry-run click recorded" in prompt
    assert "step_index=0" in prompt


def test_fake_planner_returns_configured_plan_and_decisions_in_order() -> None:
    plan = task_plan()
    decisions = (
        AgentDecision(
            current_step_id="step-1",
            rationale_summary="The icon is visible",
            action=ClickAction(x=20, y=30),
            expected_outcome="The browser window appears",
        ),
        AgentDecision(
            current_step_id="step-2",
            rationale_summary="The browser is open",
            action=FinishAction(success=True, summary="Browser opened"),
            expected_outcome="The run stops successfully",
        ),
    )
    planner = FakePlanner(plan=plan, decisions=decisions)
    initial = observation()

    assert isinstance(planner, MultimodalPlanner)
    assert planner.create_plan("Open the browser", initial) == plan
    state = AgentState(
        goal="Open the browser",
        plan=plan,
        observation=initial,
        decisions=(),
        results=(),
    )
    assert planner.next_action(state) == decisions[0]
    assert planner.next_action(state) == decisions[1]


def test_fake_planner_reports_decision_queue_exhaustion() -> None:
    plan = task_plan()
    planner = FakePlanner(plan=plan, decisions=())
    state = AgentState(
        goal="Open the browser",
        plan=plan,
        observation=observation(),
        decisions=(),
        results=(),
    )

    with pytest.raises(PlannerError, match="no configured decision"):
        planner.next_action(state)
