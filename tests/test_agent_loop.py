from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.loop import GUIAgent
from gui_agent.agent.planner import FakePlanner
from gui_agent.agent.policy import SafetyPolicy
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    AgentState,
    ClickAction,
    FinishAction,
    Observation,
    StepResult,
    TaskPlan,
    TaskStep,
)
from gui_agent.control.controller import DesktopController
from gui_agent.types import Point, ScreenRegion, ScreenshotResult

BOUNDS = ScreenRegion(left=0, top=0, width=100, height=80)


def make_observation(step_index: int) -> Observation:
    image = np.full((80, 100, 3), step_index, dtype=np.uint8)
    screenshot = ScreenshotResult(
        image=image,
        monitor_index=1,
        captured_at=datetime(2026, 8, 25, tzinfo=UTC),
        origin=Point(0, 0),
    )
    return Observation(screenshot=screenshot, detections=(), step_index=step_index)


def make_plan(goal: str = "Open Browser") -> TaskPlan:
    return TaskPlan(
        goal=goal,
        steps=(TaskStep(id="step-1", description="Complete the task"),),
    )


def decision(action: AgentAction) -> AgentDecision:
    return AgentDecision(
        current_step_id="step-1",
        rationale_summary="The next safe action is visible.",
        action=action,
        expected_outcome="The local testbed advances.",
    )


class RecordingObserver:
    def __init__(
        self,
        events: list[str],
        *,
        fail_on_step: int | None = None,
    ) -> None:
        self.events = events
        self.fail_on_step = fail_on_step
        self.calls: list[int] = []

    def observe(self, step_index: int) -> Observation:
        self.events.append(f"observe:{step_index}")
        self.calls.append(step_index)
        if step_index == self.fail_on_step:
            raise RuntimeError("observation failed")
        return make_observation(step_index)


class RecordingPlanner(FakePlanner):
    def __init__(
        self,
        events: list[str],
        *,
        decisions: Iterable[AgentDecision],
        fail_create: bool = False,
        fail_next_call: int | None = None,
    ) -> None:
        super().__init__(plan=make_plan(), decisions=decisions)
        self.events = events
        self.fail_create = fail_create
        self.fail_next_call = fail_next_call
        self.create_calls = 0
        self.states: list[AgentState] = []

    def create_plan(self, goal: str, observation: Observation) -> TaskPlan:
        self.events.append("create_plan")
        self.create_calls += 1
        if self.fail_create:
            raise RuntimeError("planning failed")
        return super().create_plan(goal, observation)

    def next_action(self, state: AgentState) -> AgentDecision:
        self.events.append(f"next_action:{state.observation.step_index}")
        self.states.append(state)
        if self.fail_next_call is not None and len(self.states) == self.fail_next_call:
            raise RuntimeError("next action failed")
        return super().next_action(state)


class RecordingPolicy:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail
        self.policy = SafetyPolicy()

    def authorize(
        self,
        action: AgentAction,
        observation: Observation,
        *,
        expected_outcome: str,
    ) -> None:
        self.events.append(f"authorize:{action.kind}:{observation.step_index}")
        if self.fail:
            raise RuntimeError("policy unavailable")
        self.policy.authorize(
            action,
            observation,
            expected_outcome=expected_outcome,
        )


class RecordingExecutor(ActionExecutor):
    def __init__(
        self,
        events: list[str],
        *,
        fail: bool = False,
    ) -> None:
        controller = DesktopController(
            dry_run=True,
            bounds_provider=lambda: BOUNDS,
        )
        super().__init__(controller, clock=lambda _seconds: None)
        self.events = events
        self.fail = fail
        self.calls: list[AgentAction] = []

    def execute(self, action: AgentAction, *, step_index: int) -> StepResult:
        self.events.append(f"execute:{action.kind}:{step_index}")
        self.calls.append(action)
        if self.fail:
            raise RuntimeError("execution failed")
        return super().execute(action, step_index=step_index)


@pytest.mark.parametrize(
    ("goal", "max_steps", "message"),
    [
        ("", 1, "goal"),
        ("Open Browser", 0, "max_steps"),
        ("Open Browser", True, "max_steps"),
        ("Open Browser", 1.5, "max_steps"),
    ],
)
def test_loop_rejects_invalid_goal_or_step_limit(
    goal: str,
    max_steps: Any,
    message: str,
) -> None:
    events: list[str] = []
    agent = GUIAgent(
        RecordingObserver(events),
        RecordingPlanner(events, decisions=()),
        RecordingPolicy(events),
        RecordingExecutor(events),
    )

    with pytest.raises(ValueError, match=message):
        agent.run(goal, max_steps=max_steps)

    assert events == []


def test_loop_observes_plans_executes_and_feeds_results_in_order() -> None:
    events: list[str] = []
    click = decision(ClickAction(x=20, y=30))
    finish = decision(FinishAction(success=True, summary="Browser opened"))
    observer = RecordingObserver(events)
    planner = RecordingPlanner(events, decisions=(click, finish))
    policy = RecordingPolicy(events)
    executor = RecordingExecutor(events)

    result = GUIAgent(observer, planner, policy, executor).run("Open Browser", max_steps=4)

    assert result.status == "succeeded"
    assert result.failure_stage is None
    assert result.message == "Browser opened"
    assert result.decisions == (click, finish)
    assert len(result.results) == 2
    assert result.observation is not None
    assert result.observation.step_index == 2
    assert planner.create_calls == 1
    assert planner.states[1].results == (result.results[0],)
    assert planner.states[1].observation.step_index == 1
    assert events == [
        "observe:0",
        "create_plan",
        "next_action:0",
        "authorize:click:0",
        "execute:click:0",
        "observe:1",
        "next_action:1",
        "authorize:finish:1",
        "execute:finish:1",
        "observe:2",
    ]


def test_loop_stops_at_max_steps_after_refreshing_observation() -> None:
    events: list[str] = []
    planner = RecordingPlanner(
        events,
        decisions=(decision(ClickAction(x=20, y=30)),),
    )
    observer = RecordingObserver(events)
    executor = RecordingExecutor(events)

    result = GUIAgent(
        observer,
        planner,
        RecordingPolicy(events),
        executor,
    ).run("Open Browser", max_steps=1)

    assert result.status == "stopped"
    assert result.failure_stage is None
    assert "maximum step count" in result.message
    assert observer.calls == [0, 1]
    assert len(executor.calls) == 1


def test_loop_stops_before_executing_second_identical_action() -> None:
    events: list[str] = []
    repeated = decision(ClickAction(x=20, y=30))
    executor = RecordingExecutor(events)

    result = GUIAgent(
        RecordingObserver(events),
        RecordingPlanner(events, decisions=(repeated, repeated)),
        RecordingPolicy(events),
        executor,
    ).run("Open Browser", max_steps=5)

    assert result.status == "stopped"
    assert "repeated action" in result.message
    assert result.decisions == (repeated, repeated)
    assert len(result.results) == 1
    assert executor.calls == [repeated.action]


def test_finish_with_unsuccessful_result_returns_task_failure() -> None:
    events: list[str] = []
    failed_finish = decision(FinishAction(success=False, summary="Target not found"))

    result = GUIAgent(
        RecordingObserver(events),
        RecordingPlanner(events, decisions=(failed_finish,)),
        RecordingPolicy(events),
        RecordingExecutor(events),
    ).run("Open Browser")

    assert result.status == "failed"
    assert result.failure_stage == "task"
    assert result.message == "Target not found"


@pytest.mark.parametrize(
    ("observer_step", "planner_create", "expected_stage"),
    [(0, False, "observation"), (None, True, "planning")],
)
def test_loop_reports_initial_observation_or_plan_failure(
    observer_step: int | None,
    planner_create: bool,
    expected_stage: str,
) -> None:
    events: list[str] = []
    result = GUIAgent(
        RecordingObserver(events, fail_on_step=observer_step),
        RecordingPlanner(events, decisions=(), fail_create=planner_create),
        RecordingPolicy(events),
        RecordingExecutor(events),
    ).run("Open Browser")

    assert result.status == "failed"
    assert result.failure_stage == expected_stage
    assert result.decisions == ()
    assert result.results == ()


def test_loop_preserves_completed_result_when_next_planning_fails() -> None:
    events: list[str] = []
    first = decision(ClickAction(x=20, y=30))

    result = GUIAgent(
        RecordingObserver(events),
        RecordingPlanner(events, decisions=(first,), fail_next_call=2),
        RecordingPolicy(events),
        RecordingExecutor(events),
    ).run("Open Browser")

    assert result.status == "failed"
    assert result.failure_stage == "planning"
    assert result.decisions == (first,)
    assert len(result.results) == 1
    assert result.observation is not None
    assert result.observation.step_index == 1


def test_loop_preserves_executed_step_when_refresh_observation_fails() -> None:
    events: list[str] = []
    first = decision(ClickAction(x=20, y=30))

    result = GUIAgent(
        RecordingObserver(events, fail_on_step=1),
        RecordingPlanner(events, decisions=(first,)),
        RecordingPolicy(events),
        RecordingExecutor(events),
    ).run("Open Browser")

    assert result.status == "failed"
    assert result.failure_stage == "observation"
    assert result.decisions == (first,)
    assert len(result.results) == 1
    assert result.observation is not None
    assert result.observation.step_index == 0


def test_loop_turns_policy_failure_into_denied_step_result() -> None:
    events: list[str] = []
    action = decision(ClickAction(x=20, y=30))

    result = GUIAgent(
        RecordingObserver(events),
        RecordingPlanner(events, decisions=(action,)),
        RecordingPolicy(events, fail=True),
        RecordingExecutor(events),
    ).run("Open Browser")

    assert result.status == "failed"
    assert result.failure_stage == "policy"
    assert len(result.results) == 1
    assert result.results[0].status == "denied"


def test_loop_turns_executor_failure_into_failed_step_result() -> None:
    events: list[str] = []
    action = decision(ClickAction(x=20, y=30))

    result = GUIAgent(
        RecordingObserver(events),
        RecordingPlanner(events, decisions=(action,)),
        RecordingPolicy(events),
        RecordingExecutor(events, fail=True),
    ).run("Open Browser")

    assert result.status == "failed"
    assert result.failure_stage == "execution"
    assert len(result.results) == 1
    assert result.results[0].status == "failed"
