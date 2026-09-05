import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from gui_agent.agent.events import AgentEvent
from gui_agent.agent.loop import GUIAgent
from gui_agent.agent.planner import FakePlanner
from gui_agent.agent.policy import ActionDeniedError, SafetyPolicy
from gui_agent.agent.retry import RetryPolicy
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    ClickAction,
    FinishAction,
    Observation,
    StepResult,
    TaskPlan,
    TaskStep,
    WaitAction,
)
from gui_agent.types import Point, ScreenshotResult

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "week6_robustness_tasks.json"


class EventRecorder:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class SequenceObserver:
    def __init__(self, values: Iterable[int], *, initial_failures: int = 0) -> None:
        self._values = iter(values)
        self._initial_failures = initial_failures

    def observe(self, step_index: int) -> Observation:
        if self._initial_failures:
            self._initial_failures -= 1
            raise RuntimeError("transient OCR error")
        value = next(self._values)
        return Observation(
            screenshot=ScreenshotResult(
                image=np.full((40, 60, 3), value, dtype=np.uint8),
                monitor_index=1,
                captured_at=datetime(2026, 9, 5, tzinfo=UTC),
                origin=Point(0, 0),
            ),
            detections=(),
            step_index=step_index,
        )


class AllowPolicy:
    def authorize(
        self,
        action: AgentAction,
        observation: Observation,
        *,
        expected_outcome: str,
    ) -> None:
        del action, observation, expected_outcome


class RecordingExecutor:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[AgentAction] = []

    def execute(self, action: AgentAction, *, step_index: int) -> StepResult:
        self.calls.append(action)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transient execution error")
        return StepResult(
            step_index=step_index,
            action=action,
            status="executed",
            message="synthetic action executed",
        )


class DenyPolicy:
    def authorize(
        self,
        action: AgentAction,
        observation: Observation,
        *,
        expected_outcome: str,
    ) -> None:
        del action, observation, expected_outcome
        raise ActionDeniedError("action point is outside the current observation")


def task_plan(step_id: str = "step-1") -> TaskPlan:
    return TaskPlan(
        goal="Complete the synthetic recovery task",
        steps=(TaskStep(id=step_id, description="Use the safe visible control"),),
    )


def decision(action: AgentAction, *, step_id: str = "step-1") -> AgentDecision:
    return AgentDecision(
        current_step_id=step_id,
        rationale_summary="Use a deterministic synthetic action",
        action=action,
        expected_outcome="The synthetic frame changes",
    )


def finish(*, step_id: str = "step-1") -> AgentDecision:
    return decision(
        FinishAction(success=True, summary="Synthetic task completed"),
        step_id=step_id,
    )


def build_scenario(
    scenario_id: str,
) -> tuple[GUIAgent, RecordingExecutor, EventRecorder, list[float], list[str]]:
    events = EventRecorder()
    executor = RecordingExecutor()
    clock: list[float] = []
    confirmation_prompts: list[str] = []
    policy: AllowPolicy | DenyPolicy | SafetyPolicy = AllowPolicy()
    retry_policy = RetryPolicy()
    max_replans = 1
    revised_plans: tuple[TaskPlan, ...] = ()
    decisions: tuple[AgentDecision, ...]

    if scenario_id == "transient-ocr":
        observer = SequenceObserver((0, 1, 1), initial_failures=1)
        decisions = (decision(ClickAction(x=10, y=10)), finish())
    elif scenario_id == "no-visual-change":
        observer = SequenceObserver((0, 0, 1, 1))
        decisions = (
            decision(ClickAction(x=10, y=10)),
            decision(ClickAction(x=20, y=10)),
            finish(),
        )
    elif scenario_id == "delayed-result":
        observer = SequenceObserver((0, 0, 1, 1))
        decisions = (
            decision(ClickAction(x=10, y=10)),
            decision(WaitAction(seconds=0.5)),
            finish(),
        )
    elif scenario_id == "wrong-tab-replan":
        observer = SequenceObserver((0, 0, 1, 1))
        decisions = (
            decision(ClickAction(x=10, y=10)),
            decision(ClickAction(x=20, y=10), step_id="recovery-1"),
            finish(step_id="recovery-1"),
        )
        retry_policy = RetryPolicy(max_retries_per_step=0)
        revised_plans = (task_plan("recovery-1"),)
    elif scenario_id == "transient-executor":
        observer = SequenceObserver((0, 0, 1, 1))
        decisions = (
            decision(ClickAction(x=10, y=10)),
            decision(ClickAction(x=20, y=10)),
            finish(),
        )
        executor = RecordingExecutor(failures=1)
    elif scenario_id == "policy-denied":
        observer = SequenceObserver((0,))
        decisions = (decision(ClickAction(x=100, y=100)),)
        policy = DenyPolicy()
    elif scenario_id == "confirmation-rejected":
        observer = SequenceObserver((0,))
        decisions = (decision(ClickAction(x=10, y=10)),)

        def reject_confirmation(prompt: str) -> str:
            confirmation_prompts.append(prompt)
            return "no"

        policy = SafetyPolicy(execute=True, input_fn=reject_confirmation)
    elif scenario_id == "recovery-exhausted":
        observer = SequenceObserver((0, 0, 0, 0, 0))
        decisions = (
            decision(ClickAction(x=10, y=10)),
            decision(ClickAction(x=20, y=10)),
            decision(ClickAction(x=30, y=10), step_id="recovery-1"),
            decision(ClickAction(x=40, y=10), step_id="recovery-1"),
        )
        retry_policy = RetryPolicy(max_retries_per_step=1)
        revised_plans = (task_plan("recovery-1"),)
    else:
        raise AssertionError(f"unknown test scenario: {scenario_id}")

    planner = FakePlanner(
        plan=task_plan(),
        decisions=decisions,
        revised_plans=revised_plans,
    )
    agent = GUIAgent(
        observer,
        planner,
        policy,
        executor,
        retry_policy=retry_policy,
        max_replans=max_replans,
        clock=clock.append,
        event_sink=events,
        run_id_factory=lambda: f"run-{scenario_id}",
    )
    return agent, executor, events, clock, confirmation_prompts


def configured_scenarios() -> list[dict[str, object]]:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    scenarios = payload["scenarios"]
    assert isinstance(scenarios, list)
    assert len(scenarios) == 8
    return scenarios


@pytest.mark.integration
@pytest.mark.parametrize("scenario", configured_scenarios(), ids=lambda item: item["id"])
def test_week6_failure_injection_scenario(scenario: dict[str, object]) -> None:
    scenario_id = str(scenario["id"])
    agent, executor, events, clock, confirmation_prompts = build_scenario(scenario_id)

    result = agent.run("Complete the synthetic recovery task", max_steps=8)

    event_kinds = [event.kind for event in events.events]
    assert result.status == scenario["expected_status"]
    assert result.reason_code == scenario["expected_reason_code"]
    assert event_kinds.count("retry_scheduled") == scenario["expected_retries"]
    assert event_kinds.count("plan_revised") == scenario["expected_replans"]
    assert event_kinds[-1] == "run_finished"
    assert len(executor.calls) == scenario["expected_actions"]
    assert sum(clock) == pytest.approx(scenario["expected_recovery_seconds"])
    assert len(confirmation_prompts) == scenario["expected_confirmation_prompts"]
    if scenario_id in {"policy-denied", "confirmation-rejected"}:
        assert clock == []
