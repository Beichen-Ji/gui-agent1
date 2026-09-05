import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from gui_agent.agent.events import (
    AgentEvent,
    EventEmitter,
    JSONLEventSink,
    action_metadata,
    goal_metadata,
    observation_metadata,
)
from gui_agent.agent.loop import GUIAgent
from gui_agent.agent.planner import FakePlanner
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    ClickAction,
    FinishAction,
    Observation,
    StepResult,
    TaskPlan,
    TaskStep,
    TypeTextAction,
)
from gui_agent.types import BoundingBox, OCRDetection, Point, ScreenshotResult


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


def observation(step_index: int) -> Observation:
    return Observation(
        screenshot=ScreenshotResult(
            image=np.full((20, 30, 3), step_index, dtype=np.uint8),
            monitor_index=1,
            captured_at=datetime(2026, 9, 5, tzinfo=UTC),
            origin=Point(0, 0),
        ),
        detections=(
            OCRDetection("Private screen text", 0.9, BoundingBox(1, 1, 20, 8)),
        ),
        step_index=step_index,
    )


def test_event_emitter_injects_stable_sequence_utc_time_and_run_id() -> None:
    sink = RecordingSink()
    start = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    timestamps = iter((start, start + timedelta(seconds=1)))
    emitter = EventEmitter(sink, run_id="run-fixed", clock=lambda: next(timestamps))

    emitter.emit("run_started", {"goal_sha256": "a" * 64})
    emitter.emit("run_finished", {"status": "succeeded"})

    assert [event.sequence for event in sink.events] == [1, 2]
    assert {event.run_id for event in sink.events} == {"run-fixed"}
    assert all(event.timestamp.utcoffset() == timedelta(0) for event in sink.events)


def test_jsonl_sink_writes_exactly_one_json_object_per_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = JSONLEventSink(path)
    event = AgentEvent(
        sequence=1,
        timestamp=datetime(2026, 9, 5, tzinfo=UTC),
        run_id="run-1",
        kind="run_finished",
        payload={"status": "succeeded"},
    )

    sink.emit(event)
    sink.emit(event)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["kind"] == "run_finished" for line in lines)


def test_event_metadata_redacts_goal_typed_text_and_ocr_content() -> None:
    goal = "Open a private local file"
    typed = "do-not-store-this-secret"
    current = observation(0)

    goal_payload = goal_metadata(goal)
    action_payload = action_metadata(TypeTextAction(text=typed))
    observation_payload = observation_metadata(current)
    serialized = json.dumps(
        {**goal_payload, **action_payload, **observation_payload},
        ensure_ascii=False,
    )

    assert goal_payload == {"goal_sha256": hashlib.sha256(goal.encode()).hexdigest()}
    assert action_payload == {"action_kind": "type_text", "text_length": len(typed)}
    assert observation_payload["ocr_count"] == 1
    assert len(str(observation_payload["ocr_summary_sha256"])) == 64
    assert goal not in serialized
    assert typed not in serialized
    assert "Private screen text" not in serialized


class Observer:
    def observe(self, step_index: int) -> Observation:
        return observation(step_index)


class Policy:
    def authorize(
        self,
        action: AgentAction,
        observation: Observation,
        *,
        expected_outcome: str,
    ) -> None:
        del action, observation, expected_outcome


class Executor:
    def __init__(self) -> None:
        self.calls: list[AgentAction] = []

    def execute(self, action: AgentAction, *, step_index: int) -> StepResult:
        self.calls.append(action)
        return StepResult(
            step_index=step_index,
            action=action,
            status="dry_run",
            message="safe preview",
        )


def planner() -> FakePlanner:
    plan = TaskPlan(
        goal="Open the browser",
        steps=(TaskStep(id="step-1", description="Open it"),),
    )
    return FakePlanner(
        plan=plan,
        decisions=(
            AgentDecision(
                current_step_id="step-1",
                rationale_summary="Click the visible control",
                action=ClickAction(x=10, y=10),
                expected_outcome="The browser opens",
            ),
            AgentDecision(
                current_step_id="step-1",
                rationale_summary="The task is complete",
                action=FinishAction(success=True, summary="Done"),
                expected_outcome="The loop stops",
            ),
        ),
    )


def test_loop_streams_lifecycle_events_and_always_finishes() -> None:
    sink = RecordingSink()

    result = GUIAgent(
        Observer(),
        planner(),
        Policy(),
        Executor(),
        event_sink=sink,
        run_id_factory=lambda: "run-fixed",
    ).run("Open the browser")

    kinds = [event.kind for event in sink.events]
    assert result.status == "succeeded"
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_finished"
    for required in (
        "plan_created",
        "step_started",
        "observation_completed",
        "action_proposed",
        "action_authorized",
        "action_executed",
        "verification_completed",
    ):
        assert required in kinds


def test_planner_exception_still_emits_run_finished() -> None:
    sink = RecordingSink()
    empty = FakePlanner(
        plan=TaskPlan(
            goal="Open the browser",
            steps=(TaskStep(id="step-1", description="Open it"),),
        ),
        decisions=(),
    )

    result = GUIAgent(
        Observer(),
        empty,
        Policy(),
        Executor(),
        event_sink=sink,
    ).run("Open the browser")

    assert result.status == "failed"
    assert sink.events[-1].kind == "run_finished"


class FailingSink:
    def emit(self, event: AgentEvent) -> None:
        if event.kind == "action_executed":
            raise OSError("log unavailable")


def test_sink_failure_never_repeats_a_desktop_action() -> None:
    executor = Executor()

    result = GUIAgent(
        Observer(),
        planner(),
        Policy(),
        executor,
        event_sink=FailingSink(),
    ).run("Open the browser")

    assert result.status == "succeeded"
    assert [action.kind for action in executor.calls] == ["click", "finish"]

