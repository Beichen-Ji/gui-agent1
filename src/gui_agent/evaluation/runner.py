"""Run simulated tasks with stage timing and resumable report writes."""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from gui_agent.agent.events import JSONLEventSink
from gui_agent.agent.loop import GUIAgent
from gui_agent.agent.planner import MultimodalPlanner
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    AgentState,
    Observation,
    ReplanContext,
    StepResult,
    TaskPlan,
)
from gui_agent.evaluation.metrics import TaskOutcome, TimingBreakdown, outcome_from_run
from gui_agent.evaluation.report import (
    EvaluationContext,
    EvaluationReport,
    build_evaluation_report,
    load_evaluation_report,
    write_evaluation_report,
)
from gui_agent.evaluation.suite import EvaluationTask
from gui_agent.perception.ocr import OCRBackend
from gui_agent.simulation.harness import (
    SimulatedActionExecutor,
    SimulatedDesktop,
    SimulatedObservationSource,
    SimulationPolicy,
)
from gui_agent.simulation.state import TestbedState

PlannerFactory = Callable[[EvaluationTask, tuple[int, int]], MultimodalPlanner]
MonotonicClock = Callable[[], float]


@dataclass(slots=True)
class _TimingAccumulator:
    planner_ms: float = 0.0
    perception_ms: float = 0.0
    execution_ms: float = 0.0


class _TimedPlanner:
    def __init__(
        self,
        planner: MultimodalPlanner,
        timing: _TimingAccumulator,
        clock: MonotonicClock,
    ) -> None:
        self._planner = planner
        self._timing = timing
        self._clock = clock

    def create_plan(self, goal: str, observation: Observation) -> TaskPlan:
        started = self._clock()
        try:
            return self._planner.create_plan(goal, observation)
        finally:
            self._timing.planner_ms += (self._clock() - started) * 1000

    def next_action(self, state: AgentState) -> AgentDecision:
        started = self._clock()
        try:
            return self._planner.next_action(state)
        finally:
            self._timing.planner_ms += (self._clock() - started) * 1000

    def revise_plan(self, state: AgentState, failure: ReplanContext) -> TaskPlan:
        started = self._clock()
        try:
            return self._planner.revise_plan(state, failure)
        finally:
            self._timing.planner_ms += (self._clock() - started) * 1000


class _TimedObserver:
    def __init__(
        self,
        observer: SimulatedObservationSource,
        timing: _TimingAccumulator,
        clock: MonotonicClock,
    ) -> None:
        self._observer = observer
        self._timing = timing
        self._clock = clock

    def observe(self, step_index: int) -> Observation:
        started = self._clock()
        try:
            return self._observer.observe(step_index)
        finally:
            self._timing.perception_ms += (self._clock() - started) * 1000


class _TimedExecutor:
    def __init__(
        self,
        executor: SimulatedActionExecutor,
        timing: _TimingAccumulator,
        clock: MonotonicClock,
    ) -> None:
        self._executor = executor
        self._timing = timing
        self._clock = clock

    def execute(self, action: AgentAction, *, step_index: int) -> StepResult:
        started = self._clock()
        try:
            return self._executor.execute(action, step_index=step_index)
        finally:
            self._timing.execution_ms += (self._clock() - started) * 1000


def _run_task(
    task: EvaluationTask,
    *,
    context: EvaluationContext,
    resolution: tuple[int, int],
    planner_factory: PlannerFactory,
    output_dir: Path,
    ocr: OCRBackend | None,
    clock: MonotonicClock,
) -> TaskOutcome:
    task_dir = output_dir / task.id
    task_dir.mkdir(parents=True, exist_ok=True)
    events_path = task_dir / "events.jsonl"
    events_path.unlink(missing_ok=True)
    state = TestbedState(task_dir / "sandbox", fault_profile=task.fault_profile)
    state.activate_app(task.initial_state.get("active_app", task.app))  # type: ignore[arg-type]
    desktop = SimulatedDesktop(state, canvas=resolution)
    timing = _TimingAccumulator()
    observer = _TimedObserver(
        SimulatedObservationSource(
            desktop,
            mode=context.observation_mode,
            ocr=ocr,
        ),
        timing,
        clock,
    )
    executor = _TimedExecutor(SimulatedActionExecutor(desktop), timing, clock)
    planner = _TimedPlanner(planner_factory(task, resolution), timing, clock)
    agent = GUIAgent(
        observer,
        planner,
        SimulationPolicy(),
        executor,
        clock=desktop.advance,
        event_sink=JSONLEventSink(events_path),
    )
    started = clock()
    result = agent.run(
        task.instruction,
        success_criteria=task.success_criteria,
        max_steps=task.max_steps,
    )
    wall_ms = (clock() - started) * 1000
    return outcome_from_run(
        task,
        resolution,
        result,
        TimingBreakdown(
            wall_ms=max(0.0, wall_ms),
            planner_ms=max(0.0, timing.planner_ms),
            perception_ms=max(0.0, timing.perception_ms),
            execution_ms=max(0.0, timing.execution_ms),
        ),
    )


def run_evaluation(
    tasks: Iterable[EvaluationTask],
    *,
    context: EvaluationContext,
    resolution: tuple[int, int],
    planner_factory: PlannerFactory,
    output_dir: Path,
    ocr: OCRBackend | None = None,
    resume: bool = False,
    clock: MonotonicClock = time.perf_counter,
) -> EvaluationReport:
    """Run missing or invalid tasks and persist progress after each outcome."""
    requested = tuple(tasks)
    report_path = output_dir / "evaluation.json"
    existing: EvaluationReport | None = None
    if report_path.exists():
        if not resume:
            raise ValueError(f"evaluation output already exists: {report_path}")
        existing = load_evaluation_report(report_path)
        if not existing.matches_context(context):
            raise ValueError("existing evaluation report provenance does not match this run")

    outcomes = {
        (outcome.task_id, outcome.resolution): outcome
        for outcome in (existing.outcomes if existing is not None else ())
    }
    pending = tuple(
        task
        for task in requested
        if (task.id, resolution) not in outcomes
        or outcomes[(task.id, resolution)].status == "invalid"
    )
    if existing is not None and not pending:
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_evaluation_report(context, tuple(outcomes.values()))
    for task in pending:
        outcome = _run_task(
            task,
            context=context,
            resolution=resolution,
            planner_factory=planner_factory,
            output_dir=output_dir,
            ocr=ocr,
            clock=clock,
        )
        outcomes[(task.id, resolution)] = outcome
        report = build_evaluation_report(context, tuple(outcomes.values()))
        write_evaluation_report(report, report_path, overwrite=report_path.exists())
    return report


__all__ = ["MonotonicClock", "PlannerFactory", "run_evaluation"]
