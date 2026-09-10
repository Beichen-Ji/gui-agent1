import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import Field, model_validator

from gui_agent._models import StrictFrozenModel
from gui_agent.agent.loop import AgentRunResult
from gui_agent.agent.types import FailureReason
from gui_agent.evaluation.suite import Difficulty, EvaluationTask
from gui_agent.simulation.apps import AppId


class TimingBreakdown(StrictFrozenModel):
    wall_ms: float = Field(ge=0.0)
    planner_ms: float = Field(ge=0.0)
    perception_ms: float = Field(ge=0.0)
    execution_ms: float = Field(ge=0.0)


class TaskOutcome(StrictFrozenModel):
    task_id: str = Field(min_length=1, max_length=80)
    app: AppId
    difficulty: Difficulty
    resolution: tuple[int, int]
    status: Literal["succeeded", "failed", "stopped", "invalid"]
    reason_code: FailureReason | None
    invalid_reason: str | None = Field(default=None, min_length=1, max_length=120)
    step_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    replan_count: int = Field(ge=0)
    verification_pass_rate: float = Field(ge=0.0, le=1.0)
    wall_ms: float = Field(ge=0.0)
    planner_ms: float = Field(ge=0.0)
    perception_ms: float = Field(ge=0.0)
    execution_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _consistent_outcome(self) -> "TaskOutcome":
        width, height = self.resolution
        if width <= 0 or height <= 0:
            raise ValueError("outcome resolution must be positive")
        if self.status == "invalid" and self.invalid_reason is None:
            raise ValueError("invalid outcomes require an invalid reason")
        if self.status != "invalid" and self.invalid_reason is not None:
            raise ValueError("valid outcomes cannot have an invalid reason")
        if self.status == "succeeded" and self.reason_code is not None:
            raise ValueError("successful outcomes cannot have a failure reason")
        return self


class SuiteMetrics(StrictFrozenModel):
    task_count: int = Field(ge=0)
    valid_count: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    macro_success_rate: float = Field(ge=0.0, le=1.0)
    success_rate_by_difficulty: dict[str, float]
    success_rate_by_app: dict[str, float]
    error_rate: float = Field(ge=0.0, le=1.0)
    error_rate_by_reason: dict[str, float]
    stopped_rate: float = Field(ge=0.0, le=1.0)
    median_wall_ms: float = Field(ge=0.0)
    mean_wall_ms: float = Field(ge=0.0)
    p95_wall_ms: float = Field(ge=0.0)
    median_steps_to_success: float = Field(ge=0.0)
    mean_retries: float = Field(ge=0.0)
    mean_replans: float = Field(ge=0.0)


def _rate(items: Sequence[TaskOutcome], status: str) -> float:
    return sum(outcome.status == status for outcome in items) / len(items) if items else 0.0


def _group_success_rates(
    outcomes: Sequence[TaskOutcome],
    *,
    attribute: Literal["app", "difficulty"],
) -> dict[str, float]:
    groups: dict[str, list[TaskOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[str(getattr(outcome, attribute))].append(outcome)
    return {
        key: _rate(tuple(values), "succeeded")
        for key, values in sorted(groups.items())
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def calculate_suite_metrics(outcomes: Iterable[TaskOutcome]) -> SuiteMetrics:
    all_outcomes = tuple(outcomes)
    valid = tuple(outcome for outcome in all_outcomes if outcome.status != "invalid")
    successes = tuple(outcome for outcome in valid if outcome.status == "succeeded")
    by_app = _group_success_rates(valid, attribute="app")
    failures = Counter(
        outcome.reason_code
        for outcome in valid
        if outcome.status == "failed" and outcome.reason_code is not None
    )
    valid_count = len(valid)
    return SuiteMetrics(
        task_count=len(all_outcomes),
        valid_count=valid_count,
        success_rate=_rate(valid, "succeeded"),
        macro_success_rate=statistics.fmean(by_app.values()) if by_app else 0.0,
        success_rate_by_difficulty=_group_success_rates(valid, attribute="difficulty"),
        success_rate_by_app=by_app,
        error_rate=_rate(valid, "failed"),
        error_rate_by_reason={
            str(reason): count / valid_count
            for reason, count in sorted(failures.items())
        }
        if valid_count
        else {},
        stopped_rate=_rate(valid, "stopped"),
        median_wall_ms=statistics.median(outcome.wall_ms for outcome in valid)
        if valid
        else 0.0,
        mean_wall_ms=statistics.fmean(outcome.wall_ms for outcome in valid)
        if valid
        else 0.0,
        p95_wall_ms=_percentile(tuple(outcome.wall_ms for outcome in valid), 0.95),
        median_steps_to_success=statistics.median(
            outcome.step_count for outcome in successes
        )
        if successes
        else 0.0,
        mean_retries=statistics.fmean(outcome.retry_count for outcome in valid)
        if valid
        else 0.0,
        mean_replans=statistics.fmean(outcome.replan_count for outcome in valid)
        if valid
        else 0.0,
    )


def outcome_from_run(
    task: EvaluationTask,
    resolution: tuple[int, int],
    run: AgentRunResult,
    timing: TimingBreakdown,
) -> TaskOutcome:
    contains_dry_run = any(result.status == "dry_run" for result in run.results)
    status: Literal["succeeded", "failed", "stopped", "invalid"] = (
        "invalid" if contains_dry_run else run.status
    )
    retry_count = (
        sum(max(0, step.attempts - 1) for step in run.progress.steps)
        if run.progress is not None
        else 0
    )
    verification_pass_rate = (
        sum(result.passed for result in run.verifications) / len(run.verifications)
        if run.verifications
        else 0.0
    )
    return TaskOutcome(
        task_id=task.id,
        app=task.app,
        difficulty=task.difficulty,
        resolution=resolution,
        status=status,
        reason_code=None if contains_dry_run else run.reason_code,
        invalid_reason="dry_run_result" if contains_dry_run else None,
        step_count=len(run.results),
        retry_count=retry_count,
        replan_count=run.progress.replan_count if run.progress is not None else 0,
        verification_pass_rate=verification_pass_rate,
        wall_ms=timing.wall_ms,
        planner_ms=timing.planner_ms,
        perception_ms=timing.perception_ms,
        execution_ms=timing.execution_ms,
    )


__all__ = [
    "SuiteMetrics",
    "TaskOutcome",
    "TimingBreakdown",
    "calculate_suite_metrics",
    "outcome_from_run",
]
