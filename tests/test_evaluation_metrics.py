from pathlib import Path

import pytest

from gui_agent.agent.loop import AgentRunResult
from gui_agent.agent.types import ClickAction, StepResult
from gui_agent.evaluation.metrics import (
    TaskOutcome,
    TimingBreakdown,
    calculate_suite_metrics,
    outcome_from_run,
)
from gui_agent.evaluation.suite import load_task_suite


def _outcome(
    task_id: str,
    *,
    app: str,
    status: str,
    wall_ms: float,
    reason_code: str | None = None,
    difficulty: str = "easy",
    steps: int = 2,
    retries: int = 0,
    replans: int = 0,
) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        app=app,  # type: ignore[arg-type]
        difficulty=difficulty,  # type: ignore[arg-type]
        resolution=(1280, 720),
        status=status,  # type: ignore[arg-type]
        reason_code=reason_code,  # type: ignore[arg-type]
        invalid_reason="dry_run_result" if status == "invalid" else None,
        step_count=steps,
        retry_count=retries,
        replan_count=replans,
        verification_pass_rate=0.5,
        wall_ms=wall_ms,
        planner_ms=wall_ms / 2,
        perception_ms=wall_ms / 4,
        execution_ms=wall_ms / 4,
    )


def test_metrics_exclude_invalid_runs_from_every_rate_denominator() -> None:
    outcomes = (
        _outcome("a", app="browser", status="succeeded", wall_ms=100, steps=2),
        _outcome(
            "b",
            app="browser",
            status="failed",
            reason_code="execution_error",
            wall_ms=300,
            difficulty="medium",
            retries=2,
        ),
        _outcome(
            "c",
            app="files",
            status="stopped",
            reason_code="repeated_action",
            wall_ms=500,
            difficulty="hard",
            replans=1,
        ),
        _outcome("d", app="files", status="invalid", wall_ms=999),
        _outcome("e", app="files", status="succeeded", wall_ms=200, steps=4),
    )

    metrics = calculate_suite_metrics(outcomes)

    assert metrics.task_count == 5
    assert metrics.valid_count == 4
    assert metrics.success_rate == pytest.approx(0.5)
    assert metrics.error_rate == pytest.approx(0.25)
    assert metrics.stopped_rate == pytest.approx(0.25)
    assert metrics.error_rate_by_reason == {"execution_error": 0.25}
    assert metrics.median_wall_ms == 250
    assert metrics.mean_wall_ms == 275
    assert metrics.p95_wall_ms == pytest.approx(470)
    assert metrics.median_steps_to_success == 3
    assert metrics.mean_retries == 0.5
    assert metrics.mean_replans == 0.25


def test_metrics_report_macro_and_micro_success_rates_separately() -> None:
    outcomes = (
        _outcome("a", app="browser", status="succeeded", wall_ms=10),
        _outcome("b", app="browser", status="succeeded", wall_ms=10),
        _outcome("c", app="files", status="failed", wall_ms=10),
    )

    metrics = calculate_suite_metrics(outcomes)

    assert metrics.success_rate == pytest.approx(2 / 3)
    assert metrics.success_rate_by_app == {"browser": 1.0, "files": 0.0}
    assert metrics.macro_success_rate == pytest.approx(0.5)


def test_dry_run_result_is_marked_invalid_instead_of_success() -> None:
    task = load_task_suite(Path("configs/week7_task_suite.json"))[0]
    action = ClickAction(x=10, y=10)
    run = AgentRunResult(
        goal=task.instruction,
        status="succeeded",
        message="preview",
        plan=None,
        observation=None,
        decisions=(),
        results=(StepResult(step_index=0, action=action, status="dry_run", message="preview"),),
    )

    outcome = outcome_from_run(
        task,
        (1280, 720),
        run,
        TimingBreakdown(wall_ms=1, planner_ms=0.4, perception_ms=0.3, execution_ms=0.3),
    )

    assert outcome.status == "invalid"
    assert outcome.invalid_reason == "dry_run_result"
