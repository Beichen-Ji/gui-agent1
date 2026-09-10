"""Reproducible simulated evaluation suites, metrics, reports, and plots."""

from gui_agent.evaluation.metrics import (
    SuiteMetrics,
    TaskOutcome,
    TimingBreakdown,
    calculate_suite_metrics,
    outcome_from_run,
)
from gui_agent.evaluation.report import (
    EvaluationContext,
    EvaluationReport,
    build_evaluation_report,
    load_evaluation_report,
    write_evaluation_report,
)
from gui_agent.evaluation.runner import run_evaluation
from gui_agent.evaluation.suite import (
    Difficulty,
    EvaluationTask,
    EvaluationTaskSuite,
    ReferenceSolutionResult,
    build_reference_planner,
    load_task_suite,
    run_reference_solution,
)

__all__ = [
    "Difficulty",
    "EvaluationContext",
    "EvaluationReport",
    "EvaluationTask",
    "EvaluationTaskSuite",
    "ReferenceSolutionResult",
    "SuiteMetrics",
    "TaskOutcome",
    "TimingBreakdown",
    "build_evaluation_report",
    "build_reference_planner",
    "calculate_suite_metrics",
    "load_evaluation_report",
    "load_task_suite",
    "outcome_from_run",
    "run_evaluation",
    "run_reference_solution",
    "write_evaluation_report",
]
