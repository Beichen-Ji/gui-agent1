"""Render publication-ready figures from synthetic evaluation reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from gui_agent.evaluation.metrics import TaskOutcome
from gui_agent.evaluation.report import EvaluationReport, load_evaluation_report

_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
_HATCHES = ("//", "\\\\", "xx", "..", "++", "oo")
_NOTE = "Synthetic simulated desktop (not a real desktop) — n=20 per condition"
_DIFFICULTIES = ("easy", "medium", "hard")
_APPS = ("browser", "files", "messages", "editor", "settings")


def _require_reports(reports: Sequence[EvaluationReport]) -> None:
    if not reports:
        raise ValueError("at least one evaluation report is required")


def _valid_outcomes(report: EvaluationReport) -> tuple[TaskOutcome, ...]:
    return tuple(outcome for outcome in report.outcomes if outcome.status != "invalid")


def _rate(outcomes: Sequence[TaskOutcome], *, key: str, value: str) -> float:
    selected = tuple(outcome for outcome in outcomes if str(getattr(outcome, key)) == value)
    if not selected:
        return 0.0
    return 100.0 * sum(outcome.status == "succeeded" for outcome in selected) / len(selected)


def _main_ocr_reports(reports: Sequence[EvaluationReport]) -> tuple[EvaluationReport, ...]:
    selected = tuple(
        sorted(
            (
                report
                for report in reports
                if report.observation_mode == "ocr"
                and report.prompt_profile == "week4-baseline"
                and report.adapter_label is None
            ),
            key=lambda report: report.outcomes[0].resolution if report.outcomes else (0, 0),
        )
    )
    if not selected:
        raise ValueError("no baseline OCR reports are available")
    return selected


def _resolution(report: EvaluationReport) -> tuple[int, int]:
    outcomes = _valid_outcomes(report)
    if not outcomes:
        raise ValueError("evaluation report has no valid outcomes")
    resolutions = {outcome.resolution for outcome in outcomes}
    if len(resolutions) != 1:
        raise ValueError("each evaluation report must contain one resolution")
    return next(iter(resolutions))


def _condition_label(report: EvaluationReport) -> str:
    width, height = _resolution(report)
    if report.adapter_label is not None:
        return "Week 5 adapter"
    if report.observation_mode == "oracle":
        return "Oracle"
    if report.prompt_profile == "week5-grounded":
        return "Grounded prompt"
    return f"Baseline {width}x{height}"


def _finish(figure: Figure, title: str) -> Figure:
    figure.suptitle(f"{title}\n{_NOTE}", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return figure


def plot_success_by_difficulty(reports: tuple[EvaluationReport, ...]) -> Figure:
    """Plot baseline OCR success rates by difficulty and resolution."""
    _require_reports(reports)
    selected = _main_ocr_reports(reports)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    width = 0.8 / len(selected)
    positions = np.arange(len(_DIFFICULTIES), dtype=float)
    for index, report in enumerate(selected):
        outcomes = _valid_outcomes(report)
        values = [_rate(outcomes, key="difficulty", value=item) for item in _DIFFICULTIES]
        offset = (index - (len(selected) - 1) / 2) * width
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=f"{_resolution(report)[0]}x{_resolution(report)[1]}",
            color=_PALETTE[index % len(_PALETTE)],
            hatch=_HATCHES[index % len(_HATCHES)],
            edgecolor="black",
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=8)
    axis.set_xticks(positions, tuple(item.title() for item in _DIFFICULTIES))
    axis.set_xlabel("Difficulty")
    axis.set_ylabel("Success rate (%)")
    axis.set_ylim(0, 100)
    axis.legend(title="Resolution")
    return _finish(figure, "Success rate by difficulty and resolution")


def plot_success_by_app(reports: tuple[EvaluationReport, ...]) -> Figure:
    """Plot baseline OCR success rates by application and resolution."""
    _require_reports(reports)
    selected = _main_ocr_reports(reports)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / len(selected)
    positions = np.arange(len(_APPS), dtype=float)
    for index, report in enumerate(selected):
        outcomes = _valid_outcomes(report)
        values = [_rate(outcomes, key="app", value=item) for item in _APPS]
        offset = (index - (len(selected) - 1) / 2) * width
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=f"{_resolution(report)[0]}x{_resolution(report)[1]}",
            color=_PALETTE[index % len(_PALETTE)],
            hatch=_HATCHES[index % len(_HATCHES)],
            edgecolor="black",
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=7)
    axis.set_xticks(positions, tuple(item.title() for item in _APPS))
    axis.set_xlabel("Application")
    axis.set_ylabel("Success rate (%)")
    axis.set_ylim(0, 100)
    axis.legend(title="Resolution")
    return _finish(figure, "Success rate by application and resolution")


def plot_success_and_latency_by_resolution(
    reports: tuple[EvaluationReport, ...],
) -> Figure:
    """Plot success rate and median wall time across baseline resolutions."""
    _require_reports(reports)
    selected = _main_ocr_reports(reports)
    labels = [f"{_resolution(report)[0]}x{_resolution(report)[1]}" for report in selected]
    success = [100.0 * report.metrics.success_rate for report in selected]
    latency = [report.metrics.median_wall_ms for report in selected]
    figure, success_axis = plt.subplots(figsize=(9, 5.5))
    latency_axis = success_axis.twinx()
    first = success_axis.plot(
        labels,
        success,
        color=_PALETTE[0],
        marker="o",
        linestyle="-",
        label="Success rate",
    )
    second = latency_axis.plot(
        labels,
        latency,
        color=_PALETTE[1],
        marker="s",
        linestyle="--",
        label="Median wall time",
    )
    for label, value in zip(labels, success, strict=True):
        success_axis.annotate(
            f"{value:.1f}%",
            (label, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    success_axis.set_xlabel("Resolution")
    success_axis.set_ylabel("Success rate (%)")
    latency_axis.set_ylabel("Median wall time (ms)")
    success_axis.set_ylim(0, 100)
    success_axis.legend(first + second, ("Success rate", "Median wall time"), loc="best")
    return _finish(figure, "Success rate and latency by resolution")


def plot_error_distribution(reports: tuple[EvaluationReport, ...]) -> Figure:
    """Plot stacked failure-reason shares for valid tasks by condition."""
    _require_reports(reports)
    labels = [_condition_label(report) for report in reports]
    reasons = sorted(
        {
            outcome.reason_code
            for report in reports
            for outcome in _valid_outcomes(report)
            if outcome.status == "failed" and outcome.reason_code is not None
        }
    )
    if not reasons:
        raise ValueError("evaluation reports contain no categorized failures")
    figure, axis = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(reports), dtype=float)
    bottoms = np.zeros(len(reports), dtype=float)
    for index, reason in enumerate(reasons):
        values = []
        for report in reports:
            outcomes = _valid_outcomes(report)
            count = sum(
                outcome.status == "failed" and outcome.reason_code == reason
                for outcome in outcomes
            )
            values.append(100.0 * count / len(outcomes) if outcomes else 0.0)
        bars = axis.bar(
            positions,
            values,
            bottom=bottoms,
            label=str(reason).replace("_", " "),
            color=_PALETTE[index % len(_PALETTE)],
            hatch=_HATCHES[index % len(_HATCHES)],
            edgecolor="black",
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.1f}" if value else "" for value in values],
            label_type="center",
            fontsize=7,
        )
        bottoms += np.asarray(values)
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_xlabel("Condition")
    axis.set_ylabel("Share of valid tasks (%)")
    axis.legend(title="Failure reason", fontsize=8)
    return _finish(figure, "Failure reason distribution")


def plot_steps_by_difficulty(reports: tuple[EvaluationReport, ...]) -> Figure:
    """Plot step-count distributions for successful baseline OCR tasks."""
    _require_reports(reports)
    selected = _main_ocr_reports(reports)
    groups: dict[str, list[int]] = defaultdict(list)
    for report in selected:
        for outcome in _valid_outcomes(report):
            if outcome.status == "succeeded":
                groups[str(outcome.difficulty)].append(outcome.step_count)
    if not any(groups.values()):
        raise ValueError("evaluation reports contain no successful outcomes")
    data = [groups[item] if groups[item] else [0] for item in _DIFFICULTIES]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    boxes = axis.boxplot(
        data,
        tick_labels=tuple(item.title() for item in _DIFFICULTIES),
        patch_artist=True,
    )
    for index, box in enumerate(boxes["boxes"]):
        box.set_facecolor(_PALETTE[index])
        box.set_hatch(_HATCHES[index])
    axis.set_xlabel("Difficulty")
    axis.set_ylabel("Steps to success (count)")
    return _finish(figure, "Successful-task step distribution")


def plot_timing_breakdown(reports: tuple[EvaluationReport, ...]) -> Figure:
    """Plot mean perception, planning, and execution time by condition."""
    _require_reports(reports)
    labels = [_condition_label(report) for report in reports]
    figure, axis = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(reports), dtype=float)
    bottoms = np.zeros(len(reports), dtype=float)
    fields = (
        ("perception_ms", "Perception"),
        ("planner_ms", "Planner"),
        ("execution_ms", "Execution"),
    )
    for index, (field, label) in enumerate(fields):
        values = [
            float(np.mean([getattr(outcome, field) for outcome in _valid_outcomes(report)]))
            for report in reports
        ]
        bars = axis.bar(
            positions,
            values,
            bottom=bottoms,
            label=label,
            color=_PALETTE[index],
            hatch=_HATCHES[index],
            edgecolor="black",
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.0f}" if value else "" for value in values],
            label_type="center",
            fontsize=7,
        )
        bottoms += np.asarray(values)
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_xlabel("Condition")
    axis.set_ylabel("Mean time per task (ms)")
    axis.legend(title="Stage")
    return _finish(figure, "Per-task timing breakdown")


def load_report_series(input_dir: Path) -> tuple[EvaluationReport, ...]:
    """Load all evaluation reports beneath a directory in stable order."""
    paths = tuple(sorted(input_dir.rglob("evaluation.json"))) if input_dir.is_dir() else ()
    if not paths:
        raise ValueError(f"no evaluation reports found under: {input_dir}")
    return tuple(load_evaluation_report(path) for path in paths)


_PLOTS: tuple[tuple[str, Callable[[tuple[EvaluationReport, ...]], Figure]], ...] = (
    ("success-by-difficulty", plot_success_by_difficulty),
    ("success-by-app", plot_success_by_app),
    ("success-latency-by-resolution", plot_success_and_latency_by_resolution),
    ("failure-reasons", plot_error_distribution),
    ("steps-by-difficulty", plot_steps_by_difficulty),
    ("timing-breakdown", plot_timing_breakdown),
)


def save_week7_figures(
    reports: tuple[EvaluationReport, ...],
    output_dir: Path,
) -> tuple[Path, ...]:
    """Save every Week 7 figure as PNG and SVG and return written paths."""
    _require_reports(reports)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, plotter in _PLOTS:
        figure = plotter(reports)
        try:
            for suffix in (".png", ".svg"):
                path = output_dir / f"{stem}{suffix}"
                figure.savefig(path, dpi=180 if suffix == ".png" else None, bbox_inches="tight")
                written.append(path)
        finally:
            plt.close(figure)
    return tuple(written)


__all__ = [
    "load_report_series",
    "plot_error_distribution",
    "plot_steps_by_difficulty",
    "plot_success_and_latency_by_resolution",
    "plot_success_by_app",
    "plot_success_by_difficulty",
    "plot_timing_breakdown",
    "save_week7_figures",
]
