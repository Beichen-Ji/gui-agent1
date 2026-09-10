from collections.abc import Callable
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from gui_agent.evaluation.metrics import TaskOutcome  # noqa: E402
from gui_agent.evaluation.plots import (  # noqa: E402
    load_report_series,
    plot_error_distribution,
    plot_steps_by_difficulty,
    plot_success_and_latency_by_resolution,
    plot_success_by_app,
    plot_success_by_difficulty,
    plot_timing_breakdown,
    save_week7_figures,
)
from gui_agent.evaluation.report import (  # noqa: E402
    EvaluationContext,
    EvaluationReport,
    build_evaluation_report,
    write_evaluation_report,
)

PLOTTERS: tuple[Callable[[tuple[EvaluationReport, ...]], Figure], ...] = (
    plot_success_by_difficulty,
    plot_success_by_app,
    plot_success_and_latency_by_resolution,
    plot_error_distribution,
    plot_steps_by_difficulty,
    plot_timing_breakdown,
)


def _report(
    *,
    resolution: tuple[int, int],
    observation_mode: str = "ocr",
    prompt_profile: str = "week4-baseline",
) -> EvaluationReport:
    context = EvaluationContext(
        suite_sha256="a" * 64,
        conditions_sha256="b" * 64,
        git_revision="deadbeef",
        model="fake",
        prompt_profile=prompt_profile,
        ocr_profile="balanced",
        observation_mode=observation_mode,  # type: ignore[arg-type]
        seed=7,
        environment={"python": "3.11"},
    )
    apps = ("browser", "files", "messages", "editor", "settings")
    difficulties = ("easy", "medium", "hard")
    outcomes = []
    for index in range(20):
        status = "succeeded" if index % 3 == 0 else "failed"
        outcomes.append(
            TaskOutcome(
                task_id=f"task-{index:02d}",
                app=apps[index % len(apps)],  # type: ignore[arg-type]
                difficulty=difficulties[index % len(difficulties)],  # type: ignore[arg-type]
                resolution=resolution,
                status=status,  # type: ignore[arg-type]
                reason_code=None if status == "succeeded" else "expected_text_missing",
                invalid_reason=None,
                step_count=index % 5 + 1,
                retry_count=index % 2,
                replan_count=index % 3,
                verification_pass_rate=0.75,
                wall_ms=1000.0 + index * 10,
                planner_ms=600.0 + index,
                perception_ms=250.0 + index,
                execution_ms=150.0 + index,
            )
        )
    return build_evaluation_report(context, tuple(outcomes))


@pytest.fixture
def reports() -> tuple[EvaluationReport, ...]:
    return (
        _report(resolution=(1280, 720)),
        _report(resolution=(1920, 1080)),
        _report(resolution=(2560, 1440)),
        _report(resolution=(1920, 1080), observation_mode="oracle"),
        _report(resolution=(1920, 1080), prompt_profile="week5-grounded"),
    )


@pytest.mark.parametrize("plotter", PLOTTERS)
def test_plotters_reject_empty_data(
    plotter: Callable[[tuple[EvaluationReport, ...]], Figure],
) -> None:
    with pytest.raises(ValueError, match="at least one evaluation report"):
        plotter(())


@pytest.mark.parametrize("plotter", PLOTTERS)
def test_plotters_include_sample_context_and_axis_units(
    plotter: Callable[[tuple[EvaluationReport, ...]], Figure],
    reports: tuple[EvaluationReport, ...],
) -> None:
    figure = plotter(reports)
    try:
        title_text = " ".join(text.get_text() for text in figure.texts)
        labels = [axis.get_xlabel() + axis.get_ylabel() for axis in figure.axes]
        assert "Synthetic simulated desktop" in title_text
        assert "n=20" in title_text
        assert any("%" in label or "ms" in label or "Steps" in label for label in labels)
    finally:
        matplotlib.pyplot.close(figure)


def test_save_week7_figures_writes_six_png_and_svg_pairs(
    tmp_path: Path,
    reports: tuple[EvaluationReport, ...],
) -> None:
    paths = save_week7_figures(reports, tmp_path / "figures")

    assert len(paths) == 12
    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert all(path.stat().st_size > 0 for path in paths)


def test_load_report_series_finds_nested_owned_reports(tmp_path: Path) -> None:
    first = _report(resolution=(1280, 720))
    second = _report(resolution=(1920, 1080), observation_mode="oracle")
    write_evaluation_report(first, tmp_path / "first" / "evaluation.json")
    write_evaluation_report(second, tmp_path / "second" / "evaluation.json")

    loaded = load_report_series(tmp_path)

    assert loaded == (first, second)


def test_load_report_series_rejects_directory_without_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no evaluation reports"):
        load_report_series(tmp_path)
