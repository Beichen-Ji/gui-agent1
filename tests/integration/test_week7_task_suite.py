import hashlib
from pathlib import Path

import pytest

from gui_agent.evaluation.report import EvaluationContext
from gui_agent.evaluation.runner import run_evaluation
from gui_agent.evaluation.suite import build_reference_planner, load_task_suite


@pytest.mark.integration
@pytest.mark.parametrize("resolution", [(1280, 720), (1920, 1080), (2560, 1440)])
def test_all_twenty_tasks_complete_through_the_fake_planner_pipeline(
    tmp_path: Path,
    resolution: tuple[int, int],
) -> None:
    suite_path = Path("configs/week7_task_suite.json")
    tasks = load_task_suite(suite_path)
    report = run_evaluation(
        tasks,
        context=EvaluationContext(
            suite_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
            conditions_sha256="0" * 64,
            git_revision="integration-test",
            model="fake-reference",
            prompt_profile="week4-baseline",
            ocr_profile="oracle",
            observation_mode="oracle",
            seed=7,
            environment={"runtime": "pytest"},
        ),
        resolution=resolution,
        planner_factory=lambda task, canvas: build_reference_planner(task, canvas=canvas),
        output_dir=tmp_path / f"week7-{resolution[0]}x{resolution[1]}",
    )

    assert len(report.outcomes) == 20
    assert report.metrics.success_rate == 1.0
    assert {outcome.status for outcome in report.outcomes} == {"succeeded"}
