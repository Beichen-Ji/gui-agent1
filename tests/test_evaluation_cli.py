import json
from pathlib import Path

import pytest

from gui_agent.evaluation.cli import (
    EvaluationConditionSet,
    expand_run_matrix,
    load_condition_set,
    main,
)
from gui_agent.evaluation.suite import load_task_suite

SUITE_PATH = Path("configs/week7_task_suite.json")
CONDITIONS_PATH = Path("configs/week7_conditions.json")


def test_bundled_condition_matrix_expands_to_120_unique_runs() -> None:
    tasks = load_task_suite(SUITE_PATH)
    conditions = load_condition_set(CONDITIONS_PATH)

    matrix = expand_run_matrix(tasks, conditions)

    assert len(matrix) == 120
    assert len({item.id for item in matrix}) == 120
    assert len({item.condition.id for item in matrix}) == 6
    assert round(sum(item.estimated_seconds for item in matrix) / 60) == 110
    assert all(item.task.id in item.id for item in matrix)


def test_condition_set_rejects_duplicate_effective_conditions() -> None:
    raw = json.loads(CONDITIONS_PATH.read_text(encoding="utf-8"))
    raw["conditions"].append(dict(raw["conditions"][0], name="duplicate-main"))

    with pytest.raises(ValueError, match="duplicate"):
        EvaluationConditionSet.model_validate_json(json.dumps(raw))


def test_dry_run_plan_prints_budget_without_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "must-not-exist"

    assert main(
        [
            "--suite",
            str(SUITE_PATH),
            "--conditions",
            str(CONDITIONS_PATH),
            "--output",
            str(output),
            "--provider",
            "qwen",
            "--dry-run-plan",
        ]
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["run_count"] == 120
    assert summary["condition_count"] == 6
    assert summary["estimated_minutes"] == 110
    assert summary["executed"] is False
    assert not output.exists()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--suite", "missing.json", "--conditions", "missing.json", "--output", "out"],
        [
            "--suite",
            str(SUITE_PATH),
            "--conditions",
            str(CONDITIONS_PATH),
            "--output",
            "out",
            "--provider",
            "remote",
        ],
    ],
)
def test_evaluation_cli_rejects_missing_or_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as captured:
        main(argv)
    assert captured.value.code == 2
