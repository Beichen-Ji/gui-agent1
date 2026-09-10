from pathlib import Path

import pytest

from gui_agent.evaluation.metrics import TaskOutcome
from gui_agent.evaluation.report import (
    EvaluationContext,
    build_evaluation_report,
    load_evaluation_report,
    write_evaluation_report,
)


def _context() -> EvaluationContext:
    return EvaluationContext(
        suite_sha256="a" * 64,
        conditions_sha256="b" * 64,
        git_revision="deadbeef",
        model="fake",
        prompt_profile="week4-baseline",
        ocr_profile="balanced",
        observation_mode="oracle",
        seed=7,
        environment={"python": "3.11", "torch": "test"},
    )


def _outcome(task_id: str, status: str) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        app="browser",
        difficulty="easy",
        resolution=(1280, 720),
        status=status,  # type: ignore[arg-type]
        reason_code=None,
        invalid_reason="dry_run_result" if status == "invalid" else None,
        step_count=1,
        retry_count=0,
        replan_count=0,
        verification_pass_rate=1.0,
        wall_ms=10,
        planner_ms=4,
        perception_ms=3,
        execution_ms=3,
    )


def test_report_json_is_byte_deterministic_regardless_of_outcome_order(tmp_path: Path) -> None:
    first = build_evaluation_report(
        _context(),
        (_outcome("b", "failed"), _outcome("a", "succeeded")),
    )
    second = build_evaluation_report(
        _context(),
        (_outcome("a", "succeeded"), _outcome("b", "failed")),
    )

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_evaluation_report(first, first_path)
    write_evaluation_report(second, second_path)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert load_evaluation_report(first_path) == first


def test_writer_refuses_non_json_and_unowned_overwrite(tmp_path: Path) -> None:
    report = build_evaluation_report(_context(), (_outcome("a", "succeeded"),))
    binary = tmp_path / "keep.bin"
    binary.write_bytes(b"keep")
    with pytest.raises(ValueError, match="JSON"):
        write_evaluation_report(report, binary, overwrite=True)
    assert binary.read_bytes() == b"keep"

    unowned = tmp_path / "keep.json"
    unowned.write_text('{"kind":"user-data"}', encoding="utf-8")
    with pytest.raises(ValueError, match="owned"):
        write_evaluation_report(report, unowned, overwrite=True)
    assert unowned.read_text(encoding="utf-8") == '{"kind":"user-data"}'


def test_writer_can_atomically_replace_an_owned_report(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    first = build_evaluation_report(_context(), (_outcome("a", "failed"),))
    second = build_evaluation_report(_context(), (_outcome("a", "succeeded"),))

    write_evaluation_report(first, path)
    write_evaluation_report(second, path, overwrite=True)

    assert load_evaluation_report(path) == second
