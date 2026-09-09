from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from gui_agent.evaluation.suite import (
    EvaluationTask,
    EvaluationTaskSuite,
    load_task_suite,
    run_reference_solution,
)

SUITE_PATH = Path("configs/week7_task_suite.json")


def _task_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "valid-task",
        "app": "browser",
        "difficulty": "easy",
        "instruction": "Open the browser application.",
        "success_criteria": "The screen displays 'Search query'.",
        "initial_state": {"active_app": "files"},
        "fault_profile": "none",
        "max_steps": 2,
        "reference_actions": (
            {"kind": "click", "x": 141, "y": 47},
            {"kind": "finish", "success": True, "summary": "Browser opened"},
        ),
        "tags": ("navigation",),
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"app": "calculator"}, "app"),
        ({"difficulty": "expert"}, "difficulty"),
        ({"max_steps": 21}, "less than or equal to 20"),
        ({"success_criteria": "Search query is visible."}, "quoted text"),
        ({"reference_actions": ()}, "at least 1"),
    ],
)
def test_task_schema_rejects_invalid_values(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        EvaluationTask.model_validate(_task_payload(**override))


def test_suite_rejects_duplicate_task_ids() -> None:
    task = _task_payload()
    with pytest.raises(ValidationError, match="unique"):
        EvaluationTaskSuite.model_validate(
            {"schema_version": 1, "kind": "gui-agent-week7-task-suite", "tasks": (task, task)}
        )


def test_bundled_suite_has_the_required_application_and_difficulty_distribution() -> None:
    tasks = load_task_suite(SUITE_PATH)

    assert len(tasks) == 20
    assert Counter(task.app for task in tasks) == {
        "browser": 4,
        "files": 4,
        "messages": 4,
        "settings": 4,
        "editor": 4,
    }
    assert Counter(task.difficulty for task in tasks) == {
        "easy": 8,
        "medium": 7,
        "hard": 5,
    }
    assert len({task.id for task in tasks}) == 20
    assert all("'" in task.success_criteria for task in tasks)
    assert all(len(task.reference_actions) <= task.max_steps for task in tasks)


def test_every_bundled_task_has_a_working_oracle_solution(tmp_path: Path) -> None:
    tasks = load_task_suite(SUITE_PATH)

    outcomes = [
        run_reference_solution(task, root=tmp_path / task.id)
        for task in tasks
    ]

    assert all(outcome.succeeded for outcome in outcomes), [
        outcome for outcome in outcomes if not outcome.succeeded
    ]
    assert all(
        outcome.action_count <= task.max_steps
        for outcome, task in zip(outcomes, tasks, strict=True)
    )
