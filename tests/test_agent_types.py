from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    ClickAction,
    TaskPlan,
    TaskStep,
    TypeTextAction,
)

ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)


def test_action_union_uses_kind_to_parse_valid_click_json() -> None:
    action = ACTION_ADAPTER.validate_json(
        '{"kind":"click","x":410,"y":220,"button":"right","clicks":2}'
    )

    assert action == ClickAction(x=410, y=220, button="right", clicks=2)


def test_action_union_rejects_unknown_action_kind() -> None:
    with pytest.raises(ValidationError, match="unknown_action"):
        ACTION_ADAPTER.validate_python({"kind": "unknown_action", "command": "rm -rf"})


@pytest.mark.parametrize("bad_x", [True, "410", 410.0])
def test_click_rejects_non_integer_coordinate_types(bad_x: Any) -> None:
    with pytest.raises(ValidationError):
        ClickAction.model_validate({"kind": "click", "x": bad_x, "y": 220})


def test_text_action_rejects_more_than_500_characters() -> None:
    with pytest.raises(ValidationError, match="500"):
        TypeTextAction(text="x" * 501)


@pytest.mark.parametrize("step_count", [0, 21])
def test_task_plan_rejects_step_counts_outside_safe_bounds(step_count: int) -> None:
    steps = tuple(
        TaskStep(id=f"step-{index}", description=f"Do step {index}")
        for index in range(step_count)
    )

    with pytest.raises(ValidationError):
        TaskPlan(goal="Open the browser", steps=steps)


def test_agent_decision_parses_action_and_forbids_extra_model_fields() -> None:
    payload = {
        "current_step_id": "step-1",
        "rationale_summary": "The browser icon is visible.",
        "action": {"kind": "click", "x": 20, "y": 30},
        "expected_outcome": "The browser opens.",
        "hidden_chain_of_thought": "must not be retained",
    }

    with pytest.raises(ValidationError, match="hidden_chain_of_thought"):
        AgentDecision.model_validate(payload)
