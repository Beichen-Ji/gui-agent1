from datetime import UTC, datetime
from typing import cast

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from gui_agent.agent.policy import (
    CONFIRMATION_PHRASE,
    ActionDeniedError,
    SafetyPolicy,
)
from gui_agent.agent.types import (
    AgentAction,
    ClickAction,
    DragAction,
    FinishAction,
    HotkeyAction,
    Observation,
    ScrollAction,
    TypeTextAction,
    WaitAction,
)
from gui_agent.types import Point, ScreenshotResult

ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)


def test_dry_run_authorizes_valid_action_without_prompting() -> None:
    prompts: list[str] = []

    def unexpected_prompt(prompt: str) -> str:
        prompts.append(prompt)
        return "unexpected"

    policy = SafetyPolicy(input_fn=unexpected_prompt)

    policy.authorize(
        ClickAction(x=-60, y=20),
        _observation(),
        expected_outcome="Browser opens",
    )

    assert prompts == []


@pytest.mark.parametrize(
    "action",
    [
        ClickAction(x=-61, y=20),
        ClickAction(x=0, y=60),
        DragAction(start_x=-60, start_y=20, end_x=0, end_y=60),
        ScrollAction(clicks=1, x=0, y=60),
    ],
)
def test_policy_rejects_coordinates_outside_current_observation(
    action: AgentAction,
) -> None:
    with pytest.raises(ActionDeniedError, match="outside the current observation"):
        SafetyPolicy().authorize(
            action,
            _observation(),
            expected_outcome="No unsafe pointer movement",
        )


def test_policy_rejects_scroll_with_only_one_coordinate() -> None:
    with pytest.raises(ActionDeniedError, match="both x and y"):
        SafetyPolicy().authorize(
            ScrollAction(clicks=1, x=-50),
            _observation(),
            expected_outcome="Scroll the panel",
        )


@pytest.mark.parametrize(
    "action",
    [
        HotkeyAction(keys=("ctrl", "command")),
        TypeTextAction.model_construct(text="x" * 501),
        TypeTextAction.model_construct(text="   "),
        WaitAction.model_construct(seconds=0.0),
        WaitAction.model_construct(seconds=-1.0),
    ],
)
def test_policy_rejects_actions_that_bypass_safe_runtime_limits(
    action: AgentAction,
) -> None:
    with pytest.raises(ActionDeniedError):
        SafetyPolicy().authorize(
            action,
            _observation(),
            expected_outcome="Remain safe",
        )


def test_action_schema_rejects_shell_and_file_delete_requests_before_policy() -> None:
    for payload in (
        {"kind": "shell", "command": "whoami"},
        {"kind": "delete_file", "path": "week4-demo.txt"},
    ):
        with pytest.raises(ValidationError):
            ACTION_ADAPTER.validate_python(payload)


def test_live_policy_requires_exact_confirmation_phrase() -> None:
    policy = SafetyPolicy(execute=True, input_fn=lambda _prompt: "yes")

    with pytest.raises(ActionDeniedError, match="not confirmed"):
        policy.authorize(
            ClickAction(x=-50, y=30),
            _observation(),
            expected_outcome="Browser opens",
        )


def test_live_confirmation_uses_escaped_truncated_text_preview() -> None:
    prompts: list[str] = []
    text = "line one\n" + "private-value-" * 20

    def confirm(prompt: str) -> str:
        prompts.append(prompt)
        return CONFIRMATION_PHRASE

    SafetyPolicy(execute=True, input_fn=confirm).authorize(
        TypeTextAction(text=text),
        _observation(),
        expected_outcome="Local testbed shows the message",
    )

    assert len(prompts) == 1
    assert "line one\\n" in prompts[0]
    assert text not in prompts[0]
    assert "Local testbed shows the message" in prompts[0]
    assert CONFIRMATION_PHRASE in prompts[0]


def test_finish_is_internal_and_never_prompts_for_desktop_confirmation() -> None:
    def unexpected_prompt(_prompt: str) -> str:
        raise AssertionError("finish must not request desktop confirmation")

    SafetyPolicy(execute=True, input_fn=unexpected_prompt).authorize(
        FinishAction(success=True, summary="Done"),
        _observation(),
        expected_outcome="Stop the loop",
    )


def test_policy_rejects_non_schema_objects_fail_closed() -> None:
    with pytest.raises(ActionDeniedError, match="unsupported action"):
        SafetyPolicy().authorize(
            cast(AgentAction, object()),
            _observation(),
            expected_outcome="Never execute unknown tools",
        )


def _observation() -> Observation:
    screenshot = ScreenshotResult(
        image=np.zeros((40, 60, 3), dtype=np.uint8),
        monitor_index=1,
        captured_at=datetime(2026, 8, 25, tzinfo=UTC),
        origin=Point(-60, 20),
    )
    return Observation(screenshot=screenshot, detections=(), step_index=0)
