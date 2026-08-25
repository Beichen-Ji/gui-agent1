import pytest

import gui_agent.control.controller as controller_module
from gui_agent.agent.executor import ActionExecutionError, ActionExecutor
from gui_agent.agent.types import (
    AgentAction,
    ClickAction,
    DragAction,
    FinishAction,
    HotkeyAction,
    ScrollAction,
    TypeTextAction,
    WaitAction,
)
from gui_agent.control.controller import DesktopController
from gui_agent.types import Point, ScreenRegion

BOUNDS = ScreenRegion(left=-100, top=-50, width=300, height=200)


class RecordingDesktop:
    def __init__(self, *, click_error: Exception | None = None) -> None:
        self.click_error = click_error
        self.calls: list[tuple[object, ...]] = []

    def move_to(self, point: Point, *, duration: float) -> None:
        self.calls.append(("move_to", point, duration))

    def click(self, point: Point, *, button: str, clicks: int) -> None:
        if self.click_error is not None:
            raise self.click_error
        self.calls.append(("click", point, button, clicks))

    def type_text(self, text: str, *, interval: float) -> None:
        self.calls.append(("type_text", text, interval))

    def hotkey(self, *keys: str) -> None:
        self.calls.append(("hotkey", keys))

    def scroll(self, clicks: int, *, point: Point | None) -> None:
        self.calls.append(("scroll", clicks, point))

    def drag_to(
        self,
        start: Point,
        end: Point,
        *,
        duration: float,
        button: str,
    ) -> None:
        self.calls.append(("drag_to", start, end, duration, button))


def live_controller(desktop: RecordingDesktop) -> DesktopController:
    return DesktopController(
        dry_run=False,
        backend=desktop,
        bounds_provider=lambda: BOUNDS,
    )


def test_executor_maps_every_schema_action_to_the_existing_controller() -> None:
    desktop = RecordingDesktop()
    waits: list[float] = []
    executor = ActionExecutor(live_controller(desktop), clock=waits.append)
    actions: tuple[AgentAction, ...] = (
        ClickAction(x=-10, y=20, button="right", clicks=2),
        TypeTextAction(text="week4 test message"),
        HotkeyAction(keys=("CTRL", "s")),
        ScrollAction(clicks=-3, x=10, y=30),
        DragAction(start_x=0, start_y=0, end_x=20, end_y=30, duration=0.75),
        WaitAction(seconds=0.25),
        FinishAction(success=True, summary="Task completed"),
    )

    results = tuple(
        executor.execute(action, step_index=index) for index, action in enumerate(actions)
    )

    assert [result.status for result in results] == ["executed"] * 7
    assert desktop.calls == [
        ("click", Point(-10, 20), "right", 2),
        ("type_text", "week4 test message", 0.02),
        ("hotkey", ("ctrl", "s")),
        ("scroll", -3, Point(10, 30)),
        ("drag_to", Point(0, 0), Point(20, 30), 0.75, "left"),
    ]
    assert waits == [0.25]
    assert len(executor.controller.history) == 5


def test_dry_run_executor_never_constructs_pyautogui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_adapter(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not construct PyAutoGUIAdapter")

    monkeypatch.setattr(controller_module, "PyAutoGUIAdapter", unexpected_adapter)
    controller = DesktopController(dry_run=True, bounds_provider=lambda: BOUNDS)

    result = ActionExecutor(controller).execute(
        ClickAction(x=0, y=0),
        step_index=0,
    )

    assert result.status == "dry_run"
    assert len(controller.history) == 1


def test_executor_result_message_does_not_repeat_typed_text() -> None:
    secret_text = "potential-password-value"
    executor = ActionExecutor(live_controller(RecordingDesktop()))

    result = executor.execute(TypeTextAction(text=secret_text), step_index=0)

    assert secret_text not in result.message


def test_executor_wraps_controller_errors_and_preserves_cause() -> None:
    original = OSError("desktop input failed")
    executor = ActionExecutor(live_controller(RecordingDesktop(click_error=original)))

    with pytest.raises(ActionExecutionError, match="click") as captured:
        executor.execute(ClickAction(x=0, y=0), step_index=0)

    assert captured.value.__cause__ is original


def test_executor_rejects_partial_scroll_coordinates_even_if_policy_is_bypassed() -> None:
    executor = ActionExecutor(live_controller(RecordingDesktop()))

    with pytest.raises(ActionExecutionError, match="both x and y"):
        executor.execute(ScrollAction(clicks=1, x=10), step_index=0)


def test_executor_uses_injected_clock_for_wait_errors() -> None:
    original = RuntimeError("clock unavailable")

    def failing_clock(_seconds: float) -> None:
        raise original

    executor = ActionExecutor(live_controller(RecordingDesktop()), clock=failing_clock)

    with pytest.raises(ActionExecutionError, match="wait") as captured:
        executor.execute(WaitAction(seconds=0.1), step_index=0)

    assert captured.value.__cause__ is original
