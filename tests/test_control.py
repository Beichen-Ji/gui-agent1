from dataclasses import FrozenInstanceError
from datetime import timedelta
from typing import cast

import pytest

import gui_agent.control as control
import gui_agent.control.controller as controller_module
from gui_agent.control.controller import (
    ActionRecord,
    DesktopBackend,
    DesktopController,
    PyAutoGUIAdapter,
)
from gui_agent.types import Point, ScreenRegion

from .conftest import FakeMSS


class FakeDesktop:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def move_to(self, point: Point, *, duration: float) -> None:
        self.calls.append(("move_to", point, duration))

    def click(self, point: Point, *, button: str, clicks: int) -> None:
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


class FakePyAutoGUI:
    def __init__(self) -> None:
        self.FAILSAFE = False
        self.PAUSE = -1.0
        self.calls: list[tuple[object, ...]] = []

    def moveTo(self, x: int, y: int, *, duration: float | None = None) -> None:  # noqa: N802
        if duration is None:
            self.calls.append(("moveTo", x, y))
        else:
            self.calls.append(("moveTo", x, y, duration))

    def click(
        self,
        x: int,
        y: int,
        *,
        button: str,
        clicks: int,
    ) -> None:
        self.calls.append(("click", x, y, button, clicks))

    def write(self, text: str, *, interval: float) -> None:
        self.calls.append(("write", text, interval))

    def hotkey(self, *keys: str) -> None:
        self.calls.append(("hotkey", keys))

    def scroll(self, clicks: int) -> None:
        self.calls.append(("scroll", clicks))

    def dragTo(  # noqa: N802
        self,
        x: int,
        y: int,
        *,
        duration: float,
        button: str,
    ) -> None:
        self.calls.append(("dragTo", x, y, duration, button))


BOUNDS = ScreenRegion(left=-100, top=-50, width=300, height=200)


def make_controller(
    desktop: FakeDesktop,
    *,
    dry_run: bool = False,
) -> DesktopController:
    return DesktopController(
        dry_run=dry_run,
        backend=desktop,
        bounds_provider=lambda: BOUNDS,
    )


def test_dry_run_click_records_action_without_touching_backend() -> None:
    fake_desktop = FakeDesktop()
    controller = DesktopController(
        dry_run=True,
        backend=fake_desktop,
        bounds_provider=lambda: ScreenRegion(-100, 0, 300, 200),
    )

    record = controller.click(Point(10, 20))

    assert record.name == "click"
    assert record.parameters == (
        ("button", "left"),
        ("clicks", 1),
        ("point", Point(10, 20)),
    )
    assert fake_desktop.calls == []


def test_controller_defaults_to_dry_run_and_does_not_construct_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_adapter(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("PyAutoGUIAdapter must not be constructed during dry-run")

    monkeypatch.setattr(controller_module, "PyAutoGUIAdapter", unexpected_adapter)
    controller = DesktopController(bounds_provider=lambda: BOUNDS)

    controller.click(Point(0, 0))

    assert controller.dry_run is True


def test_action_record_is_immutable_utc_and_history_is_a_tuple() -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    record = controller.click(Point(0, 0))

    assert isinstance(record, ActionRecord)
    assert record.created_at.utcoffset() == timedelta(0)
    assert controller.history == (record,)
    with pytest.raises(FrozenInstanceError):
        record.name = "changed"  # type: ignore[misc]


def test_live_move_to_records_and_calls_exact_arguments() -> None:
    desktop = FakeDesktop()
    controller = make_controller(desktop)

    record = controller.move_to(Point(-20, 30), duration=0.4)

    assert record.parameters == (("duration", 0.4), ("point", Point(-20, 30)))
    assert desktop.calls == [("move_to", Point(-20, 30), 0.4)]


def test_live_click_accepts_negative_origin_boundary() -> None:
    desktop = FakeDesktop()
    controller = make_controller(desktop)

    controller.click(Point(-100, -50), button="middle", clicks=3)

    assert desktop.calls == [("click", Point(-100, -50), "middle", 3)]


def test_double_and_right_click_delegate_with_safe_literal_parameters() -> None:
    desktop = FakeDesktop()
    controller = make_controller(desktop)

    double_record = controller.double_click(Point(1, 2))
    right_record = controller.right_click(Point(3, 4))

    assert double_record.parameters == (
        ("button", "left"),
        ("clicks", 2),
        ("point", Point(1, 2)),
    )
    assert right_record.parameters == (
        ("button", "right"),
        ("clicks", 1),
        ("point", Point(3, 4)),
    )
    assert desktop.calls == [
        ("click", Point(1, 2), "left", 2),
        ("click", Point(3, 4), "right", 1),
    ]


def test_live_type_text_records_and_calls_exact_arguments() -> None:
    desktop = FakeDesktop()
    controller = make_controller(desktop)

    record = controller.type_text("hello 世界", interval=0.03)

    assert record.parameters == (("interval", 0.03), ("text", "hello 世界"))
    assert desktop.calls == [("type_text", "hello 世界", 0.03)]


def test_live_hotkey_normalizes_and_calls_exact_arguments() -> None:
    desktop = FakeDesktop()
    controller = make_controller(desktop)

    record = controller.hotkey("CTRL", "Shift", "s")

    assert record.parameters == (("keys", ("ctrl", "shift", "s")),)
    assert desktop.calls == [("hotkey", ("ctrl", "shift", "s"))]


def test_live_scroll_records_optional_point_and_calls_exact_arguments() -> None:
    desktop = FakeDesktop()
    controller = make_controller(desktop)

    with_point = controller.scroll(-4, point=Point(10, 20))
    without_point = controller.scroll(2)

    assert with_point.parameters == (("clicks", -4), ("point", Point(10, 20)))
    assert without_point.parameters == (("clicks", 2), ("point", None))
    assert desktop.calls == [
        ("scroll", -4, Point(10, 20)),
        ("scroll", 2, None),
    ]


def test_live_drag_records_and_calls_exact_arguments() -> None:
    desktop = FakeDesktop()
    controller = make_controller(desktop)

    record = controller.drag_to(
        Point(-10, 5),
        Point(100, 120),
        duration=0.75,
        button="right",
    )

    assert record.parameters == (
        ("button", "right"),
        ("duration", 0.75),
        ("end", Point(100, 120)),
        ("start", Point(-10, 5)),
    )
    assert desktop.calls == [
        ("drag_to", Point(-10, 5), Point(100, 120), 0.75, "right")
    ]


@pytest.mark.parametrize(
    "point",
    [Point(-101, 0), Point(200, 0), Point(0, -51), Point(0, 150)],
)
def test_controller_rejects_each_half_open_boundary(point: Point) -> None:
    desktop = FakeDesktop()
    controller = make_controller(desktop)

    with pytest.raises(ValueError, match="outside the virtual desktop"):
        controller.click(point)

    assert controller.history == ()
    assert desktop.calls == []


@pytest.mark.parametrize("point", [Point(-100, -50), Point(199, 149)])
def test_controller_accepts_lower_and_last_inclusive_pixels(point: Point) -> None:
    desktop = FakeDesktop()

    make_controller(desktop).move_to(point, duration=0)

    assert desktop.calls == [("move_to", point, 0.0)]


@pytest.mark.parametrize("button", ["", "primary", "LEFT"])
def test_click_rejects_unknown_button(button: str) -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="unsupported button"):
        controller.click(Point(0, 0), button=button)

    assert controller.history == ()


@pytest.mark.parametrize("clicks", [0, -1, 1.5, True])
def test_click_rejects_non_positive_or_non_integer_count(clicks: object) -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="clicks must be a positive integer"):
        controller.click(Point(0, 0), clicks=cast(int, clicks))


@pytest.mark.parametrize("duration", [-0.1, float("inf"), float("nan")])
def test_move_rejects_negative_or_non_finite_duration(duration: float) -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="duration must be finite and non-negative"):
        controller.move_to(Point(0, 0), duration=duration)


@pytest.mark.parametrize("duration", [-0.1, float("inf"), float("nan")])
def test_drag_rejects_negative_or_non_finite_duration(duration: float) -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="duration must be finite and non-negative"):
        controller.drag_to(Point(0, 0), Point(1, 1), duration=duration)


def test_drag_rejects_unknown_button_and_out_of_bounds_endpoint() -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="unsupported button"):
        controller.drag_to(Point(0, 0), Point(1, 1), button="primary")
    with pytest.raises(ValueError, match="outside the virtual desktop"):
        controller.drag_to(Point(0, 0), Point(200, 1))


@pytest.mark.parametrize("interval", [-0.1, float("inf"), float("nan")])
def test_type_text_rejects_negative_or_non_finite_interval(interval: float) -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="interval must be finite and non-negative"):
        controller.type_text("text", interval=interval)


def test_type_text_rejects_empty_text() -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="text must not be empty"):
        controller.type_text("")


@pytest.mark.parametrize("keys", [(), ("ctrl", "command"), ("ctrl+s",)])
def test_hotkey_rejects_empty_or_unknown_sequence(keys: tuple[str, ...]) -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="unsupported hotkey sequence"):
        controller.hotkey(*keys)


@pytest.mark.parametrize("clicks", [1.5, True])
def test_scroll_rejects_non_integer_clicks(clicks: object) -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="scroll clicks must be an integer"):
        controller.scroll(cast(int, clicks))


def test_scroll_rejects_out_of_bounds_optional_point() -> None:
    controller = make_controller(FakeDesktop(), dry_run=True)

    with pytest.raises(ValueError, match="outside the virtual desktop"):
        controller.scroll(1, point=Point(200, 0))


@pytest.mark.parametrize("pause", [-0.1, float("inf"), float("nan")])
def test_controller_rejects_negative_or_non_finite_pause(pause: float) -> None:
    with pytest.raises(ValueError, match="pause must be finite and non-negative"):
        DesktopController(pause=pause, bounds_provider=lambda: BOUNDS)


def test_live_controller_lazily_creates_adapter_with_configured_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desktop = FakeDesktop()
    constructed_with: list[float] = []

    def adapter_factory(pause: float) -> FakeDesktop:
        constructed_with.append(pause)
        return desktop

    monkeypatch.setattr(controller_module, "PyAutoGUIAdapter", adapter_factory)
    controller = DesktopController(
        dry_run=False,
        pause=0.35,
        bounds_provider=lambda: BOUNDS,
    )
    assert constructed_with == []

    controller.click(Point(0, 0))
    controller.click(Point(1, 1))

    assert constructed_with == [0.35]
    assert desktop.calls == [
        ("click", Point(0, 0), "left", 1),
        ("click", Point(1, 1), "left", 1),
    ]


def test_default_bounds_provider_reads_virtual_monitor_without_capture(
    fake_mss: FakeMSS,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(controller_module, "_mss_session", lambda: fake_mss)
    controller = DesktopController(dry_run=True)

    controller.click(Point(-3, 0))

    assert fake_mss.requests == []


def test_pyautogui_adapter_enables_failsafe_and_maps_every_operation() -> None:
    pyautogui = FakePyAutoGUI()
    adapter = PyAutoGUIAdapter(0.2, module=pyautogui)

    adapter.move_to(Point(1, 2), duration=0.3)
    adapter.click(Point(3, 4), button="right", clicks=2)
    adapter.type_text("hello", interval=0.04)
    adapter.hotkey("ctrl", "s")
    adapter.scroll(-2, point=Point(5, 6))
    adapter.scroll(1, point=None)
    adapter.drag_to(Point(7, 8), Point(9, 10), duration=0.6, button="left")

    assert pyautogui.FAILSAFE is True
    assert pyautogui.PAUSE == 0.2
    assert pyautogui.calls == [
        ("moveTo", 1, 2, 0.3),
        ("click", 3, 4, "right", 2),
        ("write", "hello", 0.04),
        ("hotkey", ("ctrl", "s")),
        ("moveTo", 5, 6),
        ("scroll", -2),
        ("scroll", 1),
        ("moveTo", 7, 8),
        ("dragTo", 9, 10, 0.6, "left"),
    ]


def test_control_public_api_and_backend_protocol() -> None:
    desktop = FakeDesktop()

    assert control.ActionRecord is ActionRecord
    assert control.DesktopBackend is DesktopBackend
    assert control.DesktopController is DesktopController
    assert control.PyAutoGUIAdapter is PyAutoGUIAdapter
    assert isinstance(desktop, DesktopBackend)
