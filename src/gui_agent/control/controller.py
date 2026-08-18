import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, Self, cast, runtime_checkable

from gui_agent.types import Point, ScreenRegion

VALID_BUTTONS = frozenset({"left", "middle", "right"})
VALID_KEYS = frozenset(
    {
        "ctrl",
        "shift",
        "alt",
        "win",
        "enter",
        "esc",
        "tab",
        "space",
        "backspace",
        "delete",
    }
    | {chr(code) for code in range(ord("a"), ord("z") + 1)}
    | {str(number) for number in range(10)}
    | {f"f{number}" for number in range(1, 13)}
)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    name: str
    parameters: tuple[tuple[str, object], ...]
    created_at: datetime


@runtime_checkable
class DesktopBackend(Protocol):
    def move_to(self, point: Point, *, duration: float) -> None: ...

    def click(self, point: Point, *, button: str, clicks: int) -> None: ...

    def type_text(self, text: str, *, interval: float) -> None: ...

    def hotkey(self, *keys: str) -> None: ...

    def scroll(self, clicks: int, *, point: Point | None) -> None: ...

    def drag_to(
        self,
        start: Point,
        end: Point,
        *,
        duration: float,
        button: str,
    ) -> None: ...


class _PyAutoGUIModule(Protocol):
    FAILSAFE: bool
    PAUSE: float

    def moveTo(  # noqa: N802
        self,
        x: int,
        y: int,
        *,
        duration: float = 0.0,
    ) -> None: ...

    def click(
        self,
        x: int,
        y: int,
        *,
        button: str,
        clicks: int,
    ) -> None: ...

    def write(self, text: str, *, interval: float) -> None: ...

    def hotkey(self, *keys: str) -> None: ...

    def scroll(self, clicks: int) -> None: ...

    def dragTo(  # noqa: N802
        self,
        x: int,
        y: int,
        *,
        duration: float,
        button: str,
    ) -> None: ...


class _MSSSession(Protocol):
    monitors: Sequence[Mapping[str, int]]

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None: ...


def _duration(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def _mss_session() -> _MSSSession:
    from mss import mss

    return cast(_MSSSession, mss())


def _default_virtual_bounds() -> ScreenRegion:
    with _mss_session() as session:
        virtual = session.monitors[0]
        return ScreenRegion(
            left=virtual["left"],
            top=virtual["top"],
            width=virtual["width"],
            height=virtual["height"],
        )


class PyAutoGUIAdapter:
    def __init__(
        self,
        pause: float = 0.1,
        *,
        module: _PyAutoGUIModule | None = None,
    ) -> None:
        pause = _duration(pause, "pause")
        if module is None:
            import pyautogui

            module = cast(_PyAutoGUIModule, pyautogui)
        module.FAILSAFE = True
        module.PAUSE = pause
        self._module = module

    def move_to(self, point: Point, *, duration: float) -> None:
        self._module.moveTo(point.x, point.y, duration=duration)

    def click(self, point: Point, *, button: str, clicks: int) -> None:
        self._module.click(point.x, point.y, button=button, clicks=clicks)

    def type_text(self, text: str, *, interval: float) -> None:
        self._module.write(text, interval=interval)

    def hotkey(self, *keys: str) -> None:
        self._module.hotkey(*keys)

    def scroll(self, clicks: int, *, point: Point | None) -> None:
        if point is not None:
            self._module.moveTo(point.x, point.y)
        self._module.scroll(clicks)

    def drag_to(
        self,
        start: Point,
        end: Point,
        *,
        duration: float,
        button: str,
    ) -> None:
        self._module.moveTo(start.x, start.y)
        self._module.dragTo(end.x, end.y, duration=duration, button=button)


class DesktopController:
    def __init__(
        self,
        *,
        dry_run: bool = True,
        pause: float = 0.1,
        backend: DesktopBackend | None = None,
        bounds_provider: Callable[[], ScreenRegion] | None = None,
    ) -> None:
        self.dry_run = dry_run
        self._pause = _duration(pause, "pause")
        self._backend = backend
        self._bounds_provider = bounds_provider or _default_virtual_bounds
        self._history: list[ActionRecord] = []

    @property
    def history(self) -> tuple[ActionRecord, ...]:
        return tuple(self._history)

    def _point(self, point: Point) -> None:
        if not self._bounds_provider().contains(point):
            raise ValueError(f"point is outside the virtual desktop: {point}")

    def _backend_instance(self) -> DesktopBackend:
        if self._backend is None:
            self._backend = PyAutoGUIAdapter(self._pause)
        return self._backend

    def _dispatch(
        self,
        name: str,
        parameters: tuple[tuple[str, object], ...],
        operation: Callable[[DesktopBackend], None],
    ) -> ActionRecord:
        record = ActionRecord(name, parameters, datetime.now(UTC))
        self._history.append(record)
        if not self.dry_run:
            operation(self._backend_instance())
        return record

    def move_to(self, point: Point, *, duration: float = 0.2) -> ActionRecord:
        self._point(point)
        duration = _duration(duration, "duration")
        return self._dispatch(
            "move_to",
            (("duration", duration), ("point", point)),
            lambda backend: backend.move_to(point, duration=duration),
        )

    def click(
        self,
        point: Point,
        *,
        button: str = "left",
        clicks: int = 1,
    ) -> ActionRecord:
        self._point(point)
        if button not in VALID_BUTTONS:
            raise ValueError(f"unsupported button: {button}")
        if isinstance(clicks, bool) or not isinstance(clicks, int) or clicks < 1:
            raise ValueError("clicks must be a positive integer")
        return self._dispatch(
            "click",
            (("button", button), ("clicks", clicks), ("point", point)),
            lambda backend: backend.click(point, button=button, clicks=clicks),
        )

    def double_click(self, point: Point) -> ActionRecord:
        return self.click(point, clicks=2)

    def right_click(self, point: Point) -> ActionRecord:
        return self.click(point, button="right")

    def type_text(self, text: str, *, interval: float = 0.02) -> ActionRecord:
        if not text:
            raise ValueError("text must not be empty")
        interval = _duration(interval, "interval")
        return self._dispatch(
            "type_text",
            (("interval", interval), ("text", text)),
            lambda backend: backend.type_text(text, interval=interval),
        )

    def hotkey(self, *keys: str) -> ActionRecord:
        normalized = tuple(key.casefold() for key in keys)
        if not normalized or any(key not in VALID_KEYS for key in normalized):
            raise ValueError(f"unsupported hotkey sequence: {keys}")
        return self._dispatch(
            "hotkey",
            (("keys", normalized),),
            lambda backend: backend.hotkey(*normalized),
        )

    def scroll(self, clicks: int, *, point: Point | None = None) -> ActionRecord:
        if isinstance(clicks, bool) or not isinstance(clicks, int):
            raise ValueError("scroll clicks must be an integer")
        if point is not None:
            self._point(point)
        return self._dispatch(
            "scroll",
            (("clicks", clicks), ("point", point)),
            lambda backend: backend.scroll(clicks, point=point),
        )

    def drag_to(
        self,
        start: Point,
        end: Point,
        *,
        duration: float = 0.5,
        button: str = "left",
    ) -> ActionRecord:
        self._point(start)
        self._point(end)
        duration = _duration(duration, "duration")
        if button not in VALID_BUTTONS:
            raise ValueError(f"unsupported button: {button}")
        return self._dispatch(
            "drag_to",
            (
                ("button", button),
                ("duration", duration),
                ("end", end),
                ("start", start),
            ),
            lambda backend: backend.drag_to(
                start,
                end,
                duration=duration,
                button=button,
            ),
        )
