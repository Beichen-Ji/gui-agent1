import math
from collections.abc import Callable

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
from gui_agent.control.controller import VALID_BUTTONS, VALID_KEYS
from gui_agent.types import Point, ScreenRegion

CONFIRMATION_PHRASE = "EXECUTE ACTION"


class ActionDeniedError(RuntimeError):
    """A model-proposed action failed the local safety policy."""


def _escaped_preview(value: str, *, max_length: int) -> str:
    preview = repr(value[:max_length])
    return f"{preview}{'…' if len(value) > max_length else ''}"


class SafetyPolicy:
    """Validate untrusted model actions before they reach desktop control."""

    def __init__(
        self,
        *,
        execute: bool = False,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self._execute = execute
        self._input_fn = input_fn

    @property
    def execute(self) -> bool:
        return self._execute

    def authorize(
        self,
        action: AgentAction,
        observation: Observation,
        *,
        expected_outcome: str,
    ) -> None:
        description = self._validate(action, observation)
        if not isinstance(expected_outcome, str) or not expected_outcome.strip():
            raise ActionDeniedError("expected outcome must not be blank")
        if not self._execute or isinstance(action, FinishAction):
            return
        prompt = (
            "Live desktop action requires confirmation.\n"
            f"Action: {description}\n"
            f"Expected outcome: {_escaped_preview(expected_outcome, max_length=120)}\n"
            f"Type {CONFIRMATION_PHRASE!r} to continue: "
        )
        try:
            response = self._input_fn(prompt)
        except Exception as error:
            raise ActionDeniedError("live action confirmation could not be obtained") from error
        if response != CONFIRMATION_PHRASE:
            raise ActionDeniedError("live action was not confirmed")

    def _validate(self, action: AgentAction, observation: Observation) -> str:
        screenshot = observation.screenshot
        bounds = ScreenRegion(
            left=screenshot.origin.x,
            top=screenshot.origin.y,
            width=screenshot.width,
            height=screenshot.height,
        )
        if isinstance(action, ClickAction):
            point = self._point(action.x, action.y, bounds)
            if action.button not in VALID_BUTTONS:
                raise ActionDeniedError("click button is unsupported")
            if (
                isinstance(action.clicks, bool)
                or not isinstance(action.clicks, int)
                or not 1 <= action.clicks <= 2
            ):
                raise ActionDeniedError("click count is outside the safe range")
            return (
                f"click point=({point.x},{point.y}) "
                f"button={action.button} clicks={action.clicks}"
            )
        if isinstance(action, TypeTextAction):
            if (
                not isinstance(action.text, str)
                or not action.text.strip()
                or len(action.text) > 500
            ):
                raise ActionDeniedError("typed text is outside the safe limits")
            return f"type_text text={_escaped_preview(action.text, max_length=32)}"
        if isinstance(action, HotkeyAction):
            if (
                not isinstance(action.keys, tuple)
                or not action.keys
                or len(action.keys) > 4
                or any(
                    not isinstance(key, str) or key.casefold() not in VALID_KEYS
                    for key in action.keys
                )
            ):
                raise ActionDeniedError("hotkey contains unsupported keys")
            normalized = "+".join(key.casefold() for key in action.keys)
            return f"hotkey keys={normalized}"
        if isinstance(action, ScrollAction):
            if (
                isinstance(action.clicks, bool)
                or not isinstance(action.clicks, int)
                or not -20 <= action.clicks <= 20
            ):
                raise ActionDeniedError("scroll amount is outside the safe range")
            if (action.x is None) != (action.y is None):
                raise ActionDeniedError("scroll requires both x and y or neither")
            if action.x is None or action.y is None:
                return f"scroll clicks={action.clicks}"
            point = self._point(action.x, action.y, bounds)
            return f"scroll clicks={action.clicks} point=({point.x},{point.y})"
        if isinstance(action, DragAction):
            start = self._point(action.start_x, action.start_y, bounds)
            end = self._point(action.end_x, action.end_y, bounds)
            if (
                not isinstance(action.duration, (int, float))
                or isinstance(action.duration, bool)
                or not math.isfinite(action.duration)
                or not 0.0 <= action.duration <= 5.0
            ):
                raise ActionDeniedError("drag duration is outside the safe range")
            return (
                f"drag start=({start.x},{start.y}) end=({end.x},{end.y}) "
                f"duration={action.duration}"
            )
        if isinstance(action, WaitAction):
            if (
                not isinstance(action.seconds, (int, float))
                or isinstance(action.seconds, bool)
                or not math.isfinite(action.seconds)
                or not 0.0 < action.seconds <= 5.0
            ):
                raise ActionDeniedError(
                    "wait duration must be greater than zero and at most 5 seconds"
                )
            return f"wait seconds={action.seconds}"
        if isinstance(action, FinishAction):
            if (
                not isinstance(action.success, bool)
                or not isinstance(action.summary, str)
                or not action.summary.strip()
                or len(action.summary) > 500
            ):
                raise ActionDeniedError("finish result is outside the safe limits")
            return f"finish success={action.success}"
        raise ActionDeniedError("unsupported action type")

    @staticmethod
    def _point(x: object, y: object, bounds: ScreenRegion) -> Point:
        try:
            point = Point(x=x, y=y)  # type: ignore[arg-type]
        except ValueError as error:
            raise ActionDeniedError("action coordinates are invalid") from error
        if not bounds.contains(point):
            raise ActionDeniedError("action point is outside the current observation")
        return point


__all__ = ["CONFIRMATION_PHRASE", "ActionDeniedError", "SafetyPolicy"]
