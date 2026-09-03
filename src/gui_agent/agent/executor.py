import time
from collections.abc import Callable
from typing import Literal

from gui_agent.agent.types import (
    AgentAction,
    ClickAction,
    DragAction,
    FinishAction,
    HotkeyAction,
    ScrollAction,
    StepResult,
    TypeTextAction,
    WaitAction,
)
from gui_agent.control.controller import DesktopController
from gui_agent.types import Point


class ActionExecutionError(RuntimeError):
    """An authorized action could not be mapped or executed safely."""


class ActionExecutor:
    """The only Agent-layer adapter allowed to call DesktopController."""

    def __init__(
        self,
        controller: DesktopController,
        *,
        clock: Callable[[float], None] = time.sleep,
    ) -> None:
        self._controller = controller
        self._clock = clock

    @property
    def controller(self) -> DesktopController:
        return self._controller

    def execute(self, action: AgentAction, *, step_index: int) -> StepResult:
        kind = getattr(action, "kind", "unknown")
        try:
            self._dispatch(action)
        except ActionExecutionError:
            raise
        except Exception as error:
            raise ActionExecutionError(f"failed to execute {kind} action") from error

        if isinstance(action, FinishAction):
            return StepResult(
                step_index=step_index,
                action=action,
                status="executed",
                message=f"finish signal accepted; success={action.success}",
            )
        status: Literal["dry_run", "executed"] = (
            "dry_run" if self._controller.dry_run else "executed"
        )
        message = (
            f"{kind} recorded without desktop input"
            if self._controller.dry_run
            else f"{kind} executed"
        )
        return StepResult(
            step_index=step_index,
            action=action,
            status=status,
            message=message,
        )

    def _dispatch(self, action: AgentAction) -> None:
        if isinstance(action, ClickAction):
            self._controller.click(
                Point(action.x, action.y),
                button=action.button,
                clicks=action.clicks,
            )
            return
        if isinstance(action, TypeTextAction):
            self._controller.type_text(action.text)
            return
        if isinstance(action, HotkeyAction):
            self._controller.hotkey(*action.keys)
            return
        if isinstance(action, ScrollAction):
            if (action.x is None) != (action.y is None):
                raise ActionExecutionError("scroll requires both x and y or neither")
            point = (
                None
                if action.x is None or action.y is None
                else Point(action.x, action.y)
            )
            self._controller.scroll(action.clicks, point=point)
            return
        if isinstance(action, DragAction):
            self._controller.drag_to(
                Point(action.start_x, action.start_y),
                Point(action.end_x, action.end_y),
                duration=action.duration,
            )
            return
        if isinstance(action, WaitAction):
            if not self._controller.dry_run:
                self._clock(action.seconds)
            return
        if isinstance(action, FinishAction):
            return
        raise ActionExecutionError("unsupported action type")


__all__ = ["ActionExecutionError", "ActionExecutor"]
