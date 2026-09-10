"""Expose deterministic observations and actions over in-process state only."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from typing import Literal, TypeAlias

from gui_agent.agent.policy import SafetyPolicy
from gui_agent.agent.types import (
    AgentAction,
    ClickAction,
    DragAction,
    FinishAction,
    HotkeyAction,
    Observation,
    ScrollAction,
    StepResult,
    TypeTextAction,
    WaitAction,
)
from gui_agent.perception.ocr import OCRBackend
from gui_agent.simulation.render import layout_desktop, render_desktop
from gui_agent.simulation.state import TestbedState
from gui_agent.types import (
    BoundingBox,
    ImageArray,
    OCRDetection,
    Point,
    ScreenRegion,
    ScreenshotResult,
)

ObservationMode: TypeAlias = Literal["oracle", "ocr"]


@dataclass(slots=True)
class SimulatedDesktop:
    """Own local testbed state, deterministic rendering, and simulated time."""

    state: TestbedState
    canvas: tuple[int, int] = (1280, 720)
    elapsed_seconds: float = 0.0
    hitboxes: dict[str, BoundingBox] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Reject canvases too small for the deterministic layout."""
        width, height = self.canvas
        if width < 320 or height < 180:
            raise ValueError("simulation canvas must be at least 320x180")

    @property
    def bounds(self) -> ScreenRegion:
        """Return zero-origin bounds for the simulated canvas."""
        return ScreenRegion(left=0, top=0, width=self.canvas[0], height=self.canvas[1])

    def render(self) -> ImageArray:
        """Render state and refresh the hitboxes used by simulated actions."""
        image, hitboxes = render_desktop(self.state, self.canvas)
        self.hitboxes = hitboxes
        return image

    def advance(self, seconds: float) -> None:
        """Advance only the in-process clock and delayed state transitions."""
        self.elapsed_seconds += seconds
        self.state.wait(seconds)


class SimulatedObservationSource:
    """Render the simulated desktop and describe it like desktop perception."""

    def __init__(
        self,
        desktop: SimulatedDesktop,
        *,
        mode: ObservationMode,
        ocr: OCRBackend | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        """Configure oracle or OCR perception over a simulated desktop."""
        if mode not in {"oracle", "ocr"}:
            raise ValueError(f"unknown simulated observation mode: {mode}")
        if not isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
            raise ValueError("minimum confidence must be between 0.0 and 1.0")
        if mode == "ocr" and ocr is None:
            raise ValueError("ocr mode requires an OCR backend")
        self._desktop = desktop
        self._mode = mode
        self._ocr = ocr
        self._min_confidence = min_confidence

    @property
    def desktop(self) -> SimulatedDesktop:
        """Return the in-process simulated desktop."""
        return self._desktop

    def observe(self, step_index: int) -> Observation:
        """Render one indexed observation without capturing the real desktop."""
        image = self._desktop.render()
        screenshot = ScreenshotResult(
            image=image,
            monitor_index=None,
            captured_at=datetime.now(UTC),
            origin=Point(0, 0),
        )
        if self._mode == "oracle":
            layout = layout_desktop(self._desktop.state, self._desktop.canvas)
            detections = tuple(
                OCRDetection(text=control.label, confidence=1.0, box=control.box)
                for control in layout
            )
        else:
            assert self._ocr is not None
            detections = tuple(
                self._ocr.recognize(
                    image,
                    origin=screenshot.origin,
                    min_confidence=self._min_confidence,
                )
            )
        return Observation(
            screenshot=screenshot,
            detections=detections,
            step_index=step_index,
        )


class SimulatedActionExecutor:
    """Apply actions in process and never emit real desktop input."""

    def __init__(
        self,
        desktop: SimulatedDesktop,
        *,
        clock: Callable[[float], None] = lambda _seconds: None,
    ) -> None:
        """Bind simulated state and an optional simulated wait callback."""
        self._desktop = desktop
        self._clock = clock

    @property
    def desktop(self) -> SimulatedDesktop:
        """Return the in-process simulated desktop."""
        return self._desktop

    def execute(self, action: AgentAction, *, step_index: int) -> StepResult:
        """Apply one action to simulated state and return a structured result."""
        try:
            message = self._dispatch(action)
        except (OSError, UnicodeError, ValueError) as error:
            return StepResult(
                step_index=step_index,
                action=action,
                status="failed",
                message=f"simulated {action.kind} failed: {error}",
            )
        return StepResult(
            step_index=step_index,
            action=action,
            status="executed",
            message=message,
        )

    def _dispatch(self, action: AgentAction) -> str:
        if isinstance(action, ClickAction):
            point = Point(action.x, action.y)
            self._require_in_bounds(point)
            self._desktop.render()
            control_id = self._control_at(point)
            if control_id is None:
                return f"simulated click missed all controls at ({point.x}, {point.y})"
            self._desktop.state.click(control_id)
            return f"simulated click activated {control_id}"
        if isinstance(action, TypeTextAction):
            self._desktop.state.type_text(action.text)
            return "simulated text entered into the focused control"
        if isinstance(action, HotkeyAction):
            self._desktop.state.hotkey(action.keys)
            return f"simulated hotkey {'+'.join(action.keys)} applied"
        if isinstance(action, ScrollAction):
            if (action.x is None) != (action.y is None):
                raise ValueError("scroll requires both x and y or neither")
            if action.x is not None and action.y is not None:
                self._require_in_bounds(Point(action.x, action.y))
            self._desktop.state.scroll(action.clicks)
            return f"simulated scroll moved by {action.clicks} clicks"
        if isinstance(action, DragAction):
            start = Point(action.start_x, action.start_y)
            end = Point(action.end_x, action.end_y)
            self._require_in_bounds(start)
            self._require_in_bounds(end)
            self._desktop.render()
            control_id = self._control_at(start)
            if control_id != "settings.volume":
                raise ValueError("drag must start on the settings volume slider")
            slider = self._desktop.hitboxes[control_id]
            width = slider.right - slider.left
            value = min(1.0, max(0.0, (end.x - slider.left) / width))
            self._desktop.state.drag(control_id, value)
            return f"simulated drag set {control_id} to {value:.3f}"
        if isinstance(action, WaitAction):
            self._clock(action.seconds)
            self._desktop.advance(action.seconds)
            return f"simulated clock advanced by {action.seconds:.3f} seconds"
        if isinstance(action, FinishAction):
            return f"simulated finish accepted; success={action.success}"
        raise ValueError("unsupported simulated action type")

    def _control_at(self, point: Point) -> str | None:
        for control_id, box in self._desktop.hitboxes.items():
            if box.left <= point.x < box.right and box.top <= point.y < box.bottom:
                return control_id
        return None

    def _require_in_bounds(self, point: Point) -> None:
        if not self._desktop.bounds.contains(point):
            raise ValueError("action point is outside the simulated desktop")


class SimulationPolicy:
    """Validate simulated actions without ever authorizing real desktop input."""

    def __init__(self) -> None:
        """Create a preview-only validator with live execution disabled."""
        self._validator = SafetyPolicy(execute=False)

    def authorize(
        self,
        action: AgentAction,
        observation: Observation,
        *,
        expected_outcome: str,
    ) -> None:
        """Validate an action while keeping the real-input gate disabled."""
        self._validator.authorize(
            action,
            observation,
            expected_outcome=expected_outcome,
        )


__all__ = [
    "ObservationMode",
    "SimulatedActionExecutor",
    "SimulatedDesktop",
    "SimulatedObservationSource",
    "SimulationPolicy",
]
