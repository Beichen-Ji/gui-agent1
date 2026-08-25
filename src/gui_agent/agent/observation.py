from pathlib import Path
from typing import Protocol

from gui_agent.agent.types import Observation
from gui_agent.perception.ocr import OCRBackend
from gui_agent.types import ScreenshotResult


class CaptureBackend(Protocol):
    def capture_monitor(
        self,
        monitor_index: int = 1,
        *,
        save_path: Path | None = None,
    ) -> ScreenshotResult: ...


class ObservationBuilder:
    """Combine one in-memory desktop capture with OCR detections."""

    def __init__(
        self,
        capture: CaptureBackend,
        ocr: OCRBackend,
        *,
        monitor_index: int = 1,
        min_confidence: float = 0.0,
    ) -> None:
        self._capture = capture
        self._ocr = ocr
        self._monitor_index = monitor_index
        self._min_confidence = min_confidence

    def observe(self, step_index: int) -> Observation:
        screenshot = self._capture.capture_monitor(
            self._monitor_index,
            save_path=None,
        )
        detections = self._ocr.recognize(
            screenshot.image,
            origin=screenshot.origin,
            min_confidence=self._min_confidence,
        )
        return Observation(
            screenshot=screenshot,
            detections=tuple(detections),
            step_index=step_index,
        )


__all__ = ["CaptureBackend", "ObservationBuilder"]
