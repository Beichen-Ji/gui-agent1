from pathlib import Path
from typing import Protocol

from gui_agent.agent.types import Observation
from gui_agent.perception.ocr import OCRBackend
from gui_agent.types import ScreenRegion, ScreenshotResult


class CaptureBackend(Protocol):
    def capture_monitor(
        self,
        monitor_index: int = 1,
        *,
        save_path: Path | None = None,
    ) -> ScreenshotResult: ...

    def capture_region(
        self,
        region: ScreenRegion,
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
        monitor_index: int | None = None,
        region: ScreenRegion | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        if monitor_index is not None and region is not None:
            raise ValueError("monitor_index and region are mutually exclusive")
        self._capture = capture
        self._ocr = ocr
        self._monitor_index = 1 if monitor_index is None and region is None else monitor_index
        self._region = region
        self._min_confidence = min_confidence

    def observe(self, step_index: int) -> Observation:
        if self._region is None:
            assert self._monitor_index is not None
            screenshot = self._capture.capture_monitor(
                self._monitor_index,
                save_path=None,
            )
        else:
            screenshot = self._capture.capture_region(
                self._region,
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
