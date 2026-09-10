"""Build OCR observations from in-memory desktop captures."""

from pathlib import Path
from typing import Protocol

from gui_agent.agent.types import Observation
from gui_agent.perception.fingerprint import image_fingerprint
from gui_agent.perception.ocr import OCRBackend
from gui_agent.types import OCRDetection, Point, ScreenRegion, ScreenshotResult


class CaptureBackend(Protocol):
    """Capture physical monitors or bounded absolute desktop regions."""

    def capture_monitor(
        self,
        monitor_index: int = 1,
        *,
        save_path: Path | None = None,
    ) -> ScreenshotResult:
        """Capture one monitor, optionally using an explicit persistence path."""
        ...

    def capture_region(
        self,
        region: ScreenRegion,
        *,
        save_path: Path | None = None,
    ) -> ScreenshotResult:
        """Capture one absolute desktop region with optional persistence."""
        ...


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
        """Configure one capture mode and an OCR confidence threshold."""
        if monitor_index is not None and region is not None:
            raise ValueError("monitor_index and region are mutually exclusive")
        self._capture = capture
        self._ocr = ocr
        self._monitor_index = 1 if monitor_index is None and region is None else monitor_index
        self._region = region
        self._min_confidence = min_confidence
        self._cache_key: tuple[str, Point, float, str] | None = None
        self._cache_detections: tuple[OCRDetection, ...] = ()

    def observe(self, step_index: int) -> Observation:
        """Capture a frame and reuse OCR only when the complete cache key matches."""
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
        cache_key = self._frame_cache_key(screenshot)
        if cache_key == self._cache_key:
            detections = self._cache_detections
        else:
            detections = tuple(
                self._ocr.recognize(
                    screenshot.image,
                    origin=screenshot.origin,
                    min_confidence=self._min_confidence,
                )
            )
            self._cache_key = cache_key
            self._cache_detections = detections
        return Observation(
            screenshot=screenshot,
            detections=detections,
            step_index=step_index,
        )

    def clear_cache(self) -> None:
        """Force the next observation to run OCR even for an unchanged frame."""
        self._cache_key = None
        self._cache_detections = ()

    def _frame_cache_key(self, screenshot: ScreenshotResult) -> tuple[str, Point, float, str]:
        profile = repr(getattr(self._ocr, "cache_token", type(self._ocr).__qualname__))
        return (
            image_fingerprint(screenshot.image),
            screenshot.origin,
            self._min_confidence,
            profile,
        )


__all__ = ["CaptureBackend", "ObservationBuilder"]
