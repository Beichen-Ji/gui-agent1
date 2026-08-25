from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from gui_agent.agent.observation import ObservationBuilder
from gui_agent.types import BoundingBox, ImageArray, OCRDetection, Point, ScreenshotResult

DEFAULT_ORIGIN = Point(0, 0)


class FakeCapture:
    def __init__(
        self,
        screenshot: ScreenshotResult,
        *,
        error: Exception | None = None,
    ) -> None:
        self.screenshot = screenshot
        self.error = error
        self.calls: list[tuple[int, Path | None]] = []

    def capture_monitor(
        self,
        monitor_index: int = 1,
        *,
        save_path: Path | None = None,
    ) -> ScreenshotResult:
        self.calls.append((monitor_index, save_path))
        if self.error is not None:
            raise self.error
        return self.screenshot


class FakeOCR:
    def __init__(
        self,
        detections: list[OCRDetection],
        *,
        error: Exception | None = None,
    ) -> None:
        self.detections = detections
        self.error = error
        self.calls: list[tuple[ImageArray, Point, float]] = []

    def recognize(
        self,
        image: ImageArray,
        *,
        origin: Point = DEFAULT_ORIGIN,
        min_confidence: float = 0.0,
    ) -> list[OCRDetection]:
        self.calls.append((image, origin, min_confidence))
        if self.error is not None:
            raise self.error
        return list(self.detections)


def screenshot_fixture() -> ScreenshotResult:
    return ScreenshotResult(
        image=np.zeros((40, 60, 3), dtype=np.uint8),
        monitor_index=2,
        captured_at=datetime(2026, 8, 25, tzinfo=UTC),
        origin=Point(-60, 20),
    )


def test_observe_captures_once_and_runs_ocr_at_screenshot_origin() -> None:
    screenshot = screenshot_fixture()
    detection = OCRDetection("Browser", 0.95, BoundingBox(-50, 30, 10, 50))
    capture = FakeCapture(screenshot)
    ocr = FakeOCR([detection])

    observation = ObservationBuilder(
        capture,
        ocr,
        monitor_index=2,
        min_confidence=0.5,
    ).observe(step_index=3)

    assert observation.screenshot is screenshot
    assert observation.detections == (detection,)
    assert observation.step_index == 3
    assert capture.calls == [(2, None)]
    assert len(ocr.calls) == 1
    ocr_image, ocr_origin, ocr_confidence = ocr.calls[0]
    assert ocr_image is screenshot.image
    assert ocr_origin == Point(-60, 20)
    assert ocr_confidence == 0.5


def test_observe_propagates_capture_errors_without_running_ocr() -> None:
    capture = FakeCapture(screenshot_fixture(), error=RuntimeError("capture unavailable"))
    ocr = FakeOCR([])

    with pytest.raises(RuntimeError, match="capture unavailable"):
        ObservationBuilder(capture, ocr).observe(step_index=0)

    assert ocr.calls == []


def test_observe_propagates_ocr_errors() -> None:
    capture = FakeCapture(screenshot_fixture())
    ocr = FakeOCR([], error=RuntimeError("OCR unavailable"))

    with pytest.raises(RuntimeError, match="OCR unavailable"):
        ObservationBuilder(capture, ocr).observe(step_index=0)

    assert capture.calls == [(1, None)]
