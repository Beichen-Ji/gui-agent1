from datetime import UTC, datetime
from typing import cast

import numpy as np
import pytest

from gui_agent.types import (
    BoundingBox,
    ImageArray,
    OCRDetection,
    Point,
    ScreenRegion,
    ScreenshotResult,
)

UTC_NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("coordinates", [(1.5, 2), (1, 2.5), (True, 2)])
def test_point_rejects_non_integer_coordinates(coordinates: tuple[object, object]) -> None:
    x, y = cast(tuple[int, int], coordinates)

    with pytest.raises(ValueError, match="integer"):
        Point(x, y)


def test_region_derives_half_open_edges() -> None:
    region = ScreenRegion(left=-100, top=20, width=640, height=480)

    assert region.right == 540
    assert region.bottom == 500
    assert region.contains(Point(-100, 20))
    assert not region.contains(Point(540, 500))


@pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (-1, 10), (10, -1)])
def test_region_rejects_nonpositive_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        ScreenRegion(left=0, top=0, width=width, height=height)


def test_region_contains_complete_subregion() -> None:
    desktop = ScreenRegion(left=-100, top=-50, width=300, height=200)

    assert desktop.contains_region(ScreenRegion(left=-100, top=-50, width=300, height=200))
    assert not desktop.contains_region(ScreenRegion(left=199, top=0, width=2, height=2))


def test_box_derives_integer_center() -> None:
    assert BoundingBox(10, 20, 31, 41).center == Point(20, 30)


@pytest.mark.parametrize(
    "coordinates",
    [(0, 0, 0, 10), (0, 0, 10, 0), (10, 0, 9, 10), (0, 10, 10, 9)],
)
def test_box_rejects_nonpositive_size(coordinates: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        BoundingBox(*coordinates)


def test_screenshot_derives_dimensions_from_bgr_array() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    result = ScreenshotResult(
        image=image,
        monitor_index=1,
        captured_at=UTC_NOW,
        origin=Point(-20, 10),
    )

    assert result.width == 3
    assert result.height == 2


def test_screenshot_rejects_non_uint8_image() -> None:
    image = np.zeros((2, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="uint8"):
        ScreenshotResult(
            image=cast(ImageArray, image),
            monitor_index=1,
            captured_at=UTC_NOW,
            origin=Point(0, 0),
        )


@pytest.mark.parametrize(
    "shape",
    [(0, 3, 3), (2, 0, 3), (2, 3), (2, 3, 1), (2, 3, 4)],
)
def test_screenshot_rejects_non_bgr_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="BGR"):
        ScreenshotResult(
            image=np.zeros(shape, dtype=np.uint8),
            monitor_index=1,
            captured_at=UTC_NOW,
            origin=Point(0, 0),
        )


def test_screenshot_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ScreenshotResult(
            image=np.zeros((2, 3, 3), dtype=np.uint8),
            monitor_index=1,
            captured_at=datetime(2026, 8, 17, 12, 0),
            origin=Point(0, 0),
        )


def test_screenshot_rejects_monitor_zero() -> None:
    with pytest.raises(ValueError, match="monitor_index"):
        ScreenshotResult(
            image=np.zeros((2, 3, 3), dtype=np.uint8),
            monitor_index=0,
            captured_at=UTC_NOW,
            origin=Point(0, 0),
        )


def test_ocr_detection_derives_center() -> None:
    result = OCRDetection("Save", 0.75, BoundingBox(10, 20, 30, 40))

    assert result.center == Point(20, 30)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("inf"), float("nan")])
def test_ocr_detection_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        OCRDetection("Save", confidence, BoundingBox(0, 0, 10, 10))


def test_ocr_detection_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="text"):
        OCRDetection("   ", 0.9, BoundingBox(0, 0, 10, 10))
