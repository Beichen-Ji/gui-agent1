from pathlib import Path
from typing import cast

import numpy as np
import pytest

import gui_agent.perception as perception
from gui_agent.perception.localization import (
    HIGH_CONFIDENCE_BGR,
    LOW_CONFIDENCE_BGR,
    MatchMode,
    annotate_detections,
    find_text,
)
from gui_agent.types import BoundingBox, ImageArray, OCRDetection, Point


def detection(text: str, *, confidence: float = 0.9) -> OCRDetection:
    return OCRDetection(
        text=text,
        confidence=confidence,
        box=BoundingBox(left=10, top=20, right=50, bottom=40),
    )


def test_localization_public_api_is_exported() -> None:
    assert perception.MatchMode is MatchMode
    assert perception.find_text is find_text
    assert perception.annotate_detections is annotate_detections


def test_find_text_supports_case_insensitive_contains_without_choosing() -> None:
    detections = [detection("Save"), detection("Save As"), detection("Cancel")]

    matches = find_text(
        detections,
        "save",
        mode=MatchMode.CONTAINS,
        case_sensitive=False,
    )

    assert [item.text for item in matches] == ["Save", "Save As"]


def test_find_text_exact_case_sensitive_preserves_duplicates_and_order() -> None:
    detections = [
        detection("Save"),
        detection("save"),
        detection("Save As"),
        detection("Save"),
    ]

    matches = find_text(detections, "Save")

    assert matches == [detections[0], detections[3]]


def test_find_text_exact_can_ignore_case() -> None:
    detections = [detection("SAVE"), detection("save as")]

    matches = find_text(detections, "save", case_sensitive=False)

    assert matches == [detections[0]]


def test_find_text_returns_empty_list_when_nothing_matches() -> None:
    assert find_text([detection("Cancel")], "Save") == []


def test_find_text_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        find_text([detection("Save")], "")


def test_annotate_detections_draws_on_independent_copy_with_confidence_colors() -> None:
    image = np.zeros((80, 110, 3), dtype=np.uint8)
    detections = [
        OCRDetection("Save", 0.9, BoundingBox(5, 25, 45, 60)),
        OCRDetection("Cancel", 0.4, BoundingBox(60, 25, 105, 60)),
    ]

    annotated = annotate_detections(image, detections, confidence_cutoff=0.8)

    assert not np.shares_memory(image, annotated)
    assert np.count_nonzero(image) == 0
    assert tuple(annotated[25, 5]) == HIGH_CONFIDENCE_BGR
    assert tuple(annotated[42, 25]) == HIGH_CONFIDENCE_BGR
    assert tuple(annotated[25, 60]) == LOW_CONFIDENCE_BGR
    assert tuple(annotated[42, 82]) == LOW_CONFIDENCE_BGR
    assert np.count_nonzero(annotated[:24]) > 0


def test_annotate_detections_treats_cutoff_as_high_confidence() -> None:
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    item = OCRDetection("X", 0.8, BoundingBox(5, 10, 20, 25))

    annotated = annotate_detections(image, [item], confidence_cutoff=0.8)

    assert tuple(annotated[10, 5]) == HIGH_CONFIDENCE_BGR


def test_annotate_detections_translates_absolute_boxes_by_image_origin() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    item = OCRDetection("X", 0.9, BoundingBox(-90, 210, -70, 230))

    annotated = annotate_detections(image, [item], origin=Point(-100, 200))

    assert tuple(annotated[10, 10]) == HIGH_CONFIDENCE_BGR
    assert tuple(annotated[20, 20]) == HIGH_CONFIDENCE_BGR


def test_annotate_detections_without_items_returns_unchanged_copy() -> None:
    image = np.full((5, 6, 3), 17, dtype=np.uint8)

    annotated = annotate_detections(image, [])

    assert np.array_equal(annotated, image)
    assert not np.shares_memory(annotated, image)


@pytest.mark.parametrize("cutoff", [-0.01, 1.01, float("inf"), float("nan")])
def test_annotate_detections_rejects_invalid_confidence_cutoff(cutoff: float) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="confidence_cutoff"):
        annotate_detections(image, [], confidence_cutoff=cutoff)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3, 3), dtype=np.float32),
        np.zeros((0, 3, 3), dtype=np.uint8),
        np.zeros((2, 3), dtype=np.uint8),
        np.zeros((2, 3, 4), dtype=np.uint8),
    ],
)
def test_annotate_detections_rejects_invalid_image(
    image: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> None:
    with pytest.raises(ValueError, match="image"):
        annotate_detections(cast(ImageArray, image), [])


def test_annotate_detections_reports_missing_requested_font(tmp_path: Path) -> None:
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    item = OCRDetection("Save", 0.9, BoundingBox(5, 10, 20, 25))
    missing_font = tmp_path / "missing.ttf"

    with pytest.raises(ValueError, match=r"font_path does not exist: .*missing\.ttf"):
        annotate_detections(image, [item], font_path=missing_font)
