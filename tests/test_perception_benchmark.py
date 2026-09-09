from pathlib import Path

import pytest

from gui_agent.perception.benchmark import (
    benchmark_profiles,
    box_iou,
    latency_summary,
    load_benchmark_manifest,
    normalize_text,
    score_detections,
)
from gui_agent.types import BoundingBox, OCRDetection

FIXTURE = Path(__file__).parent / "fixtures" / "ocr_benchmark" / "manifest.json"


def detection(text: str, box: BoundingBox) -> OCRDetection:
    return OCRDetection(text=text, confidence=0.9, box=box)


def test_benchmark_normalizes_text_without_losing_words() -> None:
    assert normalize_text("  Save\n  FILE  ") == "save file"


def test_box_iou_uses_intersection_over_union() -> None:
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(5, 0, 15, 10)

    assert box_iou(first, first) == 1.0
    assert box_iou(first, second) == pytest.approx(1 / 3)


def test_scoring_matches_text_and_box_once_for_precision_recall_f1() -> None:
    expected = (
        detection("Save File", BoundingBox(0, 0, 20, 10)),
        detection("Cancel", BoundingBox(30, 0, 50, 10)),
    )
    predicted = (
        detection(" save  file ", BoundingBox(1, 0, 21, 10)),
        detection("Wrong", BoundingBox(30, 0, 50, 10)),
        detection("Save File", BoundingBox(70, 0, 90, 10)),
    )

    score = score_detections(predicted, expected, iou_threshold=0.5)

    assert score.true_positives == 1
    assert score.false_positives == 2
    assert score.false_negatives == 1
    assert score.precision == pytest.approx(1 / 3)
    assert score.recall == 0.5
    assert score.f1 == pytest.approx(0.4)


def test_latency_summary_reports_median_and_p95() -> None:
    summary = latency_summary((1.0, 2.0, 3.0, 4.0, 5.0))

    assert summary.median_ms == 3.0
    assert summary.p95_ms == pytest.approx(4.8)


def test_latency_summary_rejects_empty_or_negative_samples() -> None:
    with pytest.raises(ValueError, match="latency"):
        latency_summary(())
    with pytest.raises(ValueError, match="latency"):
        latency_summary((1.0, -1.0))


def test_manifest_renders_deterministic_bgr_cases_with_ground_truth() -> None:
    cases = load_benchmark_manifest(FIXTURE)

    assert [case.id for case in cases] == ["toolbar-buttons", "small-status-text"]
    assert cases[0].image.shape == (240, 640, 3)
    assert cases[0].image.dtype.name == "uint8"
    assert [item.text for item in cases[0].expected] == [
        "Browser",
        "Files",
        "Messages",
    ]


class EmptyBackend:
    cache_token = "fake-profile"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image: object, **_kwargs: object) -> list[OCRDetection]:
        del image
        self.calls += 1
        return []


def test_benchmark_profiles_measures_cold_and_exact_frame_cache() -> None:
    backends: list[EmptyBackend] = []

    def factory(_profile: str) -> EmptyBackend:
        backend = EmptyBackend()
        backends.append(backend)
        return backend

    report = benchmark_profiles(
        FIXTURE,
        ("fast", "balanced"),
        warmup=1,
        runs=2,
        backend_factory=factory,
    )

    assert report.schema_version == 1
    assert set(report.profiles) == {"fast", "balanced"}
    assert len(backends) == 2
    assert all(backend.calls == 6 for backend in backends)
    assert all(result.cold_latency.median_ms >= 0 for result in report.profiles.values())
    assert all(result.cached_latency.median_ms >= 0 for result in report.profiles.values())
