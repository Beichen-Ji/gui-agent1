import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from statistics import median
from typing import cast

import cv2
import numpy as np

from gui_agent.agent.observation import ObservationBuilder
from gui_agent.perception.ocr import OCRBackend
from gui_agent.types import (
    BoundingBox,
    ImageArray,
    OCRDetection,
    Point,
    ScreenshotResult,
)


def normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection_width = max(0, min(first.right, second.right) - max(first.left, second.left))
    intersection_height = max(
        0,
        min(first.bottom, second.bottom) - max(first.top, second.top),
    )
    intersection = intersection_width * intersection_height
    first_area = (first.right - first.left) * (first.bottom - first.top)
    second_area = (second.right - second.left) * (second.bottom - second.top)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


@dataclass(frozen=True, slots=True)
class DetectionScore:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def score_detections(
    predicted: tuple[OCRDetection, ...],
    expected: tuple[OCRDetection, ...],
    *,
    iou_threshold: float = 0.5,
) -> DetectionScore:
    if not isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    unmatched = set(range(len(expected)))
    true_positives = 0
    for detection in predicted:
        candidates = [
            (box_iou(detection.box, expected[index].box), index)
            for index in unmatched
            if normalize_text(detection.text) == normalize_text(expected[index].text)
        ]
        if not candidates:
            continue
        overlap, index = max(candidates)
        if overlap >= iou_threshold:
            unmatched.remove(index)
            true_positives += 1
    false_positives = len(predicted) - true_positives
    false_negatives = len(expected) - true_positives
    precision = (
        true_positives / len(predicted)
        if predicted
        else (1.0 if not expected else 0.0)
    )
    recall = (
        true_positives / len(expected)
        if expected
        else (1.0 if not predicted else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionScore(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


@dataclass(frozen=True, slots=True)
class LatencySummary:
    median_ms: float
    p95_ms: float


def latency_summary(samples_ms: tuple[float, ...]) -> LatencySummary:
    if not samples_ms or any(
        isinstance(sample, bool)
        or not isinstance(sample, (int, float))
        or not isfinite(sample)
        or sample < 0.0
        for sample in samples_ms
    ):
        raise ValueError("latency samples must be finite non-negative numbers")
    values = np.asarray(samples_ms, dtype=np.float64)
    return LatencySummary(
        median_ms=float(median(samples_ms)),
        p95_ms=float(np.percentile(values, 95)),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    image: ImageArray
    expected: tuple[OCRDetection, ...]


@dataclass(frozen=True, slots=True)
class ProfileBenchmark:
    profile: str
    score: DetectionScore
    cold_latency: LatencySummary
    cached_latency: LatencySummary
    cache_median_reduction: float


@dataclass(frozen=True, slots=True)
class OCRBenchmarkReport:
    schema_version: int
    case_count: int
    warmup: int
    runs: int
    profiles: Mapping[str, ProfileBenchmark]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "case_count": self.case_count,
            "warmup": self.warmup,
            "runs": self.runs,
            "profiles": {
                name: {
                    "precision": result.score.precision,
                    "recall": result.score.recall,
                    "f1": result.score.f1,
                    "cold_median_ms": result.cold_latency.median_ms,
                    "cold_p95_ms": result.cold_latency.p95_ms,
                    "cached_median_ms": result.cached_latency.median_ms,
                    "cached_p95_ms": result.cached_latency.p95_ms,
                    "cache_median_reduction": result.cache_median_reduction,
                }
                for name, result in self.profiles.items()
            },
        }


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _render_text(image: ImageArray, text: str, box: BoundingBox) -> None:
    available_width = box.right - box.left - 6
    available_height = box.bottom - box.top - 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    size, _ = cv2.getTextSize(text, font, scale, thickness)
    if size[0] > available_width or size[1] > available_height:
        scale *= min(available_width / max(size[0], 1), available_height / max(size[1], 1))
    baseline = box.top + max(1, (box.bottom - box.top + size[1]) // 2)
    cv2.putText(
        image,
        text,
        (box.left + 3, min(box.bottom - 3, baseline)),
        font,
        max(scale, 0.2),
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def load_benchmark_manifest(path: Path) -> tuple[BenchmarkCase, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read OCR benchmark manifest: {path}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("OCR benchmark manifest requires schema_version 1")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("OCR benchmark manifest requires non-empty cases")

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("OCR benchmark cases must be objects")
        case_id = raw_case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("OCR benchmark case IDs must be non-empty and unique")
        seen.add(case_id)
        width = _integer(raw_case.get("width"), label="case width", minimum=1)
        height = _integer(raw_case.get("height"), label="case height", minimum=1)
        raw_elements = raw_case.get("elements")
        if not isinstance(raw_elements, list) or not raw_elements:
            raise ValueError("OCR benchmark cases require non-empty elements")
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        expected: list[OCRDetection] = []
        for raw_element in raw_elements:
            if not isinstance(raw_element, dict):
                raise ValueError("OCR benchmark elements must be objects")
            text = raw_element.get("text")
            raw_box = raw_element.get("box")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("OCR benchmark element text must not be blank")
            if (
                not isinstance(raw_box, list)
                or len(raw_box) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_box)
            ):
                raise ValueError("OCR benchmark element box must contain four integers")
            left, top, right, bottom = cast(list[int], raw_box)
            box = BoundingBox(left, top, right, bottom)
            if left < 0 or top < 0 or right > width or bottom > height:
                raise ValueError("OCR benchmark element box must stay inside the image")
            _render_text(image, text, box)
            expected.append(OCRDetection(text, 1.0, box))
        cases.append(BenchmarkCase(case_id, image, tuple(expected)))
    return tuple(cases)


class _ArrayCapture:
    def __init__(self, image: ImageArray) -> None:
        self.image = image

    def capture_monitor(
        self,
        monitor_index: int = 1,
        *,
        save_path: Path | None = None,
    ) -> ScreenshotResult:
        del save_path
        return ScreenshotResult(
            image=self.image,
            monitor_index=monitor_index,
            captured_at=datetime.now(UTC),
            origin=Point(0, 0),
        )

    def capture_region(
        self,
        region: object,
        *,
        save_path: Path | None = None,
    ) -> ScreenshotResult:
        del region, save_path
        raise AssertionError("benchmark capture uses a synthetic monitor")


def _aggregate_score(scores: list[DetectionScore]) -> DetectionScore:
    true_positives = sum(score.true_positives for score in scores)
    false_positives = sum(score.false_positives for score in scores)
    false_negatives = sum(score.false_negatives for score in scores)
    predicted = true_positives + false_positives
    expected = true_positives + false_negatives
    precision = true_positives / predicted if predicted else (1.0 if not expected else 0.0)
    recall = true_positives / expected if expected else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionScore(
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
    )


def benchmark_profiles(
    manifest: Path,
    profiles: tuple[str, ...],
    *,
    warmup: int,
    runs: int,
    backend_factory: Callable[[str], OCRBackend],
    clock: Callable[[], float] = time.perf_counter,
) -> OCRBenchmarkReport:
    if not profiles or len(profiles) != len(set(profiles)):
        raise ValueError("profiles must contain unique profile names")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    cases = load_benchmark_manifest(manifest)
    results: dict[str, ProfileBenchmark] = {}
    for profile in profiles:
        backend = backend_factory(profile)
        for _ in range(warmup):
            for case in cases:
                backend.recognize(case.image)
        scores: list[DetectionScore] = []
        cold_samples: list[float] = []
        cached_samples: list[float] = []
        for _ in range(runs):
            for case in cases:
                builder = ObservationBuilder(
                    _ArrayCapture(case.image),
                    backend,
                    monitor_index=1,
                )
                start = clock()
                cold = builder.observe(0)
                cold_samples.append(max(0.0, (clock() - start) * 1000.0))
                start = clock()
                cached = builder.observe(1)
                cached_samples.append(max(0.0, (clock() - start) * 1000.0))
                assert cold.detections == cached.detections
                scores.append(score_detections(cold.detections, case.expected))
        cold_latency = latency_summary(tuple(cold_samples))
        cached_latency = latency_summary(tuple(cached_samples))
        reduction = (
            max(0.0, 1.0 - cached_latency.median_ms / cold_latency.median_ms)
            if cold_latency.median_ms > 0.0
            else 0.0
        )
        results[profile] = ProfileBenchmark(
            profile=profile,
            score=_aggregate_score(scores),
            cold_latency=cold_latency,
            cached_latency=cached_latency,
            cache_median_reduction=reduction,
        )
    return OCRBenchmarkReport(
        schema_version=1,
        case_count=len(cases),
        warmup=warmup,
        runs=runs,
        profiles=results,
    )


__all__ = [
    "DetectionScore",
    "BenchmarkCase",
    "LatencySummary",
    "OCRBenchmarkReport",
    "ProfileBenchmark",
    "benchmark_profiles",
    "box_iou",
    "latency_summary",
    "load_benchmark_manifest",
    "normalize_text",
    "score_detections",
]
