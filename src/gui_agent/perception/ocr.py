from collections.abc import Callable, Sequence
from math import ceil, floor, isfinite
from typing import Protocol, cast, runtime_checkable

import numpy as np

from gui_agent.perception.preprocessing import (
    DEFAULT_OCR_PROFILE,
    OCRProfile,
    get_ocr_profile,
    preprocess_image,
)
from gui_agent.types import BoundingBox, ImageArray, OCRDetection, Point


class OCRDependencyError(RuntimeError):
    """An optional OCR runtime dependency is unavailable."""


class OCRInitializationError(RuntimeError):
    """The OCR model or reader could not be initialized."""


class OCRInferenceError(RuntimeError):
    """The OCR backend failed or returned malformed output."""


class OCRReader(Protocol):
    def readtext(
        self,
        image: ImageArray,
        *,
        detail: int,
        paragraph: bool,
        decoder: str,
        beamWidth: int,
        batch_size: int,
        workers: int,
        canvas_size: int,
        mag_ratio: float,
        contrast_ths: float,
        adjust_contrast: float,
    ) -> Sequence[object]: ...


ReaderFactory = Callable[[list[str], bool | str], OCRReader]
CudaProbe = Callable[[], bool]
DEFAULT_ORIGIN = Point(0, 0)


@runtime_checkable
class OCRBackend(Protocol):
    def recognize(
        self,
        image: ImageArray,
        *,
        origin: Point = DEFAULT_ORIGIN,
        min_confidence: float = 0.0,
    ) -> list[OCRDetection]: ...


def _default_reader_factory(languages: list[str], gpu: bool | str) -> OCRReader:
    try:
        import easyocr
    except ModuleNotFoundError as error:
        raise OCRDependencyError(
            "EasyOCR is not installed; run `uv sync --extra ocr`"
        ) from error
    return cast(OCRReader, easyocr.Reader(languages, gpu=gpu, verbose=False))


def _default_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise OCRDependencyError(
            "PyTorch is not installed; run `uv sync --extra ocr`"
        ) from error
    return bool(torch.cuda.is_available())


def _validate_image(image: ImageArray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("image must be a non-empty uint8 NumPy array")
    if image.ndim == 2:
        return
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be grayscale or BGR with three channels")


class EasyOCRBackend:
    """Lazy EasyOCR adapter that returns project-standard absolute detections."""

    def __init__(
        self,
        languages: Sequence[str] = ("ch_sim", "en"),
        *,
        gpu: bool | str | None = None,
        reader_factory: ReaderFactory | None = None,
        cuda_available: CudaProbe | None = None,
        profile: str | OCRProfile = DEFAULT_OCR_PROFILE,
    ) -> None:
        if not languages or any(
            not isinstance(item, str) or not item.strip() for item in languages
        ):
            raise ValueError("languages must contain at least one non-empty language code")
        self._languages = tuple(languages)
        self._gpu = gpu
        self._reader_factory = reader_factory or _default_reader_factory
        self._cuda_available = cuda_available or _default_cuda_available
        self._reader: OCRReader | None = None
        self._profile = get_ocr_profile(profile)

    @property
    def cache_token(self) -> str:
        return repr(self._profile)

    def recognize(
        self,
        image: ImageArray,
        *,
        origin: Point = DEFAULT_ORIGIN,
        min_confidence: float = 0.0,
    ) -> list[OCRDetection]:
        _validate_image(image)
        if not isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        processed = preprocess_image(image, self._profile)
        reader = self._reader_instance()
        try:
            raw_results = reader.readtext(
                processed.image,
                detail=1,
                paragraph=False,
                decoder=self._profile.decoder,
                beamWidth=self._profile.beam_width,
                batch_size=self._profile.batch_size,
                workers=self._profile.workers,
                canvas_size=self._profile.canvas_size,
                mag_ratio=1.0,
                contrast_ths=self._profile.contrast_threshold,
                adjust_contrast=self._profile.adjust_contrast,
            )
        except Exception as error:
            raise OCRInferenceError("EasyOCR inference failed") from error
        return self._normalize(
            raw_results,
            origin,
            min_confidence,
            scale_x=processed.scale_x,
            scale_y=processed.scale_y,
        )

    def _reader_instance(self) -> OCRReader:
        if self._reader is not None:
            return self._reader
        try:
            gpu = self._gpu if self._gpu is not None else self._cuda_available()
            self._reader = self._reader_factory(list(self._languages), gpu)
        except OCRDependencyError:
            raise
        except ModuleNotFoundError as error:
            raise OCRDependencyError(
                "EasyOCR is not installed; run `uv sync --extra ocr`"
            ) from error
        except Exception as error:
            raise OCRInitializationError("failed to initialize EasyOCR reader") from error
        return self._reader

    @classmethod
    def _normalize(
        cls,
        raw_results: Sequence[object],
        origin: Point,
        min_confidence: float,
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> list[OCRDetection]:
        detections: list[OCRDetection] = []
        try:
            for raw_result in raw_results:
                detection = cls._normalize_one(
                    raw_result,
                    origin,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                if detection is not None and detection.confidence >= min_confidence:
                    detections.append(detection)
        except OCRInferenceError:
            raise
        except Exception as error:
            raise OCRInferenceError("EasyOCR returned a malformed result") from error
        return detections

    @staticmethod
    def _normalize_one(
        raw_result: object,
        origin: Point,
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> OCRDetection | None:
        if not isinstance(raw_result, (list, tuple)) or len(raw_result) != 3:
            raise OCRInferenceError("EasyOCR returned a malformed result tuple")
        raw_box, raw_text, raw_confidence = raw_result
        if not isinstance(raw_text, str):
            raise OCRInferenceError("EasyOCR returned malformed text")
        if not raw_text.strip():
            return None
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError) as error:
            raise OCRInferenceError("EasyOCR returned malformed confidence") from error
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise OCRInferenceError("EasyOCR returned malformed confidence")
        try:
            points = np.asarray(raw_box, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise OCRInferenceError("EasyOCR returned a malformed bounding box") from error
        if points.ndim != 2 or points.shape != (4, 2) or not bool(np.isfinite(points).all()):
            raise OCRInferenceError("EasyOCR returned a malformed bounding box")
        try:
            box = BoundingBox(
                left=floor(float(points[:, 0].min()) / scale_x) + origin.x,
                top=floor(float(points[:, 1].min()) / scale_y) + origin.y,
                right=ceil(float(points[:, 0].max()) / scale_x) + origin.x,
                bottom=ceil(float(points[:, 1].max()) / scale_y) + origin.y,
            )
        except ValueError as error:
            raise OCRInferenceError("EasyOCR returned a malformed bounding box") from error
        return OCRDetection(raw_text, confidence, box)
