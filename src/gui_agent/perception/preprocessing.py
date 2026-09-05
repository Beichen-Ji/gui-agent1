from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal, cast

import cv2
import numpy as np

from gui_agent.types import ImageArray


@dataclass(frozen=True, slots=True)
class OCRProfile:
    decoder: Literal["greedy", "beamsearch"]
    beam_width: int
    batch_size: int
    workers: int
    canvas_size: int
    mag_ratio: float
    contrast_threshold: float
    adjust_contrast: float
    preprocessing: Literal["none", "grayscale", "clahe"]

    def __post_init__(self) -> None:
        if self.decoder not in {"greedy", "beamsearch"}:
            raise ValueError("decoder must be greedy or beamsearch")
        self._integer("beam_width", self.beam_width, minimum=1, maximum=20)
        self._integer("batch_size", self.batch_size, minimum=1, maximum=16)
        self._integer("workers", self.workers, minimum=0, maximum=16)
        self._integer("canvas_size", self.canvas_size, minimum=64, maximum=4096)
        self._number("mag_ratio", self.mag_ratio, minimum=0.5, maximum=3.0)
        self._number(
            "contrast_threshold",
            self.contrast_threshold,
            minimum=0.0,
            maximum=1.0,
        )
        self._number(
            "adjust_contrast",
            self.adjust_contrast,
            minimum=0.0,
            maximum=1.0,
        )
        if self.preprocessing not in {"none", "grayscale", "clahe"}:
            raise ValueError("preprocessing must be none, grayscale, or clahe")

    @staticmethod
    def _integer(name: str, value: int, *, minimum: int, maximum: int) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be between {minimum} and {maximum}")

    @staticmethod
    def _number(name: str, value: float, *, minimum: float, maximum: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be between {minimum} and {maximum}")


OCR_PROFILES = MappingProxyType(
    {
        "fast": OCRProfile(
            decoder="greedy",
            beam_width=1,
            batch_size=1,
            workers=0,
            canvas_size=1280,
            mag_ratio=1.0,
            contrast_threshold=0.1,
            adjust_contrast=0.5,
            preprocessing="none",
        ),
        "balanced": OCRProfile(
            decoder="greedy",
            beam_width=1,
            batch_size=1,
            workers=0,
            canvas_size=1920,
            mag_ratio=1.25,
            contrast_threshold=0.1,
            adjust_contrast=0.5,
            preprocessing="grayscale",
        ),
        "accurate": OCRProfile(
            decoder="beamsearch",
            beam_width=5,
            batch_size=1,
            workers=0,
            canvas_size=2560,
            mag_ratio=1.5,
            contrast_threshold=0.1,
            adjust_contrast=0.7,
            preprocessing="clahe",
        ),
    }
)
DEFAULT_OCR_PROFILE = "balanced"


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    image: ImageArray
    original_size: tuple[int, int]
    scale_x: float
    scale_y: float


def get_ocr_profile(profile: str | OCRProfile) -> OCRProfile:
    if isinstance(profile, OCRProfile):
        return profile
    try:
        return OCR_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"unknown OCR profile: {profile}") from error


def preprocess_image(image: ImageArray, profile: OCRProfile) -> PreprocessedImage:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("image must be a non-empty uint8 NumPy array")
    if image.ndim not in {2, 3} or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("image must be grayscale or BGR with three channels")

    height, width = image.shape[:2]
    processed: ImageArray = image
    if profile.preprocessing in {"grayscale", "clahe"} and image.ndim == 3:
        processed = cast(ImageArray, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if profile.preprocessing == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = cast(ImageArray, clahe.apply(processed))

    if profile.mag_ratio != 1.0:
        target = (
            max(1, round(width * profile.mag_ratio)),
            max(1, round(height * profile.mag_ratio)),
        )
        processed = cast(
            ImageArray,
            cv2.resize(processed, target, interpolation=cv2.INTER_CUBIC),
        )

    processed_height, processed_width = processed.shape[:2]
    return PreprocessedImage(
        image=processed,
        original_size=(width, height),
        scale_x=processed_width / width,
        scale_y=processed_height / height,
    )


__all__ = [
    "DEFAULT_OCR_PROFILE",
    "OCR_PROFILES",
    "OCRProfile",
    "PreprocessedImage",
    "get_ocr_profile",
    "preprocess_image",
]
