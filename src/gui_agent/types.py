from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int

    def __post_init__(self) -> None:
        if isinstance(self.x, bool) or not isinstance(self.x, int):
            raise ValueError("x must be an integer")
        if isinstance(self.y, bool) or not isinstance(self.y, int):
            raise ValueError("y must be an integer")


@dataclass(frozen=True, slots=True)
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def contains(self, point: Point) -> bool:
        return self.left <= point.x < self.right and self.top <= point.y < self.bottom

    def contains_region(self, other: "ScreenRegion") -> bool:
        return (
            self.left <= other.left
            and self.top <= other.top
            and other.right <= self.right
            and other.bottom <= self.bottom
        )


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("bounding box must have positive width and height")

    @property
    def center(self) -> Point:
        return Point((self.left + self.right) // 2, (self.top + self.bottom) // 2)


@dataclass(frozen=True, slots=True)
class ScreenshotResult:
    image: ImageArray
    monitor_index: int | None
    captured_at: datetime
    origin: Point

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray) or self.image.dtype != np.uint8:
            raise ValueError("image must be a uint8 NumPy array")
        if self.image.ndim != 3 or self.image.shape[2] != 3 or 0 in self.image.shape[:2]:
            raise ValueError("image must be a non-empty BGR array with shape (H, W, 3)")
        if self.monitor_index is not None and self.monitor_index < 1:
            raise ValueError("monitor_index must be at least 1 or None")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True, slots=True)
class OCRDetection:
    text: str
    confidence: float
    box: BoundingBox

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def center(self) -> Point:
        return self.box.center
