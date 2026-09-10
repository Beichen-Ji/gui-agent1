"""Validated geometry, screenshot, and OCR value objects."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Point:
    """Represent an absolute desktop pixel coordinate."""

    x: int
    y: int

    def __post_init__(self) -> None:
        """Reject Boolean and non-integer coordinates."""
        if isinstance(self.x, bool) or not isinstance(self.x, int):
            raise ValueError("x must be an integer")
        if isinstance(self.y, bool) or not isinstance(self.y, int):
            raise ValueError("y must be an integer")


@dataclass(frozen=True, slots=True)
class ScreenRegion:
    """Represent a positive rectangular region in absolute desktop coordinates."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject empty or negative region dimensions."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @property
    def right(self) -> int:
        """Return the exclusive right edge."""
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive bottom edge."""
        return self.top + self.height

    def contains(self, point: Point) -> bool:
        """Return whether a point lies inside the half-open region."""
        return self.left <= point.x < self.right and self.top <= point.y < self.bottom

    def contains_region(self, other: "ScreenRegion") -> bool:
        """Return whether another region is fully enclosed."""
        return (
            self.left <= other.left
            and self.top <= other.top
            and other.right <= self.right
            and other.bottom <= self.bottom
        )


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Represent a positive half-open OCR bounding box."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        """Reject boxes without positive width and height."""
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("bounding box must have positive width and height")

    @property
    def center(self) -> Point:
        """Return the integer center point of the box."""
        return Point((self.left + self.right) // 2, (self.top + self.bottom) // 2)


@dataclass(frozen=True, slots=True)
class ScreenshotResult:
    """Store one timezone-stamped, uint8 BGR desktop capture."""

    image: ImageArray
    monitor_index: int | None
    captured_at: datetime
    origin: Point

    def __post_init__(self) -> None:
        """Validate image layout, monitor identity, and timestamp awareness."""
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
        """Return the image width in pixels."""
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """Return the image height in pixels."""
        return int(self.image.shape[0])


@dataclass(frozen=True, slots=True)
class OCRDetection:
    """Associate normalized OCR confidence and text with an absolute box."""

    text: str
    confidence: float
    box: BoundingBox

    def __post_init__(self) -> None:
        """Reject blank text and invalid confidence values."""
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def center(self) -> Point:
        """Return the center of the detection box."""
        return self.box.center
