import os
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gui_agent.types import ImageArray, OCRDetection, Point

HIGH_CONFIDENCE_BGR = (0, 180, 0)
LOW_CONFIDENCE_BGR = (0, 165, 255)
DEFAULT_ORIGIN = Point(0, 0)


class MatchMode(StrEnum):
    EXACT = "exact"
    CONTAINS = "contains"


def find_text(
    detections: Sequence[OCRDetection],
    query: str,
    *,
    mode: MatchMode = MatchMode.EXACT,
    case_sensitive: bool = True,
) -> list[OCRDetection]:
    if not query:
        raise ValueError("query must not be empty")

    expected = query if case_sensitive else query.casefold()
    matches: list[OCRDetection] = []
    for detection in detections:
        actual = detection.text if case_sensitive else detection.text.casefold()
        if (mode is MatchMode.EXACT and actual == expected) or (
            mode is MatchMode.CONTAINS and expected in actual
        ):
            matches.append(detection)
    return matches


def _annotation_font(
    font_path: Path | None,
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if font_path is not None:
        if not font_path.is_file():
            raise ValueError(f"font_path does not exist: {font_path}")
        return ImageFont.truetype(str(font_path), 16)

    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for candidate in (windows_fonts / "msyh.ttc", windows_fonts / "simhei.ttf"):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), 16)
    return ImageFont.load_default()


def annotate_detections(
    image: ImageArray,
    detections: Sequence[OCRDetection],
    *,
    confidence_cutoff: float = 0.8,
    font_path: Path | None = None,
    origin: Point = DEFAULT_ORIGIN,
) -> ImageArray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or 0 in image.shape[:2]
    ):
        raise ValueError("image must be a non-empty uint8 BGR array")
    if not 0.0 <= confidence_cutoff <= 1.0:
        raise ValueError("confidence_cutoff must be between 0 and 1")

    output = image.copy()
    if not detections:
        return output

    colors: list[tuple[int, int, int]] = []
    for detection in detections:
        color = (
            HIGH_CONFIDENCE_BGR
            if detection.confidence >= confidence_cutoff
            else LOW_CONFIDENCE_BGR
        )
        colors.append(color)
        box = detection.box
        left = box.left - origin.x
        top = box.top - origin.y
        right = box.right - origin.x
        bottom = box.bottom - origin.y
        center_x = detection.center.x - origin.x
        center_y = detection.center.y - origin.y
        cv2.rectangle(output, (left, top), (right - 1, bottom - 1), color, 2)
        cv2.circle(output, (center_x, center_y), 3, color, -1)

    rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    font = _annotation_font(font_path)
    for detection, bgr in zip(detections, colors, strict=True):
        label = f"{detection.text} {detection.confidence:.2f}"
        rgb_color = (bgr[2], bgr[1], bgr[0])
        label_position = (
            detection.box.left - origin.x,
            max(0, detection.box.top - origin.y - 18),
        )
        draw.text(label_position, label, fill=rgb_color, font=font)

    return cast(
        ImageArray,
        cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR),
    )
