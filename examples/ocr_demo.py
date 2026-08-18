import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2

from gui_agent.perception import EasyOCRBackend
from gui_agent.types import ImageArray, Point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EasyOCR on a caller-provided image without capturing the desktop.",
    )
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument(
        "--gpu",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="OCR device selection (default: auto)",
    )
    parser.add_argument(
        "--origin",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=(0, 0),
        help="absolute screen origin assigned to the image",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser


def _gpu_setting(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "cuda"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = cast(Path, args.image)
    if not image_path.is_file():
        print(f"OCR failed: image does not exist: {image_path}", file=sys.stderr)
        return 2
    raw_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_image is None:
        print(f"OCR failed: could not decode image: {image_path}", file=sys.stderr)
        return 2
    image = cast(ImageArray, raw_image)

    try:
        backend = EasyOCRBackend(gpu=_gpu_setting(args.gpu))
        origin = Point(*args.origin)
        detections = backend.recognize(
            image,
            origin=origin,
            min_confidence=args.min_confidence,
        )
    except (RuntimeError, ValueError) as error:
        print(f"OCR failed: {error}", file=sys.stderr)
        return 2

    print(f"detections: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        box = detection.box
        print(
            f"{index}: text={detection.text!r} confidence={detection.confidence:.2f} "
            f"box=({box.left}, {box.top}, {box.right}, {box.bottom}) "
            f"center=({detection.center.x}, {detection.center.y})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
