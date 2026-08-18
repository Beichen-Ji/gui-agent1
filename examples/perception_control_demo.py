import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import cv2

from gui_agent.control import DesktopController
from gui_agent.perception import (
    CaptureError,
    EasyOCRBackend,
    MatchMode,
    ScreenCapture,
    annotate_detections,
    find_text,
)
from gui_agent.types import ScreenRegion

CONFIRMATION_PHRASE = "CLICK THIS CANDIDATE"


def live_click_confirmed(
    *,
    execute: bool,
    candidate_count: int,
    input_fn: Callable[[str], str] = input,
) -> bool:
    if not execute or candidate_count != 1:
        return False
    response = input_fn(f"Type {CONFIRMATION_PHRASE!r} to execute one real click: ")
    return response.strip() == CONFIRMATION_PHRASE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture, OCR, locate text, and safely plan or confirm one click.",
    )
    parser.add_argument("query", help="text to locate")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--monitor", type=int, help="physical monitor index (default: 1)")
    source.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="absolute virtual-desktop region",
    )
    parser.add_argument("--contains", action="store_true", help="use substring matching")
    parser.add_argument("--ignore-case", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--confidence-cutoff", type=float, default=0.8)
    parser.add_argument(
        "--gpu",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        help="optional annotated output; no image is written when omitted",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="request one real click after all safety gates pass",
    )
    return parser


def _gpu_setting(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "cuda"


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        capture = ScreenCapture()
        if args.region is None:
            screenshot = capture.capture_monitor(args.monitor or 1)
        else:
            screenshot = capture.capture_region(ScreenRegion(*args.region))

        backend = EasyOCRBackend(gpu=_gpu_setting(args.gpu))
        detections = backend.recognize(
            screenshot.image,
            origin=screenshot.origin,
            min_confidence=args.min_confidence,
        )
        matches = find_text(
            detections,
            args.query,
            mode=MatchMode.CONTAINS if args.contains else MatchMode.EXACT,
            case_sensitive=not args.ignore_case,
        )

        print(f"candidates: {len(matches)}")
        for index, detection in enumerate(matches, start=1):
            box = detection.box
            print(
                f"{index}: text={detection.text!r} confidence={detection.confidence:.2f} "
                f"box=({box.left}, {box.top}, {box.right}, {box.bottom}) "
                f"center=({detection.center.x}, {detection.center.y})"
            )

        annotation_path = cast(Path | None, args.annotation)
        if annotation_path is not None:
            annotated = annotate_detections(
                screenshot.image,
                detections,
                confidence_cutoff=args.confidence_cutoff,
                origin=screenshot.origin,
            )
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(annotation_path), annotated):
                raise OSError(f"failed to save annotation to {annotation_path}")
            print(f"annotation saved: {annotation_path}")

        confirmed = live_click_confirmed(
            execute=args.execute,
            candidate_count=len(matches),
            input_fn=input_fn,
        )
        if len(matches) != 1:
            if args.execute:
                print("live click refused: exactly one candidate is required", file=sys.stderr)
                return 1
            return 0
        if args.execute and not confirmed:
            print("live click cancelled", file=sys.stderr)
            return 1

        controller = DesktopController(dry_run=not args.execute)
        record = controller.click(matches[0].center)
        mode = "LIVE" if args.execute else "DRY-RUN"
        print(f"{mode}: {record.name} {record.parameters}")
        return 0
    except (CaptureError, RuntimeError, ValueError, OSError) as error:
        print(f"perception/control failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
