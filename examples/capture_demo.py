import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from gui_agent.perception import CaptureError, ScreenCapture
from gui_agent.types import ScreenRegion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one monitor or an absolute virtual-desktop region.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--monitor", type=int, help="physical monitor index (default: 1)")
    source.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="absolute virtual-desktop region",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional output image; no file is written when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    capture = ScreenCapture()
    try:
        if args.region is None:
            result = capture.capture_monitor(args.monitor or 1, save_path=args.output)
        else:
            result = capture.capture_region(
                ScreenRegion(*args.region),
                save_path=args.output,
            )
    except (CaptureError, ValueError, OSError) as error:
        print(f"capture failed: {error}", file=sys.stderr)
        return 2

    print(
        f"captured {result.width}x{result.height} at "
        f"({result.origin.x}, {result.origin.y}); monitor={result.monitor_index}"
    )
    if args.output is not None:
        print(f"saved: {args.output}")
    else:
        print("not saved (pass --output to write an image)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
