import argparse
import sys
from collections.abc import Callable, Sequence

from gui_agent.control import DesktopController
from gui_agent.types import Point

CONFIRMATION_PHRASE = "CLICK THIS CANDIDATE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a dry-run click, or explicitly confirm one live click.",
    )
    parser.add_argument("--x", type=int, required=True, help="absolute screen x coordinate")
    parser.add_argument("--y", type=int, required=True, help="absolute screen y coordinate")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="request one real click after an exact confirmation phrase",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        response = input_fn(
            f"Type {CONFIRMATION_PHRASE!r} to execute one real click: "
        )
        if response.strip() != CONFIRMATION_PHRASE:
            print("live click cancelled", file=sys.stderr)
            return 1

    try:
        controller = DesktopController(dry_run=not args.execute)
        record = controller.click(Point(args.x, args.y))
    except (RuntimeError, ValueError) as error:
        print(f"control failed: {error}", file=sys.stderr)
        return 2

    mode = "LIVE" if args.execute else "DRY-RUN"
    print(f"{mode}: {record.name} {record.parameters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
