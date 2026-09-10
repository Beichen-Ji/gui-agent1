import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from gui_agent.cli_args import parse_integer_at_least
from gui_agent.perception.benchmark import benchmark_profiles
from gui_agent.perception.ocr import EasyOCRBackend
from gui_agent.perception.preprocessing import OCR_PROFILES


def _non_negative_integer(value: str) -> int:
    return parse_integer_at_least(
        value,
        minimum=0,
        message="must be non-negative",
    )


def _positive_integer(value: str) -> int:
    return parse_integer_at_least(
        value,
        minimum=1,
        message="must be positive",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark safe OCR profiles")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(OCR_PROFILES),
        required=True,
    )
    parser.add_argument("--warmup", type=_non_negative_integer, default=2)
    parser.add_argument("--runs", type=_positive_integer, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = benchmark_profiles(
        args.manifest,
        tuple(args.profiles),
        warmup=args.warmup,
        runs=args.runs,
        backend_factory=lambda profile: EasyOCRBackend(profile=profile),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
