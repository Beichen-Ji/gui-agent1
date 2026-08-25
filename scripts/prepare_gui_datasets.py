import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from itertools import islice
from pathlib import Path

from gui_agent.datasets.mind2web import iter_mind2web
from gui_agent.datasets.pipeline import AdapterReport, write_dataset
from gui_agent.datasets.schema import NormalizedGUIRecord
from gui_agent.datasets.screenagent import iter_screenagent
from gui_agent.datasets.webarena import iter_webarena


def _positive_integer(value: str) -> int:
    converted = int(value)
    if converted < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return converted


def _local_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "unknown"


def _bounded_records(
    records: Iterable[NormalizedGUIRecord],
    limit: int,
) -> tuple[list[NormalizedGUIRecord], int]:
    probed = list(islice(records, limit + 1))
    has_more = len(probed) > limit
    return probed[:limit], int(has_more)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize public GUI agent datasets")
    subparsers = parser.add_subparsers(dest="source", required=True)

    screenagent = subparsers.add_parser("screenagent", help="Process ScreenAgent JSON")
    screenagent.add_argument("--input", type=Path, required=True)
    screenagent.add_argument("--split", default="train")

    mind2web = subparsers.add_parser("mind2web", help="Stream Mind2Web from Hugging Face")
    mind2web.add_argument("--dataset", default="osunlp/Mind2Web")
    mind2web.add_argument("--split", default="train")
    mind2web.add_argument("--stream", action="store_true")

    webarena = subparsers.add_parser("webarena", help="Process WebArena task configs")
    webarena.add_argument("--input", type=Path, required=True)

    for child in (screenagent, mind2web, webarena):
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--revision")
        child.add_argument("--limit", type=_positive_integer, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = AdapterReport()
    if args.source == "screenagent":
        revision = args.revision or _local_revision(args.input)
        records = iter_screenagent(
            args.input,
            split=args.split,
            source_revision=revision,
            report=report,
        )
    elif args.source == "webarena":
        revision = args.revision or _local_revision(args.input)
        records = iter_webarena(args.input, source_revision=revision)
    else:
        from datasets import load_dataset

        revision = args.revision or "main"
        rows = load_dataset(
            args.dataset,
            split=args.split,
            streaming=args.stream,
            revision=revision,
        )
        records = iter_mind2web(
            rows,
            split=args.split,
            source_revision=revision,
            report=report,
        )

    bounded, records_skipped = _bounded_records(records, args.limit)
    manifest = write_dataset(
        bounded,
        args.output,
        records_skipped=records_skipped + report.records_skipped,
    )
    if report.issues:
        print(
            f"Skipped {report.records_skipped} unsupported or malformed source actions.",
            file=sys.stderr,
        )
        for issue in report.issues:
            print(f"- {issue}", file=sys.stderr)
        if report.suppressed_issue_count:
            print(
                f"- ... {report.suppressed_issue_count} more issue(s) omitted",
                file=sys.stderr,
            )
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
