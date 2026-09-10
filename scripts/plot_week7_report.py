from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the Week 7 evaluation reports.")
    parser.add_argument("--input", type=Path, required=True, help="Evaluation report root")
    parser.add_argument("--output", type=Path, required=True, help="Figure output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from gui_agent.evaluation.plots import load_report_series, save_week7_figures

        reports = load_report_series(args.input)
        paths = save_week7_figures(reports, args.output)
    except (ImportError, ValueError, OSError) as error:
        _parser().error(str(error))
    print(
        json.dumps(
            {"report_count": len(reports), "figure_count": len(paths), "output": str(args.output)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
