"""Expose the dataset normalization CLI as a directly executable script."""

from gui_agent.datasets.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
