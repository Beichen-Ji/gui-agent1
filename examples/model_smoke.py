"""Expose the synthetic model smoke test as an executable example."""

from gui_agent.agent.smoke import build_parser, main, synthetic_observation

__all__ = ["build_parser", "main", "synthetic_observation"]


if __name__ == "__main__":
    raise SystemExit(main())
