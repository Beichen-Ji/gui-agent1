"""Safe desktop perception, control, planning, training, and evaluation."""

__version__ = "0.2.0"


def main() -> None:
    """Run the installed ``gui-agent`` command-line entry point."""
    from gui_agent.cli import main as cli_main

    raise SystemExit(cli_main())
