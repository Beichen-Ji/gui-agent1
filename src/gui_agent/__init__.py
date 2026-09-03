__version__ = "0.2.0"


def main() -> None:
    from gui_agent.cli import main as cli_main

    raise SystemExit(cli_main())
