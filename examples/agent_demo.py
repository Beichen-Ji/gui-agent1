import sys
from collections.abc import Sequence

from gui_agent.cli import main as cli_main


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
