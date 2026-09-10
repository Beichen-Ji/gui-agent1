import argparse

import pytest

from gui_agent.cli_args import parse_integer_at_least


def test_parse_integer_at_least_accepts_its_boundary() -> None:
    assert parse_integer_at_least("0", minimum=0, message="bad") == 0
    assert parse_integer_at_least("1", minimum=1, message="bad") == 1


def test_parse_integer_at_least_uses_the_requested_argparse_error() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="exact message"):
        parse_integer_at_least("0", minimum=1, message="exact message")
    with pytest.raises(argparse.ArgumentTypeError, match="exact message"):
        parse_integer_at_least(
            "not-an-int",
            minimum=1,
            message="exact message",
            wrap_conversion_error=True,
        )


def test_parse_integer_at_least_can_preserve_raw_conversion_errors() -> None:
    with pytest.raises(ValueError):
        parse_integer_at_least("not-an-int", minimum=0, message="bad")
