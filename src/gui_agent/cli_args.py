"""Shared argument-parsing primitives for command-line interfaces."""

import argparse


def parse_integer_at_least(
    value: str,
    *,
    minimum: int,
    message: str,
    wrap_conversion_error: bool = False,
) -> int:
    """Parse an integer whose value must meet a caller-defined lower bound."""
    try:
        converted = int(value)
    except ValueError as error:
        if wrap_conversion_error:
            raise argparse.ArgumentTypeError(message) from error
        raise
    if converted < minimum:
        raise argparse.ArgumentTypeError(message)
    return converted
