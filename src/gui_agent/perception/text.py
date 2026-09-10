"""Text normalization shared by perception and agent verification."""


def normalize_text(value: str) -> str:
    """Collapse whitespace and case-fold text for stable comparisons."""
    return " ".join(value.split()).casefold()
