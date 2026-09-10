"""Lazy loaders for optional multimodal model dependencies."""

from collections.abc import Callable
from typing import cast


def load_processor(model_name: str, **kwargs: object) -> object:
    """Load a processor without importing Transformers at module import time."""
    from transformers import AutoProcessor

    loader = cast(Callable[..., object], AutoProcessor.from_pretrained)
    return loader(model_name, **kwargs)


def load_multimodal_model(model_name: str, **kwargs: object) -> object:
    """Load a multimodal model without importing Transformers eagerly."""
    from transformers import AutoModelForMultimodalLM

    loader = cast(Callable[..., object], AutoModelForMultimodalLM.from_pretrained)
    return loader(model_name, **kwargs)
