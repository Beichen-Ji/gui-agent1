"""Stable fingerprints for in-memory image arrays."""

import hashlib

import numpy as np

from gui_agent.types import ImageArray


def image_fingerprint(image: ImageArray) -> str:
    """Hash image shape, data type, and bytes in a stable order."""
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()
