import numpy as np

from gui_agent.perception.fingerprint import image_fingerprint


def test_image_fingerprint_tracks_shape_dtype_and_bytes() -> None:
    base = np.zeros((2, 3, 3), dtype=np.uint8)
    changed = base.copy()
    changed[0, 0, 0] = 1

    assert image_fingerprint(base) == image_fingerprint(base.copy())
    assert image_fingerprint(base) != image_fingerprint(changed)
    assert image_fingerprint(base) != image_fingerprint(base.astype(np.uint16))
    assert image_fingerprint(base) != image_fingerprint(base.reshape(3, 2, 3))
