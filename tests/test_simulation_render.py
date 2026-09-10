from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from gui_agent.simulation.apps import APP_IDS
from gui_agent.simulation.render import layout_desktop, render_desktop
from gui_agent.simulation.state import TestbedState
from gui_agent.types import BoundingBox


def _overlaps(first: BoundingBox, second: BoundingBox) -> bool:
    return not (
        first.right <= second.left
        or second.right <= first.left
        or first.bottom <= second.top
        or second.bottom <= first.top
    )


def test_renderer_is_byte_deterministic_and_returns_bgr_uint8(tmp_path: Path) -> None:
    state = TestbedState(tmp_path / "desktop")
    state.search("deterministic pixels")

    first_image, first_hitboxes = render_desktop(state, (1280, 720))
    second_image, second_hitboxes = render_desktop(state, (1280, 720))

    assert first_image.dtype == np.uint8
    assert first_image.shape == (720, 1280, 3)
    assert first_image.tobytes() == second_image.tobytes()
    assert first_hitboxes == second_hitboxes
    assert first_image[0, 0].tolist() == [245, 243, 240]


@pytest.mark.parametrize("app_id", APP_IDS)
@pytest.mark.parametrize("canvas", [(1280, 720), (1920, 1080), (2560, 1440)])
def test_layout_has_scaled_non_overlapping_controls_and_uncropped_labels(
    tmp_path: Path,
    app_id: str,
    canvas: tuple[int, int],
) -> None:
    state = TestbedState(tmp_path / f"desktop-{app_id}-{canvas[0]}")
    state.activate_app(app_id)  # type: ignore[arg-type]

    layout = layout_desktop(state, canvas)
    image, hitboxes = render_desktop(state, canvas)

    assert image.shape == (canvas[1], canvas[0], 3)
    assert set(hitboxes) == {control.id for control in layout}
    for first, second in combinations(hitboxes.values(), 2):
        assert not _overlaps(first, second)
    for control in layout:
        assert control.box.left <= control.label_box.left
        assert control.box.top <= control.label_box.top
        assert control.label_box.right <= control.box.right
        assert control.label_box.bottom <= control.box.bottom


def test_hitboxes_scale_with_canvas_size(tmp_path: Path) -> None:
    state = TestbedState(tmp_path / "desktop")
    _, small = render_desktop(state, (1280, 720))
    _, large = render_desktop(state, (2560, 1440))

    for control_id, small_box in small.items():
        large_box = large[control_id]
        assert large_box.left == pytest.approx(small_box.left * 2, abs=1)
        assert large_box.top == pytest.approx(small_box.top * 2, abs=1)
        assert large_box.right == pytest.approx(small_box.right * 2, abs=1)
        assert large_box.bottom == pytest.approx(small_box.bottom * 2, abs=1)
