from pathlib import Path

import cv2
import pytest

from gui_agent.perception.capture import (
    CaptureError,
    InvalidMonitorError,
    InvalidRegionError,
    ScreenCapture,
)
from gui_agent.types import Point, ScreenRegion

from .conftest import FakeMSS


def capture_for(fake_mss: FakeMSS) -> ScreenCapture:
    return ScreenCapture(mss_factory=lambda: fake_mss)


def test_capture_monitor_converts_bgra_and_keeps_absolute_origin(fake_mss: FakeMSS) -> None:
    result = capture_for(fake_mss).capture_monitor(2)

    assert result.monitor_index == 2
    assert result.origin == Point(-3, 0)
    assert result.image.shape == (2, 3, 3)
    assert result.image[0, 0].tolist() == [10, 20, 30]
    assert fake_mss.requests == [{"left": -3, "top": 0, "width": 3, "height": 2}]


def test_capture_reports_virtual_and_physical_monitor_bounds(fake_mss: FakeMSS) -> None:
    capture = capture_for(fake_mss)

    assert capture.virtual_bounds() == ScreenRegion(-3, 0, 8, 4)
    assert capture.list_monitors() == (
        ScreenRegion(0, 0, 5, 4),
        ScreenRegion(-3, 0, 3, 2),
    )


@pytest.mark.parametrize("monitor_index", [0, 3])
def test_capture_rejects_invalid_monitor(fake_mss: FakeMSS, monitor_index: int) -> None:
    with pytest.raises(InvalidMonitorError, match="between 1 and 2"):
        capture_for(fake_mss).capture_monitor(monitor_index)


def test_capture_region_supports_negative_absolute_coordinates(fake_mss: FakeMSS) -> None:
    result = capture_for(fake_mss).capture_region(ScreenRegion(-2, 1, 2, 2))

    assert result.monitor_index is None
    assert result.origin == Point(-2, 1)
    assert result.image.shape == (2, 2, 3)
    assert fake_mss.requests == [{"left": -2, "top": 1, "width": 2, "height": 2}]


@pytest.mark.parametrize(
    "region",
    [ScreenRegion(-4, 0, 1, 1), ScreenRegion(4, 0, 2, 1), ScreenRegion(0, 3, 1, 2)],
)
def test_capture_rejects_region_outside_virtual_desktop(
    fake_mss: FakeMSS,
    region: ScreenRegion,
) -> None:
    with pytest.raises(InvalidRegionError, match="outside"):
        capture_for(fake_mss).capture_region(region)


def test_capture_saves_only_when_path_is_explicit(fake_mss: FakeMSS, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "capture.png"
    capture = capture_for(fake_mss)

    capture.capture_monitor(1)
    assert not output.exists()

    result = capture.capture_monitor(1, save_path=output)
    saved = cv2.imread(str(output), cv2.IMREAD_COLOR)

    assert output.is_file()
    assert saved is not None
    assert saved.shape == result.image.shape


def test_capture_rejects_invalid_mss_frame(fake_mss: FakeMSS) -> None:
    fake_mss.invalid_frame = True

    with pytest.raises(CaptureError, match="BGRA"):
        capture_for(fake_mss).capture_monitor(1)


def test_capture_wraps_desktop_grab_error(fake_mss: FakeMSS) -> None:
    fake_mss.grab_error = OSError("desktop unavailable")

    with pytest.raises(CaptureError, match="capture") as captured:
        capture_for(fake_mss).capture_monitor(1)

    assert isinstance(captured.value.__cause__, OSError)


def test_capture_reports_save_failure(
    fake_mss: FakeMSS,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)

    with pytest.raises(CaptureError, match="failed to save"):
        capture_for(fake_mss).capture_monitor(1, save_path=tmp_path / "capture.png")
