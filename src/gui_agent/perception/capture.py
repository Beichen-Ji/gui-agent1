from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast

import cv2
import mss
import numpy as np

from gui_agent.types import ImageArray, Point, ScreenRegion, ScreenshotResult


class CaptureError(RuntimeError):
    """A desktop capture operation could not be completed."""


class InvalidMonitorError(CaptureError, ValueError):
    """A physical monitor index is outside the available range."""


class InvalidRegionError(CaptureError, ValueError):
    """A requested region is outside the virtual desktop."""


MonitorMapping = Mapping[str, int]


class MSSSession(Protocol):
    monitors: Sequence[MonitorMapping]

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def grab(self, monitor: MonitorMapping) -> object: ...


MSSFactory = Callable[[], MSSSession]


def _default_mss_factory() -> MSSSession:
    return cast(MSSSession, mss.mss())


def _region_from_monitor(monitor: MonitorMapping) -> ScreenRegion:
    try:
        return ScreenRegion(
            left=monitor["left"],
            top=monitor["top"],
            width=monitor["width"],
            height=monitor["height"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CaptureError("MSS returned invalid monitor metadata") from error


def _save_if_requested(image: ImageArray, save_path: Path | None) -> None:
    if save_path is None:
        return
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        saved = cv2.imwrite(str(save_path), image)
    except (OSError, cv2.error) as error:
        raise CaptureError(f"failed to save screenshot to {save_path}") from error
    if not saved:
        raise CaptureError(f"failed to save screenshot to {save_path}")


class ScreenCapture:
    """Capture physical monitors or absolute virtual-desktop regions with MSS."""

    def __init__(self, mss_factory: MSSFactory | None = None) -> None:
        self._mss_factory = mss_factory or _default_mss_factory

    def virtual_bounds(self) -> ScreenRegion:
        with self._mss_factory() as session:
            if not session.monitors:
                raise CaptureError("MSS did not report a virtual desktop")
            return _region_from_monitor(session.monitors[0])

    def list_monitors(self) -> tuple[ScreenRegion, ...]:
        with self._mss_factory() as session:
            if not session.monitors:
                raise CaptureError("MSS did not report a virtual desktop")
            return tuple(_region_from_monitor(item) for item in session.monitors[1:])

    def capture_monitor(
        self,
        monitor_index: int = 1,
        *,
        save_path: Path | None = None,
    ) -> ScreenshotResult:
        with self._mss_factory() as session:
            maximum = len(session.monitors) - 1
            if isinstance(monitor_index, bool) or not 1 <= monitor_index <= maximum:
                raise InvalidMonitorError(
                    f"monitor_index must be between 1 and {maximum}, got {monitor_index}"
                )
            region = _region_from_monitor(session.monitors[monitor_index])
            result = self._grab(session, region, monitor_index)
        _save_if_requested(result.image, save_path)
        return result

    def capture_region(
        self,
        region: ScreenRegion,
        *,
        save_path: Path | None = None,
    ) -> ScreenshotResult:
        with self._mss_factory() as session:
            if not session.monitors:
                raise CaptureError("MSS did not report a virtual desktop")
            virtual_bounds = _region_from_monitor(session.monitors[0])
            if not virtual_bounds.contains_region(region):
                raise InvalidRegionError(f"region {region} is outside {virtual_bounds}")
            result = self._grab(session, region, None)
        _save_if_requested(result.image, save_path)
        return result

    @staticmethod
    def _grab(
        session: MSSSession,
        region: ScreenRegion,
        monitor_index: int | None,
    ) -> ScreenshotResult:
        request = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        try:
            raw = np.asarray(session.grab(request), dtype=np.uint8)
        except Exception as error:
            raise CaptureError(f"failed to capture desktop region {region}") from error
        if raw.ndim != 3 or raw.shape[2] < 3:
            raise CaptureError("MSS returned an invalid BGRA image")
        return ScreenshotResult(
            image=raw[:, :, :3].copy(),
            monitor_index=monitor_index,
            captured_at=datetime.now(UTC),
            origin=Point(region.left, region.top),
        )
