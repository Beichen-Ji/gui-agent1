from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import pytest
from numpy.typing import NDArray


class FakeMSS:
    def __init__(self) -> None:
        self.monitors: Sequence[Mapping[str, int]] = [
            {"left": -3, "top": 0, "width": 8, "height": 4},
            {"left": 0, "top": 0, "width": 5, "height": 4},
            {"left": -3, "top": 0, "width": 3, "height": 2},
        ]
        self.requests: list[dict[str, int]] = []
        self.invalid_frame = False
        self.grab_error: Exception | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None

    def grab(self, monitor: Mapping[str, int]) -> NDArray[np.uint8]:
        if self.grab_error is not None:
            raise self.grab_error
        request = dict(monitor)
        self.requests.append(request)
        if self.invalid_frame:
            return np.zeros((request["height"], request["width"]), dtype=np.uint8)
        frame = np.empty((request["height"], request["width"], 4), dtype=np.uint8)
        frame[:, :] = [10, 20, 30, 255]
        return frame


@pytest.fixture
def fake_mss() -> FakeMSS:
    return FakeMSS()
