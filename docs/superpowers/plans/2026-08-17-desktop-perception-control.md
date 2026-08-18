# Desktop Perception and Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the Week 2 desktop perception and control foundation, starting from the Python 3.11 environment work in section 1.6 and publishing the result as a GitHub draft PR.

**Architecture:** Preserve the remote repository's `uv` project and its `perception/` and `control/` packages. Capture normalizes MSS data to BGR screenshots with absolute virtual-desktop origins; OCR normalizes EasyOCR output to absolute boxes; localization returns candidates and annotated image copies; the controller is the only layer that may call PyAutoGUI and defaults to dry-run.

**Tech Stack:** Python 3.11, uv, NumPy, MSS, OpenCV, Pillow, EasyOCR, PyTorch CUDA 12.8, PyAutoGUI, pynput, pytest, pytest-cov, Ruff, mypy, GitHub Actions.

## Global Constraints

- Use Python `>=3.11,<3.12`; never run project commands with the active Python 3.13 interpreter.
- Develop on Windows and verify the NVIDIA GeForce RTX 5070 Ti with driver 595.95.
- Use EasyOCR with `("ch_sim", "en")`; keep the `OCRBackend` interface replaceable.
- Store images internally as non-empty `numpy.uint8` BGR arrays.
- Express all public boxes and points as absolute virtual-desktop physical pixels; negative screen coordinates are valid.
- Default every controller and demonstration path to `dry_run=True`.
- Ordinary tests and CI must not capture the real desktop, download OCR models, move the mouse, click, type, scroll, or drag.
- Do not commit PDFs, screenshots, OCR/model caches, weights, logs, `.env`, or other runtime artifacts.
- Use strict Red-Green-Refactor for every production behavior: write one behavioral test, observe the expected failure, then write the minimum implementation.
- Work only on `agent/week2-perception-control`, based on remote `master`.

---

## File Map

- Modify `pyproject.toml`: Python floor/ceiling, runtime dependencies, OCR extra, development tools, PyTorch index, pytest/Ruff/mypy settings, package version.
- Regenerate `uv.lock`: exact cross-platform dependency resolution.
- Modify `.gitignore`: local OCR caches, screenshots, generated annotations, mypy cache, uv/Python artifacts.
- Modify `src/gui_agent/__init__.py`: package version and lightweight CLI entry.
- Create `src/gui_agent/types.py`: common image and coordinate data types.
- Create `src/gui_agent/perception/capture.py`: MSS capture and validation.
- Create `src/gui_agent/perception/ocr.py`: OCR protocol and EasyOCR implementation.
- Create `src/gui_agent/perception/localization.py`: text matching and annotations.
- Modify `src/gui_agent/perception/__init__.py`: stable perception exports.
- Create `src/gui_agent/control/controller.py`: safe desktop action API and PyAutoGUI adapter.
- Modify `src/gui_agent/control/__init__.py`: stable control exports.
- Create `tests/conftest.py`: deterministic MSS, OCR, desktop and image fakes.
- Create `tests/test_types.py`: type validation and derived coordinates.
- Create `tests/test_capture.py`: capture behavior without desktop access.
- Create `tests/test_ocr.py`: OCR normalization and failures without model downloads.
- Create `tests/test_localization.py`: matching and non-destructive drawing.
- Create `tests/test_control.py`: dry-run and injected live behavior.
- Create `tests/test_examples.py`: safety gates used by examples.
- Create `examples/capture_demo.py`, `examples/ocr_demo.py`, `examples/control_demo.py`, `examples/perception_control_demo.py`: manual, privacy-aware demos.
- Create `.github/workflows/ci.yml`: Windows Python 3.11 checks excluding integrations.
- Modify `README.md`: project usage, architecture, safety, commands and progress.
- Create `docs/setup/windows-setup.md`: reproducible setup plus measured local output.
- Create `docs/test-reports/week2-test-report.md`: measured tests, coverage, timings and limitations.

---

### Task 1: Reproducible Python 3.11 and CUDA Environment

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Modify: `src/gui_agent/__init__.py`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: remote `master` configuration and `.python-version` containing `3.11`.
- Produces: `uv run` commands backed by `.venv`, project version `0.2.0`, `dev` group, `ocr` extra, and a CUDA 12.8 PyTorch source.

- [ ] **Step 1: Install uv and provision Python 3.11**

Run:

```powershell
winget install --id astral-sh.uv --exact --accept-package-agreements --accept-source-agreements
uv python install 3.11
uv venv --python 3.11
uv run python --version
```

Expected: the final command prints `Python 3.11.x`, and `.venv` is inside the repository.

- [ ] **Step 2: Update project metadata and dependency boundaries**

Set the relevant `pyproject.toml` structure to:

```toml
[project]
name = "gui-agent"
version = "0.2.0"
description = "Safe desktop perception and control foundations for a GUI agent"
requires-python = ">=3.11,<3.12"
dependencies = [
    "mss>=10.1,<11",
    "numpy>=2,<3",
    "opencv-python-headless>=4.11,<5",
    "pillow>=11,<13",
    "pyautogui>=0.9.54,<1",
    "pynput>=1.8,<2",
]

[project.optional-dependencies]
ocr = [
    "easyocr>=1.7.2,<2",
    "torch>=2.7,<3",
    "torchvision>=0.22,<1",
]

[dependency-groups]
dev = [
    "mypy>=1.15,<2",
    "pytest>=8,<10",
    "pytest-cov>=6,<8",
    "ruff>=0.11,<1",
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
markers = [
    "integration: requires a real desktop, GPU, model download, or external resource",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
exclude = ["^\\.venv/"]

[[tool.mypy.overrides]]
module = ["cv2", "easyocr", "mss", "mss.*", "pyautogui", "pynput", "pynput.*"]
ignore_missing_imports = true
```

Keep the existing `uv_build` backend and update the author to `Beichen-Ji <bji7@uwo.ca>`.

- [ ] **Step 3: Add ignored local artifacts**

Extend `.gitignore` with:

```gitignore
.mypy_cache/
.python/
.EasyOCR/
screenshots/
annotations/
*.png
*.jpg
*.jpeg
```

Retain the tracked `artifacts/.gitkeep` exception and existing PDF/model/cache exclusions.

- [ ] **Step 4: Lock and install all local development dependencies**

Run:

```powershell
uv lock
uv sync --group dev --extra ocr
uv run python -c "import sys; assert sys.version_info[:2] == (3, 11); print(sys.version)"
uv run python -c "import cv2, easyocr, mss, numpy, PIL, pyautogui, pynput, torch; print('imports: ok'); print(torch.__version__); print(torch.version.cuda)"
```

Expected: imports succeed from `.venv`; PyTorch reports a CUDA 12.8 build.

- [ ] **Step 5: Verify the RTX 5070 Ti**

Run:

```powershell
uv run python -c "import torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print(torch.cuda.get_device_name(0)); print(round(p.total_memory/1024**3, 2))"
```

Expected: CUDA is available, the device name contains `RTX 5070 Ti`, and reported memory is approximately 16 GB.

- [ ] **Step 6: Update the lightweight package entry**

Set `src/gui_agent/__init__.py` to expose `__version__ = "0.2.0"` and make `main()` print the version plus a pointer to `README.md`, without importing OCR or desktop libraries at package import time.

- [ ] **Step 7: Verify metadata and commit**

Run:

```powershell
uv run python -c "import gui_agent; assert gui_agent.__version__ == '0.2.0'"
uv run ruff check src/gui_agent/__init__.py
git diff --check
git add .gitignore pyproject.toml uv.lock src/gui_agent/__init__.py
git commit -m "chore: configure Python 3.11 GPU environment"
```

---

### Task 2: Public Image and Coordinate Types

**Files:**
- Create: `src/gui_agent/types.py`
- Create: `tests/test_types.py`

**Interfaces:**
- Produces: `ImageArray`, `Point`, `ScreenRegion`, `BoundingBox`, `ScreenshotResult`, `OCRDetection`.
- Consumed by: capture, OCR, localization, control and all demos.

- [ ] **Step 1: Write failing tests for geometry and validation**

Create tests that use literal expectations:

```python
def test_region_derives_half_open_edges() -> None:
    region = ScreenRegion(left=-100, top=20, width=640, height=480)
    assert region.right == 540
    assert region.bottom == 500
    assert region.contains(Point(-100, 20))
    assert not region.contains(Point(540, 500))


def test_box_derives_integer_center() -> None:
    assert BoundingBox(10, 20, 31, 41).center == Point(20, 30)


def test_screenshot_rejects_non_uint8_image() -> None:
    image = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        ScreenshotResult(image=image, monitor_index=1, captured_at=UTC_NOW, origin=Point(0, 0))


def test_ocr_detection_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        OCRDetection("Save", 1.1, BoundingBox(0, 0, 10, 10))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_types.py -v`

Expected: collection fails because `gui_agent.types` does not exist.

- [ ] **Step 3: Implement the minimal immutable types**

Use frozen, slotted dataclasses. Implement the file with these concrete behaviors:

```python
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]

@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int

    def __post_init__(self) -> None:
        if isinstance(self.x, bool) or not isinstance(self.x, int):
            raise ValueError("x must be an integer")
        if isinstance(self.y, bool) or not isinstance(self.y, int):
            raise ValueError("y must be an integer")

@dataclass(frozen=True, slots=True)
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def contains(self, point: Point) -> bool:
        return self.left <= point.x < self.right and self.top <= point.y < self.bottom

    def contains_region(self, other: "ScreenRegion") -> bool:
        return (
            self.left <= other.left
            and self.top <= other.top
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("bounding box must have positive width and height")

    @property
    def center(self) -> Point:
        return Point((self.left + self.right) // 2, (self.top + self.bottom) // 2)

@dataclass(frozen=True, slots=True)
class ScreenshotResult:
    image: ImageArray
    monitor_index: int | None
    captured_at: datetime
    origin: Point

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray) or self.image.dtype != np.uint8:
            raise ValueError("image must be a uint8 NumPy array")
        if self.image.ndim != 3 or self.image.shape[2] != 3 or 0 in self.image.shape[:2]:
            raise ValueError("image must be a non-empty BGR array with shape (H, W, 3)")
        if self.monitor_index is not None and self.monitor_index < 1:
            raise ValueError("monitor_index must be at least 1 or None")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

@dataclass(frozen=True, slots=True)
class OCRDetection:
    text: str
    confidence: float
    box: BoundingBox

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def center(self) -> Point:
        return self.box.center
```

Validate positive region/box sizes, timezone-aware timestamps, non-empty `uint8` images with two or three dimensions, non-empty OCR text and confidence in `[0, 1]`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `uv run pytest tests/test_types.py -v`

Expected: all type tests pass.

- [ ] **Step 5: Run static checks and commit**

Run:

```powershell
uv run ruff check src/gui_agent/types.py tests/test_types.py
uv run mypy src/gui_agent/types.py
git add src/gui_agent/types.py tests/test_types.py
git commit -m "feat: define screen and OCR result types"
```

---

### Task 3: Cross-Platform Screen Capture

**Files:**
- Create: `src/gui_agent/perception/capture.py`
- Modify: `src/gui_agent/perception/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_capture.py`

**Interfaces:**
- Consumes: `Point`, `ScreenRegion`, `ScreenshotResult`.
- Produces: `CaptureError`, `InvalidMonitorError`, `InvalidRegionError`, `ScreenCapture` with `virtual_bounds()`, `list_monitors()`, `capture_monitor()` and `capture_region()`.

- [ ] **Step 1: Add a deterministic MSS fake and failing monitor test**

The fake exposes MSS-style monitor dictionaries and returns a literal BGRA array. Test:

```python
def test_capture_monitor_converts_bgra_and_keeps_absolute_origin(fake_mss_factory) -> None:
    capture = ScreenCapture(mss_factory=fake_mss_factory)
    result = capture.capture_monitor(2)
    assert result.monitor_index == 2
    assert result.origin == Point(-1280, 0)
    assert result.image.shape == (2, 3, 3)
    assert result.image[0, 0].tolist() == [10, 20, 30]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_capture.py::test_capture_monitor_converts_bgra_and_keeps_absolute_origin -v`

Expected: import failure because `perception.capture` is missing.

- [ ] **Step 3: Implement monitor capture minimally**

Implement the monitor path with this concrete structure:

```python
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import Protocol, cast

import mss
import numpy as np

from gui_agent.types import Point, ScreenRegion, ScreenshotResult

MonitorMapping = Mapping[str, int]

class MSSSession(Protocol):
    monitors: Sequence[MonitorMapping]

    def grab(self, monitor: MonitorMapping) -> object:
        raise NotImplementedError

MSSFactory = Callable[[], AbstractContextManager[MSSSession]]

def _default_mss_factory() -> AbstractContextManager[MSSSession]:
    return cast(AbstractContextManager[MSSSession], mss.mss())

def _region_from_monitor(monitor: MonitorMapping) -> ScreenRegion:
    return ScreenRegion(
        left=monitor["left"],
        top=monitor["top"],
        width=monitor["width"],
        height=monitor["height"],
    )

class ScreenCapture:
    def __init__(self, mss_factory: MSSFactory | None = None) -> None:
        self._mss_factory = mss_factory or _default_mss_factory

    def virtual_bounds(self) -> ScreenRegion:
        with self._mss_factory() as session:
            return _region_from_monitor(session.monitors[0])

    def list_monitors(self) -> tuple[ScreenRegion, ...]:
        with self._mss_factory() as session:
            return tuple(_region_from_monitor(item) for item in session.monitors[1:])

    def capture_monitor(self, monitor_index: int = 1) -> ScreenshotResult:
        with self._mss_factory() as session:
            if not 1 <= monitor_index < len(session.monitors):
                raise InvalidMonitorError(
                    f"monitor_index must be between 1 and {len(session.monitors) - 1}"
                )
            region = _region_from_monitor(session.monitors[monitor_index])
            return self._grab(session, region, monitor_index)

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
        raw = np.asarray(session.grab(request), dtype=np.uint8)
        if raw.ndim != 3 or raw.shape[2] < 3:
            raise CaptureError("MSS returned an invalid BGRA image")
        return ScreenshotResult(
            image=raw[:, :, :3].copy(),
            monitor_index=monitor_index,
            captured_at=datetime.now(timezone.utc),
            origin=Point(region.left, region.top),
        )
```

Define `CaptureError(RuntimeError)`, `InvalidMonitorError(CaptureError, ValueError)`, and later `InvalidRegionError(CaptureError, ValueError)` above the class.

- [ ] **Step 4: Verify monitor capture GREEN**

Run: `uv run pytest tests/test_capture.py::test_capture_monitor_converts_bgra_and_keeps_absolute_origin -v`

Expected: PASS.

- [ ] **Step 5: Add failing tests for region validation and explicit saving**

Cover a negative-origin valid region, a region extending one pixel beyond virtual bounds, monitor `0`, monitor `N+1`, and `save_path`. Assert `cv2.imread()` returns the expected dimensions and that no file exists when `save_path` is omitted.

- [ ] **Step 6: Run the new tests and verify RED**

Run: `uv run pytest tests/test_capture.py -v`

Expected: failures for missing `capture_region()` and save behavior.

- [ ] **Step 7: Implement region capture and explicit save**

Extend monitor capture with optional saving and add region capture:

```python
def _save_if_requested(image: ImageArray, save_path: Path | None) -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), image):
        raise CaptureError(f"failed to save screenshot to {save_path}")

def capture_monitor(
    self,
    monitor_index: int = 1,
    *,
    save_path: Path | None = None,
) -> ScreenshotResult:
    with self._mss_factory() as session:
        if not 1 <= monitor_index < len(session.monitors):
            raise InvalidMonitorError(
                f"monitor_index must be between 1 and {len(session.monitors) - 1}"
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
        virtual_bounds = _region_from_monitor(session.monitors[0])
        if not virtual_bounds.contains_region(region):
            raise InvalidRegionError(f"region {region} is outside {virtual_bounds}")
        result = self._grab(session, region, None)
    _save_if_requested(result.image, save_path)
    return result
```

Require the requested region to be completely inside `monitors[0]`. Create parent directories only for an explicit save; raise `CaptureError` if `cv2.imwrite()` returns false.

- [ ] **Step 8: Verify all capture behavior and commit**

Run:

```powershell
uv run pytest tests/test_capture.py -v
uv run ruff check src/gui_agent/perception tests/test_capture.py tests/conftest.py
uv run mypy src/gui_agent/perception/capture.py
git add src/gui_agent/perception tests/conftest.py tests/test_capture.py
git commit -m "feat: add cross-platform screen capture"
```

---

### Task 4: Pluggable EasyOCR Backend

**Files:**
- Create: `src/gui_agent/perception/ocr.py`
- Modify: `src/gui_agent/perception/__init__.py`
- Create: `tests/test_ocr.py`

**Interfaces:**
- Consumes: BGR/gray `ImageArray`, `Point`, `BoundingBox`, `OCRDetection`.
- Produces: runtime-checkable `OCRBackend` protocol and lazy `EasyOCRBackend.recognize(image, origin=Point(0, 0), min_confidence=0.0)`.

- [ ] **Step 1: Write a failing normalization test using a complete reader fake**

```python
def test_easyocr_normalizes_filters_and_offsets_results() -> None:
    raw = [
        ([[1.2, 2.8], [11.1, 2.2], [11.4, 8.7], [1.0, 8.9]], "Save", 0.91),
        ([[20, 20], [25, 20], [25, 25], [20, 25]], "low", 0.2),
    ]
    backend = EasyOCRBackend(reader_factory=reader_factory_returning(raw), gpu=False)
    result = backend.recognize(
        np.zeros((30, 40, 3), dtype=np.uint8),
        origin=Point(-100, 50),
        min_confidence=0.5,
    )
    assert result == [OCRDetection("Save", 0.91, BoundingBox(-99, 52, -88, 59))]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_ocr.py::test_easyocr_normalizes_filters_and_offsets_results -v`

Expected: import failure because `perception.ocr` is missing.

- [ ] **Step 3: Implement protocol, lazy reader and normalization**

Start with the injected-reader path so no model is imported during the unit test:

```python
from collections.abc import Callable, Sequence
from math import ceil, floor
from typing import Protocol, runtime_checkable

import numpy as np

from gui_agent.types import BoundingBox, ImageArray, OCRDetection, Point

class OCRReader(Protocol):
    def readtext(
        self,
        image: ImageArray,
        *,
        detail: int,
        paragraph: bool,
    ) -> Sequence[object]:
        raise NotImplementedError

ReaderFactory = Callable[[list[str], bool | str], OCRReader]

@runtime_checkable
class OCRBackend(Protocol):
    def recognize(
        self,
        image: ImageArray,
        *,
        origin: Point = Point(0, 0),
        min_confidence: float = 0.0,
    ) -> list[OCRDetection]:
        raise NotImplementedError

class EasyOCRBackend:
    def __init__(
        self,
        languages: Sequence[str] = ("ch_sim", "en"),
        *,
        gpu: bool | str = False,
        reader_factory: ReaderFactory,
    ) -> None:
        self._languages = list(languages)
        self._gpu = gpu
        self._reader_factory = reader_factory
        self._reader: OCRReader | None = None

    def _reader_instance(self) -> OCRReader:
        if self._reader is None:
            self._reader = self._reader_factory(self._languages, self._gpu)
        return self._reader

    def recognize(
        self,
        image: ImageArray,
        *,
        origin: Point = Point(0, 0),
        min_confidence: float = 0.0,
    ) -> list[OCRDetection]:
        raw_results = self._reader_instance().readtext(image, detail=1, paragraph=False)
        detections: list[OCRDetection] = []
        for raw_box, raw_text, raw_confidence in raw_results:
            confidence = float(raw_confidence)
            if confidence < min_confidence:
                continue
            points = np.asarray(raw_box, dtype=np.float64)
            box = BoundingBox(
                left=floor(float(points[:, 0].min())) + origin.x,
                top=floor(float(points[:, 1].min())) + origin.y,
                right=ceil(float(points[:, 0].max())) + origin.x,
                bottom=ceil(float(points[:, 1].max())) + origin.y,
            )
            detections.append(OCRDetection(str(raw_text), confidence, box))
        return detections
```

Use floor for left/top, ceil for right/bottom, then add `origin`. Create the reader only on the first recognition call.

- [ ] **Step 4: Verify normalization GREEN**

Run: `uv run pytest tests/test_ocr.py::test_easyocr_normalizes_filters_and_offsets_results -v`

Expected: PASS.

- [ ] **Step 5: Add failing tests for all public error branches**

Cover empty results, grayscale input, wrong dtype, unsupported channel count, empty languages, threshold below zero/above one, missing EasyOCR import, reader construction failure, malformed backend output and `readtext()` failure. Confirm reader construction occurs once across two calls.

- [ ] **Step 6: Run new tests and verify RED**

Run: `uv run pytest tests/test_ocr.py -v`

Expected: new error-branch tests fail because project exceptions and validation are incomplete.

- [ ] **Step 7: Implement explicit OCR failures**

Define `OCRDependencyError`, `OCRInitializationError`, and `OCRInferenceError`, then extend construction and recognition with these concrete gates:

```python
def _default_reader_factory(languages: list[str], gpu: bool | str) -> OCRReader:
    try:
        import easyocr
    except ModuleNotFoundError as error:
        raise OCRDependencyError(
            "EasyOCR is not installed; run `uv sync --extra ocr`"
        ) from error
    return cast(OCRReader, easyocr.Reader(languages, gpu=gpu))

def _default_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise OCRDependencyError(
            "PyTorch is not installed; run `uv sync --extra ocr`"
        ) from error
    return bool(torch.cuda.is_available())

def _validate_image(image: ImageArray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("image must be a non-empty uint8 NumPy array")
    if image.ndim == 2:
        return
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be grayscale or BGR with three channels")

def _reader_instance(self) -> OCRReader:
    if self._reader is not None:
        return self._reader
    gpu = self._gpu if self._gpu is not None else self._cuda_available()
    try:
        self._reader = self._reader_factory(list(self._languages), gpu)
    except OCRDependencyError:
        raise
    except Exception as error:
        raise OCRInitializationError("failed to initialize EasyOCR reader") from error
    return self._reader

def recognize(
    self,
    image: ImageArray,
    *,
    origin: Point = Point(0, 0),
    min_confidence: float = 0.0,
) -> list[OCRDetection]:
    _validate_image(image)
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    try:
        raw_results = self._reader_instance().readtext(image, detail=1, paragraph=False)
        return self._normalize(raw_results, origin, min_confidence)
    except (OCRDependencyError, OCRInitializationError):
        raise
    except Exception as error:
        raise OCRInferenceError("EasyOCR inference failed") from error
```

The final constructor accepts `gpu: bool | str | None = None`, `reader_factory: ReaderFactory | None = None`, and `cuda_available: Callable[[], bool] | None = None`; it rejects an empty language sequence and stores the default callables shown above. `_normalize()` contains the already-green loop and rejects malformed points, text, or confidence with `OCRInferenceError`. No-result output is `[]`.

- [ ] **Step 8: Verify all OCR behavior and commit**

Run:

```powershell
uv run pytest tests/test_ocr.py -v
uv run ruff check src/gui_agent/perception/ocr.py tests/test_ocr.py
uv run mypy src/gui_agent/perception/ocr.py
git add src/gui_agent/perception tests/test_ocr.py
git commit -m "feat: add pluggable EasyOCR backend"
```

---

### Task 5: OCR Text Localization and Non-Destructive Annotation

**Files:**
- Create: `src/gui_agent/perception/localization.py`
- Modify: `src/gui_agent/perception/__init__.py`
- Create: `tests/test_localization.py`

**Interfaces:**
- Consumes: `ImageArray`, ordered `OCRDetection` sequence.
- Produces: `MatchMode`, `find_text()`, `annotate_detections()`.

- [ ] **Step 1: Write failing tests for literal text matching**

```python
def test_find_text_supports_case_insensitive_contains_without_choosing() -> None:
    detections = [detection("Save"), detection("Save As"), detection("Cancel")]
    matches = find_text(
        detections,
        "save",
        mode=MatchMode.CONTAINS,
        case_sensitive=False,
    )
    assert [item.text for item in matches] == ["Save", "Save As"]
```

Also assert exact case-sensitive mode and an empty query error.

- [ ] **Step 2: Run matching tests and verify RED**

Run: `uv run pytest tests/test_localization.py -k find_text -v`

Expected: import failure because localization is missing.

- [ ] **Step 3: Implement matching minimally**

Use:

```python
from collections.abc import Sequence
from enum import StrEnum

from gui_agent.types import OCRDetection

class MatchMode(StrEnum):
    EXACT = "exact"
    CONTAINS = "contains"

def find_text(
    detections: Sequence[OCRDetection],
    query: str,
    *,
    mode: MatchMode = MatchMode.EXACT,
    case_sensitive: bool = True,
) -> list[OCRDetection]:
    if not query:
        raise ValueError("query must not be empty")
    expected = query if case_sensitive else query.casefold()
    matches: list[OCRDetection] = []
    for detection in detections:
        actual = detection.text if case_sensitive else detection.text.casefold()
        if mode is MatchMode.EXACT and actual == expected:
            matches.append(detection)
        elif mode is MatchMode.CONTAINS and expected in actual:
            matches.append(detection)
    return matches
```

Return matches in input order and never collapse or select duplicates.

- [ ] **Step 4: Verify matching GREEN**

Run: `uv run pytest tests/test_localization.py -k find_text -v`

Expected: PASS.

- [ ] **Step 5: Write failing annotation tests**

Use literal high/low confidence detections. Assert the returned array does not share storage with the input, the original remains all zero, box/center pixels change, green and orange pixels are present, invalid cutoff fails, and a caller-provided missing font path fails clearly.

- [ ] **Step 6: Run annotation tests and verify RED**

Run: `uv run pytest tests/test_localization.py -k annotate -v`

Expected: failures because annotation is missing.

- [ ] **Step 7: Implement drawing on a copy**

Implement the annotation path concretely:

```python
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gui_agent.types import ImageArray, OCRDetection

HIGH_CONFIDENCE_BGR = (0, 180, 0)
LOW_CONFIDENCE_BGR = (0, 165, 255)

def _annotation_font(font_path: Path | None) -> ImageFont.ImageFont:
    if font_path is not None:
        if not font_path.is_file():
            raise ValueError(f"font_path does not exist: {font_path}")
        return ImageFont.truetype(str(font_path), 16)
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for candidate in (windows / "msyh.ttc", windows / "simhei.ttf"):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), 16)
    return ImageFont.load_default()

def annotate_detections(
    image: ImageArray,
    detections: Sequence[OCRDetection],
    *,
    confidence_cutoff: float = 0.8,
    font_path: Path | None = None,
) -> ImageArray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a uint8 BGR array")
    if not 0.0 <= confidence_cutoff <= 1.0:
        raise ValueError("confidence_cutoff must be between 0 and 1")
    output = image.copy()
    colors: list[tuple[int, int, int]] = []
    for detection in detections:
        color = (
            HIGH_CONFIDENCE_BGR
            if detection.confidence >= confidence_cutoff
            else LOW_CONFIDENCE_BGR
        )
        colors.append(color)
        box = detection.box
        cv2.rectangle(output, (box.left, box.top), (box.right - 1, box.bottom - 1), color, 2)
        cv2.circle(output, (detection.center.x, detection.center.y), 3, color, -1)

    rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    font = _annotation_font(font_path)
    for detection, bgr in zip(detections, colors, strict=True):
        label = f"{detection.text} {detection.confidence:.2f}"
        rgb_color = (bgr[2], bgr[1], bgr[0])
        draw.text((detection.box.left, max(0, detection.box.top - 18)), label, fill=rgb_color, font=font)
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
```

Draw boxes and centers with OpenCV on `image.copy()`. Draw `"{text} {confidence:.2f}"` through Pillow after BGR→RGB conversion. Try Windows Microsoft YaHei/SimHei fonts before Pillow's default and convert back to BGR.

- [ ] **Step 8: Verify localization and commit**

Run:

```powershell
uv run pytest tests/test_localization.py -v
uv run ruff check src/gui_agent/perception/localization.py tests/test_localization.py
uv run mypy src/gui_agent/perception/localization.py
git add src/gui_agent/perception tests/test_localization.py
git commit -m "feat: add OCR-based UI localization"
```

---

### Task 6: Safe Mouse and Keyboard Controller

**Files:**
- Create: `src/gui_agent/control/controller.py`
- Modify: `src/gui_agent/control/__init__.py`
- Create: `tests/test_control.py`

**Interfaces:**
- Consumes: `Point`, `ScreenRegion` and an injected virtual-bounds provider.
- Produces: `ActionRecord`, `DesktopBackend`, `PyAutoGUIAdapter`, `DesktopController` actions.

- [ ] **Step 1: Write a failing dry-run test that proves no backend call occurs**

```python
def test_dry_run_click_records_action_without_touching_backend(fake_desktop) -> None:
    controller = DesktopController(
        dry_run=True,
        backend=fake_desktop,
        bounds_provider=lambda: ScreenRegion(-100, 0, 300, 200),
    )
    record = controller.click(Point(10, 20))
    assert record.name == "click"
    assert record.parameters == (("button", "left"), ("clicks", 1), ("point", Point(10, 20)))
    assert fake_desktop.calls == []
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_control.py::test_dry_run_click_records_action_without_touching_backend -v`

Expected: import failure because `control.controller` is missing.

- [ ] **Step 3: Implement immutable action records and dry-run dispatch**

Implement only the immutable record and click path needed by the first test:

```python
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from gui_agent.types import Point, ScreenRegion

@dataclass(frozen=True, slots=True)
class ActionRecord:
    name: str
    parameters: tuple[tuple[str, object], ...]
    created_at: datetime

class DesktopBackend(Protocol):
    def click(self, point: Point, *, button: str, clicks: int) -> None:
        raise NotImplementedError

class DesktopController:
    def __init__(
        self,
        *,
        dry_run: bool = True,
        pause: float = 0.1,
        backend: DesktopBackend | None = None,
        bounds_provider: Callable[[], ScreenRegion] | None = None,
    ) -> None:
        if pause < 0:
            raise ValueError("pause must be non-negative")
        if bounds_provider is None:
            raise ValueError("bounds_provider is required until the default provider is added")
        self.dry_run = dry_run
        self._pause = pause
        self._backend = backend
        self._bounds_provider = bounds_provider
        self._history: list[ActionRecord] = []

    @property
    def history(self) -> tuple[ActionRecord, ...]:
        return tuple(self._history)

    def click(self, point: Point, *, button: str = "left", clicks: int = 1) -> ActionRecord:
        if not self._bounds_provider().contains(point):
            raise ValueError(f"point is outside the virtual desktop: {point}")
        record = ActionRecord(
            name="click",
            parameters=(("button", button), ("clicks", clicks), ("point", point)),
            created_at=datetime.now(timezone.utc),
        )
        self._history.append(record)
        if not self.dry_run:
            if self._backend is None:
                raise RuntimeError("live control requires a desktop backend")
            self._backend.click(point, button=button, clicks=clicks)
        return record
```

Do not instantiate `PyAutoGUIAdapter` while dry-running.

- [ ] **Step 4: Verify dry-run GREEN**

Run: `uv run pytest tests/test_control.py::test_dry_run_click_records_action_without_touching_backend -v`

Expected: PASS.

- [ ] **Step 5: Add failing tests for every action and validator**

Use a complete fake backend. Cover each live action's exact arguments, `FAILSAFE`/pause configuration, negative and non-finite durations, points on each half-open boundary, unknown buttons, zero clicks, empty text, invalid keys, non-integer scroll values, and a negative-origin valid monitor.

- [ ] **Step 6: Run all controller tests and verify RED**

Run: `uv run pytest tests/test_control.py -v`

Expected: failures name unimplemented actions and validators.

- [ ] **Step 7: Implement validation and the PyAutoGUI adapter**

The adapter owns the only `import pyautogui`, sets `FAILSAFE=True` and `PAUSE=pause`, and maps methods to `moveTo`, `click`, `write`, `hotkey`, `scroll`, and `dragTo`. The default bounds provider reads MSS `monitors[0]` without taking a screenshot. Use a frozen set of PyAutoGUI-compatible key names so dry-run validates keys without importing PyAutoGUI. Extend the controller using this dispatch pattern and concrete methods:

```python
VALID_BUTTONS = frozenset({"left", "middle", "right"})
VALID_KEYS = frozenset(
    {"ctrl", "shift", "alt", "win", "enter", "esc", "tab", "space", "backspace", "delete"}
    | {chr(code) for code in range(ord("a"), ord("z") + 1)}
    | {str(number) for number in range(10)}
    | {f"f{number}" for number in range(1, 13)}
)

def _duration(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value

def _point(self, point: Point) -> None:
    if not self._bounds_provider().contains(point):
        raise ValueError(f"point is outside the virtual desktop: {point}")

def _dispatch(
    self,
    name: str,
    parameters: tuple[tuple[str, object], ...],
    operation: Callable[[DesktopBackend], None],
) -> ActionRecord:
    record = ActionRecord(name, parameters, datetime.now(timezone.utc))
    self._history.append(record)
    if not self.dry_run:
        operation(self._backend_instance())
    return record

def move_to(self, point: Point, *, duration: float = 0.2) -> ActionRecord:
    self._point(point)
    duration = _duration(duration, "duration")
    return self._dispatch(
        "move_to",
        (("duration", duration), ("point", point)),
        lambda backend: backend.move_to(point, duration=duration),
    )

def click(self, point: Point, *, button: str = "left", clicks: int = 1) -> ActionRecord:
    self._point(point)
    if button not in VALID_BUTTONS:
        raise ValueError(f"unsupported button: {button}")
    if isinstance(clicks, bool) or not isinstance(clicks, int) or clicks < 1:
        raise ValueError("clicks must be a positive integer")
    return self._dispatch(
        "click",
        (("button", button), ("clicks", clicks), ("point", point)),
        lambda backend: backend.click(point, button=button, clicks=clicks),
    )

def double_click(self, point: Point) -> ActionRecord:
    return self.click(point, clicks=2)

def right_click(self, point: Point) -> ActionRecord:
    return self.click(point, button="right")

def type_text(self, text: str, *, interval: float = 0.02) -> ActionRecord:
    if not text:
        raise ValueError("text must not be empty")
    interval = _duration(interval, "interval")
    return self._dispatch(
        "type_text",
        (("interval", interval), ("text", text)),
        lambda backend: backend.type_text(text, interval=interval),
    )

def hotkey(self, *keys: str) -> ActionRecord:
    normalized = tuple(key.casefold() for key in keys)
    if not normalized or any(key not in VALID_KEYS for key in normalized):
        raise ValueError(f"unsupported hotkey sequence: {keys}")
    return self._dispatch(
        "hotkey",
        (("keys", normalized),),
        lambda backend: backend.hotkey(*normalized),
    )

def scroll(self, clicks: int, *, point: Point | None = None) -> ActionRecord:
    if isinstance(clicks, bool) or not isinstance(clicks, int):
        raise ValueError("scroll clicks must be an integer")
    if point is not None:
        self._point(point)
    return self._dispatch(
        "scroll",
        (("clicks", clicks), ("point", point)),
        lambda backend: backend.scroll(clicks, point=point),
    )

def drag_to(
    self,
    start: Point,
    end: Point,
    *,
    duration: float = 0.5,
    button: str = "left",
) -> ActionRecord:
    self._point(start)
    self._point(end)
    duration = _duration(duration, "duration")
    if button not in VALID_BUTTONS:
        raise ValueError(f"unsupported button: {button}")
    return self._dispatch(
        "drag_to",
        (("button", button), ("duration", duration), ("end", end), ("start", start)),
        lambda backend: backend.drag_to(start, end, duration=duration, button=button),
    )
```

`DesktopBackend` declares the seven backend methods used above. `PyAutoGUIAdapter` implements them literally: `moveTo(point.x, point.y, duration=duration)`, `click(point.x, point.y, button=button, clicks=clicks)`, `write(text, interval=interval)`, `hotkey(*keys)`, `moveTo(point.x, point.y)` followed by `scroll(clicks)`, and `moveTo(start.x, start.y)` followed by `dragTo(end.x, end.y, duration=duration, button=button)`. `_backend_instance()` lazily constructs this adapter only in live mode.

- [ ] **Step 8: Verify controller safety and commit**

Run:

```powershell
uv run pytest tests/test_control.py -v
uv run ruff check src/gui_agent/control tests/test_control.py
uv run mypy src/gui_agent/control/controller.py
git add src/gui_agent/control tests/test_control.py
git commit -m "feat: add safe desktop controller"
```

---

### Task 7: Safe Demonstrations

**Files:**
- Create: `examples/capture_demo.py`
- Create: `examples/ocr_demo.py`
- Create: `examples/control_demo.py`
- Create: `examples/perception_control_demo.py`
- Create: `tests/test_examples.py`

**Interfaces:**
- Consumes: public perception/control APIs.
- Produces: command-line examples whose default execution has no irreversible desktop effect.

- [ ] **Step 1: Write failing tests for the live-click confirmation gate**

Expose from `perception_control_demo.py`:

```python
CONFIRMATION_PHRASE = "CLICK THIS CANDIDATE"

def live_click_confirmed(
    *,
    execute: bool,
    candidate_count: int,
    input_fn: Callable[[str], str] = input,
) -> bool:
    if not execute or candidate_count != 1:
        return False
    response = input_fn(f"Type {CONFIRMATION_PHRASE!r} to execute one real click: ")
    return response.strip() == CONFIRMATION_PHRASE
```

Tests assert false for default `execute=False`, false for zero/two candidates without prompting, false for a wrong phrase, and true only for one candidate plus the exact phrase.

- [ ] **Step 2: Run the safety test and verify RED**

Run: `uv run pytest tests/test_examples.py -v`

Expected: imports fail because examples are missing.

- [ ] **Step 3: Implement the confirmation function and four CLI programs**

- `capture_demo.py`: select monitor or absolute region; save only when `--output` is supplied.
- `ocr_demo.py`: accept a caller-provided image; default GPU auto; print text, confidence, box and center.
- `control_demo.py`: instantiate `DesktopController(dry_run=not args.execute)`; any live action requires the same exact confirmation phrase.
- `perception_control_demo.py`: capture, OCR, find all requested text, write an annotation only to an explicit/ignored artifact path, print every candidate, and permit a live click only for one candidate after exact confirmation.

Every program has `main() -> int` and `if __name__ == "__main__": raise SystemExit(main())`.

- [ ] **Step 4: Verify example safety GREEN**

Run:

```powershell
uv run pytest tests/test_examples.py -v
uv run python examples/control_demo.py --help
uv run python examples/perception_control_demo.py --help
```

Expected: tests pass and help commands exit zero without importing a model or touching the desktop.

- [ ] **Step 5: Static-check and commit examples**

Run:

```powershell
uv run ruff check examples tests/test_examples.py
uv run mypy examples
git add examples tests/test_examples.py
git commit -m "feat: add safe perception and control demos"
```

---

### Task 8: CI, Documentation, Integration Measurements and Final Verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`
- Create: `docs/setup/windows-setup.md`
- Create: `docs/test-reports/week2-test-report.md`

**Interfaces:**
- Consumes: all completed modules, lock file and test output.
- Produces: reproducible onboarding, Windows CI, measured Week 2 report and publish-ready branch.

- [ ] **Step 1: Create Windows CI with safe commands**

Use `astral-sh/setup-uv`, Python 3.11, `uv sync --locked --group dev`, then:

```yaml
- run: uv run ruff check .
- run: uv run mypy src tests examples
- run: uv run pytest -m "not integration" --cov=gui_agent --cov-report=term-missing --cov-report=xml
```

Do not install the `ocr` extra in ordinary CI; unit tests inject the OCR reader.

- [ ] **Step 2: Run full ordinary verification and record exact totals**

Run:

```powershell
uv run ruff check .
uv run mypy src tests examples
uv run pytest -m "not integration" --cov=gui_agent --cov-report=term-missing
```

Copy the final test count and coverage table into `week2-test-report.md` without inventing values.

- [ ] **Step 3: Run privacy-safe local integration probes**

Run an import/GPU probe, then one in-memory MSS capture without saving it. Generate a synthetic English/Chinese image in memory with Pillow and run EasyOCR on it. Measure first OCR initialization, repeated OCR, and 20 in-memory captures with `time.perf_counter()`. Do not print recognized real-screen content and do not save the real capture.

Expected checks:

```python
assert torch.cuda.is_available()
assert "RTX 5070 Ti" in torch.cuda.get_device_name(0)
assert screenshot.image.size > 0
assert screenshot.image.dtype == np.uint8
```

- [ ] **Step 4: Write setup documentation with actual local output**

Document uv installation, `uv sync --group dev --extra ocr`, activation/`uv run`, CUDA verification, dependency imports, non-saving capture probe, EasyOCR cache behavior, DPI awareness, multi-monitor negative coordinates, PyAutoGUI fail-safe, model download failures and CPU fallback. Include only measured version/device output from Step 3.

- [ ] **Step 5: Write the Week 2 report from measurements**

Include OS/build, GPU/driver, Python, PyTorch CUDA, EasyOCR and core package versions; ordinary test count; coverage; capture and OCR timings; synthetic fixture cases; controller fake-test result; DPI/coordinate strategy; unresolved limitations; and Week 3 recommendations. State explicitly if a local integration probe could not run and include its exact non-sensitive error.

- [ ] **Step 6: Complete README**

Describe purpose, implemented Week 2 scope, Mermaid architecture link, setup, unit/integration commands, safe demos, default dry-run, emergency PyAutoGUI fail-safe, privacy exclusions, project layout and known OCR/multi-monitor limitations.

- [ ] **Step 7: Re-run fresh final gates**

Run:

```powershell
uv lock --check
uv run ruff check .
uv run mypy src tests examples
uv run pytest -m "not integration" --cov=gui_agent --cov-report=term-missing
git diff --check origin/master...HEAD
git status --short
```

Expected: lock check, Ruff, mypy and tests exit zero; diff check is clean; status contains only intended documentation changes before the final commit.

- [ ] **Step 8: Commit documentation and CI**

Run:

```powershell
git add .github/workflows/ci.yml README.md docs/setup/windows-setup.md docs/test-reports/week2-test-report.md
git commit -m "docs: add Week 2 setup and test report"
```

- [ ] **Step 9: Publish using the GitHub workflow**

Inspect `git status -sb` and `git diff origin/master...HEAD`, rerun any stale checks, then push `agent/week2-perception-control` with tracking. Create a draft PR targeting `master` with sections for changes, rationale, safety impact, validation and known limitations. Do not create `v0.2.0-week2` until the PR is reviewed and merged.
