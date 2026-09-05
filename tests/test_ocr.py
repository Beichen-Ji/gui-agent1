import sys
from collections.abc import Sequence
from types import ModuleType
from typing import cast

import numpy as np
import pytest

import gui_agent.perception.ocr as ocr_module
from gui_agent.perception.ocr import (
    EasyOCRBackend,
    OCRBackend,
    OCRDependencyError,
    OCRInferenceError,
    OCRInitializationError,
)
from gui_agent.perception.preprocessing import OCR_PROFILES
from gui_agent.types import BoundingBox, ImageArray, OCRDetection, Point


class FakeReader:
    def __init__(
        self,
        results: Sequence[object] = (),
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = list(results)
        self.error = error
        self.calls: list[tuple[ImageArray, int, bool]] = []
        self.options: list[dict[str, object]] = []

    def readtext(
        self,
        image: ImageArray,
        *,
        detail: int,
        paragraph: bool,
        decoder: str,
        beamWidth: int,
        batch_size: int,
        workers: int,
        canvas_size: int,
        mag_ratio: float,
        contrast_ths: float,
        adjust_contrast: float,
    ) -> Sequence[object]:
        self.calls.append((image, detail, paragraph))
        self.options.append(
            {
                "decoder": decoder,
                "beamWidth": beamWidth,
                "batch_size": batch_size,
                "workers": workers,
                "canvas_size": canvas_size,
                "mag_ratio": mag_ratio,
                "contrast_ths": contrast_ths,
                "adjust_contrast": adjust_contrast,
            }
        )
        if self.error is not None:
            raise self.error
        return list(self.results)


class ReaderFactoryProbe:
    def __init__(
        self,
        reader: FakeReader | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.reader = reader or FakeReader()
        self.error = error
        self.calls: list[tuple[list[str], bool | str]] = []

    def __call__(self, languages: list[str], gpu: bool | str) -> FakeReader:
        self.calls.append((languages, gpu))
        if self.error is not None:
            raise self.error
        return self.reader


class FakeEasyOCRModule(ModuleType):
    def __init__(self, reader: FakeReader) -> None:
        super().__init__("easyocr")
        self.reader = reader
        self.calls: list[tuple[list[str], bool | str, bool]] = []

    def Reader(  # noqa: N802
        self,
        languages: list[str],
        *,
        gpu: bool | str,
        verbose: bool,
    ) -> FakeReader:
        self.calls.append((languages, gpu, verbose))
        return self.reader


def color_image() -> ImageArray:
    return np.zeros((30, 40, 3), dtype=np.uint8)


def test_default_easyocr_reader_disables_console_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = FakeReader()
    easyocr_module = FakeEasyOCRModule(reader)
    monkeypatch.setitem(sys.modules, "easyocr", easyocr_module)

    result = ocr_module._default_reader_factory(["ch_sim", "en"], True)

    assert result is reader
    assert easyocr_module.calls == [(["ch_sim", "en"], True, False)]


def test_easyocr_normalizes_filters_and_offsets_results() -> None:
    raw = [
        ([[1.2, 2.8], [11.1, 2.2], [11.4, 8.7], [1.0, 8.9]], "Save", 0.91),
        ([[20, 20], [25, 20], [25, 25], [20, 25]], "low", 0.2),
    ]
    factory = ReaderFactoryProbe(FakeReader(raw))
    backend = EasyOCRBackend(reader_factory=factory, gpu=False, profile="fast")

    result = backend.recognize(
        color_image(),
        origin=Point(-100, 50),
        min_confidence=0.5,
    )

    assert result == [OCRDetection("Save", 0.91, BoundingBox(-99, 52, -88, 59))]


def test_easyocr_skips_blank_text_without_losing_valid_detections() -> None:
    raw = [
        ([[1, 1], [3, 1], [3, 3], [1, 3]], "", 0.0),
        ([[5, 5], [15, 5], [15, 10], [5, 10]], "Save", 0.9),
        ([[20, 20], [25, 20], [25, 25], [20, 25]], "   ", 0.1),
    ]
    backend = EasyOCRBackend(
        reader_factory=ReaderFactoryProbe(FakeReader(raw)),
        gpu=False,
        profile="fast",
    )

    assert backend.recognize(color_image()) == [
        OCRDetection("Save", 0.9, BoundingBox(5, 5, 15, 10))
    ]


def test_easyocr_creates_reader_once_and_passes_complete_call_shape() -> None:
    reader = FakeReader()
    factory = ReaderFactoryProbe(reader)
    backend = EasyOCRBackend(reader_factory=factory, gpu=False, profile="fast")
    image = color_image()

    assert backend.recognize(image) == []
    assert backend.recognize(image) == []

    assert factory.calls == [(["ch_sim", "en"], False)]
    assert reader.calls == [(image, 1, False), (image, 1, False)]


def test_easyocr_passes_only_profile_options_and_restores_scaled_coordinates() -> None:
    raw = [([[15, 15], [45, 15], [45, 30], [15, 30]], "Save", 0.9)]
    reader = FakeReader(raw)
    backend = EasyOCRBackend(
        reader_factory=ReaderFactoryProbe(reader),
        gpu=False,
        profile="accurate",
    )

    detected = backend.recognize(color_image(), origin=Point(100, 50))

    assert reader.calls[0][0].shape == (45, 60)
    assert reader.options == [
        {
            "decoder": OCR_PROFILES["accurate"].decoder,
            "beamWidth": OCR_PROFILES["accurate"].beam_width,
            "batch_size": OCR_PROFILES["accurate"].batch_size,
            "workers": OCR_PROFILES["accurate"].workers,
            "canvas_size": OCR_PROFILES["accurate"].canvas_size,
            "mag_ratio": 1.0,
            "contrast_ths": OCR_PROFILES["accurate"].contrast_threshold,
            "adjust_contrast": OCR_PROFILES["accurate"].adjust_contrast,
        }
    ]
    assert detected == [
        OCRDetection("Save", 0.9, BoundingBox(110, 60, 130, 70))
    ]


def test_easyocr_rejects_unknown_profile_name() -> None:
    with pytest.raises(ValueError, match="profile"):
        EasyOCRBackend(
            reader_factory=ReaderFactoryProbe(),
            gpu=False,
            profile="turbo",
        )


def test_easyocr_accepts_grayscale_image() -> None:
    factory = ReaderFactoryProbe()
    backend = EasyOCRBackend(reader_factory=factory, gpu=False)
    image = np.zeros((10, 20), dtype=np.uint8)

    assert backend.recognize(image) == []


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3, 3), dtype=np.float32),
        np.zeros((0, 3, 3), dtype=np.uint8),
        np.zeros((2, 3, 1), dtype=np.uint8),
        np.zeros((2, 3, 4), dtype=np.uint8),
        np.zeros((2,), dtype=np.uint8),
    ],
)
def test_easyocr_rejects_invalid_image(
    image: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> None:
    backend = EasyOCRBackend(reader_factory=ReaderFactoryProbe(), gpu=False)

    with pytest.raises(ValueError, match="image"):
        backend.recognize(cast(ImageArray, image))


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("inf"), float("nan")])
def test_easyocr_rejects_invalid_confidence_threshold(threshold: float) -> None:
    backend = EasyOCRBackend(reader_factory=ReaderFactoryProbe(), gpu=False)

    with pytest.raises(ValueError, match="min_confidence"):
        backend.recognize(color_image(), min_confidence=threshold)


def test_easyocr_rejects_empty_language_list() -> None:
    with pytest.raises(ValueError, match="languages"):
        EasyOCRBackend(languages=(), reader_factory=ReaderFactoryProbe(), gpu=False)


def test_easyocr_auto_selects_cuda() -> None:
    factory = ReaderFactoryProbe()
    backend = EasyOCRBackend(
        reader_factory=factory,
        gpu=None,
        cuda_available=lambda: True,
    )

    backend.recognize(color_image())

    assert factory.calls == [(["ch_sim", "en"], True)]


def test_easyocr_explicit_gpu_does_not_probe_cuda() -> None:
    def unexpected_probe() -> bool:
        raise AssertionError("CUDA probe must not run for explicit GPU selection")

    factory = ReaderFactoryProbe()
    backend = EasyOCRBackend(
        reader_factory=factory,
        gpu="cuda:0",
        cuda_available=unexpected_probe,
    )

    backend.recognize(color_image())

    assert factory.calls == [(["ch_sim", "en"], "cuda:0")]


def test_easyocr_reports_missing_dependency() -> None:
    factory = ReaderFactoryProbe(error=ModuleNotFoundError("easyocr"))
    backend = EasyOCRBackend(reader_factory=factory, gpu=False)

    with pytest.raises(OCRDependencyError, match="uv sync --extra ocr") as captured:
        backend.recognize(color_image())

    assert isinstance(captured.value.__cause__, ModuleNotFoundError)


def test_easyocr_wraps_reader_initialization_error() -> None:
    factory = ReaderFactoryProbe(error=RuntimeError("bad model"))
    backend = EasyOCRBackend(reader_factory=factory, gpu=False)

    with pytest.raises(OCRInitializationError, match="initialize") as captured:
        backend.recognize(color_image())

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_easyocr_wraps_inference_error() -> None:
    reader = FakeReader(error=RuntimeError("inference failed"))
    backend = EasyOCRBackend(reader_factory=ReaderFactoryProbe(reader), gpu=False)

    with pytest.raises(OCRInferenceError, match="inference") as captured:
        backend.recognize(color_image())

    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "raw_result",
    [
        ([], "Save", 0.9),
        ([[1, 2]], "Save", 0.9),
        ([[1, 2], [4, 2], [4, 5], [1, 5]], None, 0.9),
        ([[1, 2], [4, 5]], "Save", 1.1),
        ("not-a-box", "Save", 0.9),
        ("missing confidence", "Save"),
    ],
)
def test_easyocr_rejects_malformed_backend_result(raw_result: object) -> None:
    reader = FakeReader([raw_result])
    backend = EasyOCRBackend(reader_factory=ReaderFactoryProbe(reader), gpu=False)

    with pytest.raises(OCRInferenceError, match="malformed"):
        backend.recognize(color_image())


def test_easyocr_satisfies_backend_protocol() -> None:
    backend = EasyOCRBackend(reader_factory=ReaderFactoryProbe(), gpu=False)

    assert isinstance(backend, OCRBackend)
