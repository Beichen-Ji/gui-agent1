import numpy as np
import pytest

from gui_agent.perception.preprocessing import (
    DEFAULT_OCR_PROFILE,
    OCR_PROFILES,
    OCRProfile,
    preprocess_image,
)


def test_preprocessing_accepts_bgr_and_grayscale_images() -> None:
    bgr = np.zeros((20, 30, 3), dtype=np.uint8)
    grayscale = np.zeros((20, 30), dtype=np.uint8)

    bgr_result = preprocess_image(bgr, OCR_PROFILES["balanced"])
    gray_result = preprocess_image(grayscale, OCR_PROFILES["balanced"])

    assert bgr_result.image.ndim == 2
    assert gray_result.image.ndim == 2
    assert bgr_result.original_size == (30, 20)


def test_clahe_profile_scales_image_and_records_exact_coordinate_scale() -> None:
    image = np.arange(20 * 30, dtype=np.uint8).reshape(20, 30)

    result = preprocess_image(image, OCR_PROFILES["accurate"])

    assert result.image.shape == (30, 45)
    assert result.scale_x == 1.5
    assert result.scale_y == 1.5


@pytest.mark.parametrize(
    "changes",
    [
        {"decoder": "invalid"},
        {"beam_width": 0},
        {"batch_size": 0},
        {"workers": -1},
        {"canvas_size": 0},
        {"mag_ratio": 0.0},
        {"contrast_threshold": 1.1},
        {"adjust_contrast": -0.1},
        {"preprocessing": "invalid"},
    ],
)
def test_ocr_profile_rejects_unsafe_parameters(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "decoder": "greedy",
        "beam_width": 1,
        "batch_size": 1,
        "workers": 0,
        "canvas_size": 1280,
        "mag_ratio": 1.0,
        "contrast_threshold": 0.1,
        "adjust_contrast": 0.5,
        "preprocessing": "none",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        OCRProfile(**values)  # type: ignore[arg-type]


def test_named_profiles_are_explicit_and_balanced_is_benchmarked_default() -> None:
    assert set(OCR_PROFILES) == {"fast", "balanced", "accurate"}
    assert DEFAULT_OCR_PROFILE == "balanced"
    assert OCR_PROFILES["fast"].preprocessing == "none"
    assert OCR_PROFILES["balanced"].mag_ratio > OCR_PROFILES["fast"].mag_ratio
