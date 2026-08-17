from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from examples import capture_demo, control_demo, ocr_demo, perception_control_demo
from examples.perception_control_demo import (
    CONFIRMATION_PHRASE,
    live_click_confirmed,
)
from gui_agent.control import ActionRecord
from gui_agent.types import (
    BoundingBox,
    OCRDetection,
    Point,
    ScreenRegion,
    ScreenshotResult,
)


class InputProbe:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@pytest.mark.parametrize(
    ("execute", "candidate_count"),
    [(False, 1), (True, 0), (True, 2)],
)
def test_live_click_gate_refuses_without_prompt(
    execute: bool,
    candidate_count: int,
) -> None:
    input_probe = InputProbe(CONFIRMATION_PHRASE)

    confirmed = live_click_confirmed(
        execute=execute,
        candidate_count=candidate_count,
        input_fn=input_probe,
    )

    assert confirmed is False
    assert input_probe.prompts == []


@pytest.mark.parametrize(
    "response",
    ["", "click this candidate", "CLICK THIS CANDIDATES", "yes"],
)
def test_live_click_gate_rejects_any_non_matching_phrase(response: str) -> None:
    input_probe = InputProbe(response)

    confirmed = live_click_confirmed(
        execute=True,
        candidate_count=1,
        input_fn=input_probe,
    )

    assert confirmed is False
    assert input_probe.prompts == [
        f"Type {CONFIRMATION_PHRASE!r} to execute one real click: "
    ]


def test_live_click_gate_accepts_one_candidate_and_exact_phrase() -> None:
    input_probe: Callable[[str], str] = InputProbe(CONFIRMATION_PHRASE)

    assert live_click_confirmed(
        execute=True,
        candidate_count=1,
        input_fn=input_probe,
    )


class ControllerProbe:
    instances: list["ControllerProbe"] = []

    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.points: list[Point] = []
        self.__class__.instances.append(self)

    def click(self, point: Point) -> ActionRecord:
        self.points.append(point)
        return ActionRecord("click", (("point", point),), datetime.now(UTC))


def screenshot() -> ScreenshotResult:
    return ScreenshotResult(
        image=np.zeros((40, 60, 3), dtype=np.uint8),
        monitor_index=1,
        captured_at=datetime.now(UTC),
        origin=Point(-100, 20),
    )


@pytest.mark.parametrize(
    "main_func",
    [
        capture_demo.main,
        ocr_demo.main,
        control_demo.main,
        perception_control_demo.main,
    ],
)
def test_every_example_help_exits_before_runtime_work(
    main_func: Callable[[list[str]], int],
) -> None:
    with pytest.raises(SystemExit) as captured:
        main_func(["--help"])

    assert captured.value.code == 0


def test_capture_demo_does_not_save_without_explicit_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, Path | None]] = []

    class CaptureProbe:
        def capture_monitor(
            self,
            monitor_index: int,
            *,
            save_path: Path | None,
        ) -> ScreenshotResult:
            calls.append((monitor_index, save_path))
            return screenshot()

    monkeypatch.setattr(capture_demo, "ScreenCapture", CaptureProbe)

    assert capture_demo.main([]) == 0
    assert calls == [(1, None)]


def test_capture_demo_passes_explicit_region_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, Path | None]] = []
    output = tmp_path / "capture.png"

    class CaptureProbe:
        def capture_region(
            self,
            region: object,
            *,
            save_path: Path | None,
        ) -> ScreenshotResult:
            calls.append((region, save_path))
            return screenshot()

    monkeypatch.setattr(capture_demo, "ScreenCapture", CaptureProbe)

    result = capture_demo.main(
        ["--region", "-100", "20", "60", "40", "--output", str(output)]
    )

    assert result == 0
    assert calls == [(ScreenRegion(-100, 20, 60, 40), output)]


def test_ocr_demo_uses_caller_image_and_auto_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "synthetic.png"
    image_path.write_bytes(b"fixture")
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    calls: list[tuple[object, ...]] = []

    class OCRProbe:
        def __init__(self, *, gpu: bool | str | None) -> None:
            calls.append(("init", gpu))

        def recognize(
            self,
            received: object,
            *,
            origin: Point,
            min_confidence: float,
        ) -> list[OCRDetection]:
            calls.append(("recognize", received, origin, min_confidence))
            return [OCRDetection("Save", 0.91, BoundingBox(1, 2, 10, 12))]

    monkeypatch.setattr("examples.ocr_demo.cv2.imread", lambda *_args: image)
    monkeypatch.setattr(ocr_demo, "EasyOCRBackend", OCRProbe)

    assert ocr_demo.main([str(image_path)]) == 0
    output = capsys.readouterr().out

    assert calls == [
        ("init", None),
        ("recognize", image, Point(0, 0), 0.0),
    ]
    assert "'Save'" in output
    assert "0.91" in output


def test_control_demo_defaults_to_dry_run_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ControllerProbe.instances = []
    input_probe = InputProbe(CONFIRMATION_PHRASE)
    monkeypatch.setattr(control_demo, "DesktopController", ControllerProbe)

    result = control_demo.main(["--x", "10", "--y", "20"], input_fn=input_probe)

    assert result == 0
    assert input_probe.prompts == []
    assert len(ControllerProbe.instances) == 1
    assert ControllerProbe.instances[0].dry_run is True
    assert ControllerProbe.instances[0].points == [Point(10, 20)]


def test_control_demo_execute_requires_confirmation_before_live_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ControllerProbe.instances = []
    monkeypatch.setattr(control_demo, "DesktopController", ControllerProbe)

    rejected = control_demo.main(
        ["--x", "10", "--y", "20", "--execute"],
        input_fn=InputProbe("no"),
    )
    accepted = control_demo.main(
        ["--x", "10", "--y", "20", "--execute"],
        input_fn=InputProbe(CONFIRMATION_PHRASE),
    )

    assert rejected == 1
    assert accepted == 0
    assert len(ControllerProbe.instances) == 1
    assert ControllerProbe.instances[0].dry_run is False


def test_end_to_end_demo_defaults_to_one_candidate_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shot = screenshot()
    candidate = OCRDetection("Save", 0.95, BoundingBox(-90, 30, -70, 50))
    ocr_calls: list[tuple[object, Point, float]] = []

    class CaptureProbe:
        def capture_monitor(self, monitor_index: int) -> ScreenshotResult:
            assert monitor_index == 1
            return shot

    class OCRProbe:
        def __init__(self, *, gpu: bool | str | None) -> None:
            assert gpu is None

        def recognize(
            self,
            image: object,
            *,
            origin: Point,
            min_confidence: float,
        ) -> list[OCRDetection]:
            ocr_calls.append((image, origin, min_confidence))
            return [candidate]

    ControllerProbe.instances = []
    monkeypatch.setattr(perception_control_demo, "ScreenCapture", CaptureProbe)
    monkeypatch.setattr(perception_control_demo, "EasyOCRBackend", OCRProbe)
    monkeypatch.setattr(perception_control_demo, "DesktopController", ControllerProbe)

    result = perception_control_demo.main(["Save"], input_fn=InputProbe())

    assert result == 0
    assert ocr_calls == [(shot.image, shot.origin, 0.5)]
    assert len(ControllerProbe.instances) == 1
    assert ControllerProbe.instances[0].dry_run is True
    assert ControllerProbe.instances[0].points == [candidate.center]


def test_end_to_end_execute_refuses_multiple_candidates_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shot = screenshot()
    candidates = [
        OCRDetection("Save", 0.95, BoundingBox(-90, 30, -70, 50)),
        OCRDetection("Save", 0.90, BoundingBox(-60, 30, -40, 50)),
    ]

    class CaptureProbe:
        def capture_monitor(self, _monitor_index: int) -> ScreenshotResult:
            return shot

    class OCRProbe:
        def __init__(self, *, gpu: bool | str | None) -> None:
            return None

        def recognize(self, *_args: object, **_kwargs: object) -> list[OCRDetection]:
            return candidates

    input_probe = InputProbe(CONFIRMATION_PHRASE)
    ControllerProbe.instances = []
    monkeypatch.setattr(perception_control_demo, "ScreenCapture", CaptureProbe)
    monkeypatch.setattr(perception_control_demo, "EasyOCRBackend", OCRProbe)
    monkeypatch.setattr(perception_control_demo, "DesktopController", ControllerProbe)

    result = perception_control_demo.main(
        ["Save", "--execute"],
        input_fn=input_probe,
    )

    assert result == 1
    assert input_probe.prompts == []
    assert ControllerProbe.instances == []


def test_end_to_end_annotation_is_written_only_to_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shot = screenshot()
    output = tmp_path / "nested" / "annotation.png"
    annotation_calls: list[tuple[object, list[OCRDetection], Point]] = []
    write_calls: list[tuple[str, object]] = []

    class CaptureProbe:
        def capture_monitor(self, _monitor_index: int) -> ScreenshotResult:
            return shot

    class OCRProbe:
        def __init__(self, *, gpu: bool | str | None) -> None:
            return None

        def recognize(self, *_args: object, **_kwargs: object) -> list[OCRDetection]:
            return []

    def annotate_probe(
        image: object,
        detections: list[OCRDetection],
        *,
        confidence_cutoff: float,
        origin: Point,
    ) -> object:
        assert confidence_cutoff == 0.8
        annotation_calls.append((image, detections, origin))
        return image

    def write_probe(path: str, image: object) -> bool:
        write_calls.append((path, image))
        return True

    monkeypatch.setattr(perception_control_demo, "ScreenCapture", CaptureProbe)
    monkeypatch.setattr(perception_control_demo, "EasyOCRBackend", OCRProbe)
    monkeypatch.setattr(perception_control_demo, "annotate_detections", annotate_probe)
    monkeypatch.setattr("examples.perception_control_demo.cv2.imwrite", write_probe)

    result = perception_control_demo.main(
        ["Missing", "--annotation", str(output)],
        input_fn=InputProbe(),
    )

    assert result == 0
    assert output.parent.is_dir()
    assert annotation_calls == [(shot.image, [], shot.origin)]
    assert write_calls == [(str(output), shot.image)]
