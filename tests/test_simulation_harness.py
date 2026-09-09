from pathlib import Path

import numpy as np
import pytest

from gui_agent.agent.policy import ActionDeniedError
from gui_agent.agent.types import (
    ClickAction,
    DragAction,
    FinishAction,
    HotkeyAction,
    ScrollAction,
    TypeTextAction,
    WaitAction,
)
from gui_agent.simulation.harness import (
    SimulatedActionExecutor,
    SimulatedDesktop,
    SimulatedObservationSource,
    SimulationPolicy,
)
from gui_agent.simulation.state import TestbedState
from gui_agent.types import BoundingBox, OCRDetection, Point

_ORIGIN = Point(0, 0)


class ProbeOCR:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, Point, float]] = []

    def recognize(
        self,
        image: np.ndarray,
        *,
        origin: Point = _ORIGIN,
        min_confidence: float = 0.0,
    ) -> list[OCRDetection]:
        self.calls.append((image, origin, min_confidence))
        return [OCRDetection("probe", 0.9, BoundingBox(10, 10, 80, 35))]


def _desktop(tmp_path: Path, *, fault_profile: str = "none") -> SimulatedDesktop:
    return SimulatedDesktop(
        TestbedState(
            tmp_path / "desktop",
            fault_profile=fault_profile,  # type: ignore[arg-type]
        ),
        canvas=(1280, 720),
    )


def test_oracle_observation_uses_rendered_ground_truth_boxes(tmp_path: Path) -> None:
    desktop = _desktop(tmp_path)
    observation = SimulatedObservationSource(desktop, mode="oracle").observe(3)

    assert observation.step_index == 3
    assert observation.screenshot.origin == Point(0, 0)
    assert observation.screenshot.image.shape == (720, 1280, 3)
    assert {detection.text for detection in observation.detections} == set(
        desktop.hitboxes
    )
    assert {detection.box for detection in observation.detections} == set(
        desktop.hitboxes.values()
    )
    assert all(detection.confidence == 1.0 for detection in observation.detections)


def test_ocr_observation_calls_injected_backend(tmp_path: Path) -> None:
    desktop = _desktop(tmp_path)
    ocr = ProbeOCR()

    observation = SimulatedObservationSource(
        desktop,
        mode="ocr",
        ocr=ocr,
        min_confidence=0.6,
    ).observe(0)

    assert [detection.text for detection in observation.detections] == ["probe"]
    assert len(ocr.calls) == 1
    image, origin, threshold = ocr.calls[0]
    assert image is observation.screenshot.image
    assert origin == Point(0, 0)
    assert threshold == 0.6


def test_click_hit_changes_state_but_click_miss_does_not(tmp_path: Path) -> None:
    desktop = _desktop(tmp_path)
    executor = SimulatedActionExecutor(desktop)
    SimulatedObservationSource(desktop, mode="oracle").observe(0)
    search_box = desktop.hitboxes["browser.search"]

    hit = executor.execute(
        ClickAction(x=search_box.center.x, y=search_box.center.y),
        step_index=0,
    )
    before_miss = desktop.state.snapshot()
    miss = executor.execute(ClickAction(x=10, y=200), step_index=1)

    assert hit.status == "executed"
    assert desktop.state.focused_control == "browser.search"
    assert miss.status == "executed"
    assert "missed" in miss.message
    assert desktop.state.snapshot() == before_miss


def test_executor_handles_every_action_without_ever_using_dry_run(tmp_path: Path) -> None:
    desktop = _desktop(tmp_path)
    executor = SimulatedActionExecutor(desktop)
    observer = SimulatedObservationSource(desktop, mode="oracle")
    observer.observe(0)
    search_box = desktop.hitboxes["browser.search"]
    results = [
        executor.execute(
            ClickAction(x=search_box.center.x, y=search_box.center.y),
            step_index=0,
        ),
        executor.execute(TypeTextAction(text="query"), step_index=1),
        executor.execute(HotkeyAction(keys=("ctrl", "a")), step_index=2),
        executor.execute(TypeTextAction(text="replacement"), step_index=3),
        executor.execute(WaitAction(seconds=0.1), step_index=4),
        executor.execute(FinishAction(success=True, summary="done"), step_index=5),
    ]

    desktop.state.activate_app("settings")
    observer.observe(6)
    slider = desktop.hitboxes["settings.volume"]
    results.append(
        executor.execute(
            DragAction(
                start_x=slider.left + 1,
                start_y=slider.center.y,
                end_x=slider.right - 1,
                end_y=slider.center.y,
            ),
            step_index=6,
        )
    )
    desktop.state.activate_app("editor")
    results.append(executor.execute(ScrollAction(clicks=-2), step_index=7))

    assert all(result.status == "executed" for result in results)
    assert desktop.state.settings_volume > 0.99
    assert desktop.state.editor_scroll == -2


def test_policy_rejects_out_of_bounds_without_calling_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desktop = _desktop(tmp_path)
    observation = SimulatedObservationSource(desktop, mode="oracle").observe(0)

    def unexpected_input(_prompt: str) -> str:
        raise AssertionError("simulation policy must not ask for confirmation")

    monkeypatch.setattr("builtins.input", unexpected_input)
    policy = SimulationPolicy()
    policy.authorize(
        ClickAction(x=100, y=100),
        observation,
        expected_outcome="focus a control",
    )
    with pytest.raises(ActionDeniedError, match="outside"):
        policy.authorize(
            ClickAction(x=1280, y=720),
            observation,
            expected_outcome="reject the point",
        )


def test_executor_returns_failed_for_out_of_bounds_or_unsupported_context(
    tmp_path: Path,
) -> None:
    desktop = _desktop(tmp_path)
    executor = SimulatedActionExecutor(desktop)

    outside = executor.execute(ClickAction(x=-1, y=0), step_index=0)
    unsupported = executor.execute(ScrollAction(clicks=1), step_index=1)

    assert outside.status == "failed"
    assert unsupported.status == "failed"


def test_wait_advances_clock_and_completes_delayed_search(tmp_path: Path) -> None:
    desktop = _desktop(tmp_path, fault_profile="delayed")
    desktop.state.search("eventually ready")
    clock_calls: list[float] = []
    executor = SimulatedActionExecutor(desktop, clock=clock_calls.append)

    result = executor.execute(WaitAction(seconds=0.75), step_index=0)

    assert result.status == "executed"
    assert clock_calls == [0.75]
    assert desktop.elapsed_seconds == pytest.approx(0.75)
    assert desktop.state.search_result == "Search result: eventually ready"
