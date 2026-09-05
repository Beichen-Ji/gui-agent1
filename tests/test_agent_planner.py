import base64
import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel

from examples import model_smoke
from gui_agent.agent.planner import (
    FakePlanner,
    LangChainPlanner,
    MultimodalPlanner,
    PlannerError,
    RemoteImagePermissionError,
)
from gui_agent.agent.prompts import (
    build_action_prompt,
    build_plan_prompt,
    build_replan_prompt,
)
from gui_agent.agent.qwen import QwenTransformersPlanner
from gui_agent.agent.types import (
    AgentDecision,
    AgentState,
    ClickAction,
    DragAction,
    FinishAction,
    Observation,
    PlanProgress,
    ReplanContext,
    ScrollAction,
    StepResult,
    TaskPlan,
    TaskStep,
)
from gui_agent.types import BoundingBox, OCRDetection, Point, ScreenshotResult


def observation() -> Observation:
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    screenshot = ScreenshotResult(
        image=image,
        monitor_index=1,
        captured_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        origin=Point(-100, 20),
    )
    detections = (
        OCRDetection("Save", 0.95, BoundingBox(10, 20, 60, 50)),
        OCRDetection(
            r"C:\Users\student\secret.txt sk-example123456",
            0.80,
            BoundingBox(70, 20, 300, 50),
        ),
    )
    return Observation(screenshot=screenshot, detections=detections, step_index=0)


def task_plan() -> TaskPlan:
    return TaskPlan(
        goal="Open the browser",
        steps=(
            TaskStep(id="step-1", description="Click the browser icon"),
            TaskStep(id="step-2", description="Confirm the window opened"),
        ),
    )


def test_plan_prompt_contains_goal_screen_ocr_and_action_allowlist() -> None:
    prompt = build_plan_prompt("Open the browser", observation())

    assert "Open the browser" in prompt
    assert "800x600" in prompt
    assert "origin=(-100,20)" in prompt
    assert 'text="Save"' in prompt
    for action_kind in ("click", "type_text", "hotkey", "scroll", "drag", "wait", "finish"):
        assert action_kind in prompt


def test_prompt_redacts_absolute_paths_and_likely_api_keys() -> None:
    prompt = build_plan_prompt("Inspect the visible screen", observation())

    assert r"C:\Users\student\secret.txt" not in prompt
    assert "sk-example123456" not in prompt
    assert "[local-path]" in prompt
    assert "[secret]" in prompt


def test_action_prompt_includes_plan_step_and_recent_result() -> None:
    first_decision = AgentDecision(
        current_step_id="step-1",
        rationale_summary="The icon is visible",
        action=ClickAction(x=20, y=30),
        expected_outcome="The browser window appears",
    )
    result = StepResult(
        step_index=0,
        action=first_decision.action,
        status="dry_run",
        message="Dry-run click recorded",
    )
    state = AgentState(
        goal="Open the browser",
        plan=task_plan(),
        observation=observation(),
        decisions=(first_decision,),
        results=(result,),
    )

    prompt = build_action_prompt(state)

    assert "step-1" in prompt
    assert "Click the browser icon" in prompt
    assert "dry_run" in prompt
    assert "Dry-run click recorded" in prompt
    assert "step_index=0" in prompt


def test_action_and_replan_prompts_send_only_completed_and_active_progress() -> None:
    plan = TaskPlan(
        goal="Open the browser",
        steps=(
            TaskStep(id="step-1", description="Open the app"),
            TaskStep(id="step-2", description="Use the active control"),
            TaskStep(id="step-3", description="Sensitive future detail"),
        ),
    )
    progress = PlanProgress.from_plan(plan).select_step(
        "step-2",
        verified_step_id="step-1",
    )
    failure = ReplanContext(
        reason_code="no_visual_change",
        summary="The expected panel did not appear",
    )
    state = AgentState(
        goal=plan.goal,
        plan=plan,
        progress=progress,
        replan_context=failure,
        observation=observation(),
        decisions=(),
        results=(),
    )

    action_prompt = build_action_prompt(state)
    replan_prompt = build_replan_prompt(state, failure)

    for prompt in (action_prompt, replan_prompt):
        assert "step-1" in prompt
        assert "step-2" in prompt
        assert "Sensitive future detail" not in prompt
        assert "no_visual_change" in prompt


def test_fake_planner_returns_configured_plan_and_decisions_in_order() -> None:
    plan = task_plan()
    decisions = (
        AgentDecision(
            current_step_id="step-1",
            rationale_summary="The icon is visible",
            action=ClickAction(x=20, y=30),
            expected_outcome="The browser window appears",
        ),
        AgentDecision(
            current_step_id="step-2",
            rationale_summary="The browser is open",
            action=FinishAction(success=True, summary="Browser opened"),
            expected_outcome="The run stops successfully",
        ),
    )
    planner = FakePlanner(plan=plan, decisions=decisions)
    initial = observation()

    assert isinstance(planner, MultimodalPlanner)
    assert planner.create_plan("Open the browser", initial) == plan
    state = AgentState(
        goal="Open the browser",
        plan=plan,
        observation=initial,
        decisions=(),
        results=(),
    )
    assert planner.next_action(state) == decisions[0]
    assert planner.next_action(state) == decisions[1]


def test_fake_planner_reports_decision_queue_exhaustion() -> None:
    plan = task_plan()
    planner = FakePlanner(plan=plan, decisions=())
    state = AgentState(
        goal="Open the browser",
        plan=plan,
        observation=observation(),
        decisions=(),
        results=(),
    )

    with pytest.raises(PlannerError, match="no configured decision"):
        planner.next_action(state)


class StructuredChatProbe:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.schemas: list[type[BaseModel]] = []
        self.invocations: list[object] = []

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **_kwargs: object,
    ) -> "StructuredChatProbe":
        self.schemas.append(schema)
        return self

    def invoke(self, messages: object) -> object:
        self.invocations.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_langchain_planner_sends_png_data_url_and_validates_structured_plan() -> None:
    expected = task_plan()
    chat = StructuredChatProbe([expected.model_dump(mode="json")])
    planner = LangChainPlanner(
        model_name="local-compatible-model",
        chat_model=chat,
        allow_remote_image=True,
    )

    result = planner.create_plan("Open the browser", observation())

    assert result == expected
    assert chat.schemas == [TaskPlan]
    messages = cast(list[dict[str, Any]], chat.invocations[0])
    content = cast(list[dict[str, Any]], messages[1]["content"])
    image_url = cast(str, content[1]["image_url"]["url"])
    assert image_url.startswith("data:image/png;base64,")
    encoded = image_url.partition(",")[2]
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    assert decoded.size == (800, 600)
    assert decoded.getpixel((0, 0)) == (30, 20, 10)


def test_langchain_planner_refuses_image_without_explicit_permission() -> None:
    chat = StructuredChatProbe([task_plan()])
    planner = LangChainPlanner(
        model_name="remote-model",
        chat_model=chat,
        allow_remote_image=False,
    )

    with pytest.raises(RemoteImagePermissionError, match="allow_remote_image"):
        planner.create_plan("Open the browser", observation())

    assert chat.invocations == []


def test_langchain_planner_preserves_provider_error_as_cause() -> None:
    provider_error = RuntimeError("provider offline")
    chat = StructuredChatProbe([provider_error])
    planner = LangChainPlanner(
        model_name="remote-model",
        chat_model=chat,
        allow_remote_image=True,
    )

    with pytest.raises(PlannerError, match="TaskPlan") as captured:
        planner.create_plan("Open the browser", observation())

    assert captured.value.__cause__ is provider_error


def test_langchain_planner_rejects_malformed_structured_output() -> None:
    chat = StructuredChatProbe([{"goal": "Open the browser", "steps": []}])
    planner = LangChainPlanner(
        model_name="remote-model",
        chat_model=chat,
        allow_remote_image=True,
    )

    with pytest.raises(PlannerError, match="TaskPlan") as captured:
        planner.create_plan("Open the browser", observation())

    assert captured.value.__cause__ is not None


class BatchProbe(dict[str, object]):
    def __init__(self) -> None:
        super().__init__(input_ids=np.array([[1, 2, 3]], dtype=np.int64))
        self.device: object | None = None

    def to(self, device: object) -> "BatchProbe":
        self.device = device
        return self


class ProcessorProbe:
    def __init__(self, decoded: list[str]) -> None:
        self.decoded = decoded
        self.messages: list[object] = []
        self.inputs: list[BatchProbe] = []

    def apply_chat_template(self, messages: object, **_kwargs: object) -> BatchProbe:
        self.messages.append(messages)
        inputs = BatchProbe()
        self.inputs.append(inputs)
        return inputs

    def batch_decode(self, _tokens: object, **_kwargs: object) -> list[str]:
        return [self.decoded.pop(0)]


class ModelProbe:
    device = "cuda:0"

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[np.ndarray[Any, np.dtype[np.int64]]]:
        self.generate_calls.append(kwargs)
        return [np.array([1, 2, 3, 4], dtype=np.int64)]


def test_qwen_planner_uses_in_memory_resized_image_and_parses_json() -> None:
    plan = task_plan()
    finish = AgentDecision(
        current_step_id="step-2",
        rationale_summary="The browser is open",
        action=FinishAction(success=True, summary="Browser opened"),
        expected_outcome="The run stops successfully",
    )
    processor = ProcessorProbe([plan.model_dump_json(), finish.model_dump_json()])
    model = ModelProbe()
    planner = QwenTransformersPlanner(
        processor=processor,
        model=model,
        max_image_side=400,
        max_new_tokens=256,
    )
    initial = observation()

    assert planner.create_plan("Open the browser", initial) == plan
    state = AgentState(
        goal="Open the browser",
        plan=plan,
        observation=initial,
        decisions=(),
        results=(),
    )
    assert planner.next_action(state) == finish
    messages = cast(list[dict[str, Any]], processor.messages[0])
    assert messages[0]["role"] == "system"
    assert "week4-baseline" in cast(str, messages[0]["content"])
    content = cast(list[dict[str, Any]], messages[1]["content"])
    image = cast(Image.Image, content[0]["image"])
    assert image.size == (400, 300)
    assert content[0]["type"] == "image"
    assert processor.inputs[0].device == "cuda:0"
    assert model.generate_calls[0]["max_new_tokens"] == 256
    assert model.generate_calls[0]["do_sample"] is False


@pytest.mark.parametrize(
    ("relative_action", "absolute_action"),
    [
        (
            ClickAction(x=58, y=180),
            ClickAction(x=-54, y=128),
        ),
        (
            ScrollAction(clicks=-3, x=500, y=500),
            ScrollAction(clicks=-3, x=300, y=320),
        ),
        (
            ScrollAction(clicks=2),
            ScrollAction(clicks=2),
        ),
        (
            DragAction(
                start_x=100,
                start_y=200,
                end_x=900,
                end_y=800,
                duration=0.75,
            ),
            DragAction(
                start_x=-20,
                start_y=140,
                end_x=620,
                end_y=500,
                duration=0.75,
            ),
        ),
    ],
)
def test_qwen_planner_converts_relative_pointer_coordinates_to_desktop_pixels(
    relative_action: ClickAction | ScrollAction | DragAction,
    absolute_action: ClickAction | ScrollAction | DragAction,
) -> None:
    decision = AgentDecision(
        current_step_id="step-1",
        rationale_summary="The target is visible",
        action=relative_action,
        expected_outcome="The visible target changes",
    )
    processor = ProcessorProbe([decision.model_dump_json()])
    planner = QwenTransformersPlanner(processor=processor, model=ModelProbe())
    state = AgentState(
        goal="Open the browser",
        plan=task_plan(),
        observation=observation(),
        decisions=(),
        results=(),
    )

    result = planner.next_action(state)

    assert result.action == absolute_action
    messages = cast(list[dict[str, Any]], processor.messages[0])
    content = cast(list[dict[str, Any]], messages[1]["content"])
    prompt = cast(str, content[1]["text"])
    assert "image-relative" in prompt
    assert "1000x1000" in prompt


@pytest.mark.parametrize(
    ("relative_action", "absolute_action"),
    [
        (
            ClickAction(x=0, y=999),
            ClickAction(x=-100, y=619),
        ),
        (
            ScrollAction(clicks=-3, x=999, y=0),
            ScrollAction(clicks=-3, x=699, y=20),
        ),
        (
            DragAction(
                start_x=0,
                start_y=999,
                end_x=999,
                end_y=0,
            ),
            DragAction(
                start_x=-100,
                start_y=619,
                end_x=699,
                end_y=20,
            ),
        ),
    ],
)
def test_qwen_planner_maps_grid_endpoints_using_original_image_after_resize(
    relative_action: ClickAction | ScrollAction | DragAction,
    absolute_action: ClickAction | ScrollAction | DragAction,
) -> None:
    decision = AgentDecision(
        current_step_id="step-1",
        rationale_summary="The target is at an image boundary",
        action=relative_action,
        expected_outcome="The boundary target changes",
    )
    processor = ProcessorProbe([decision.model_dump_json()])
    planner = QwenTransformersPlanner(
        processor=processor,
        model=ModelProbe(),
        max_image_side=400,
    )
    state = AgentState(
        goal="Open the browser",
        plan=task_plan(),
        observation=observation(),
        decisions=(),
        results=(),
    )

    result = planner.next_action(state)

    assert result.action == absolute_action
    messages = cast(list[dict[str, Any]], processor.messages[0])
    content = cast(list[dict[str, Any]], messages[1]["content"])
    image = cast(Image.Image, content[0]["image"])
    assert image.size == (400, 300)


@pytest.mark.parametrize("coordinate", [-1, 1000])
def test_qwen_planner_rejects_pointer_coordinates_outside_relative_grid(
    coordinate: int,
) -> None:
    decision = AgentDecision(
        current_step_id="step-1",
        rationale_summary="The target is visible",
        action=ClickAction(x=coordinate, y=500),
        expected_outcome="The visible target changes",
    )
    planner = QwenTransformersPlanner(
        processor=ProcessorProbe([decision.model_dump_json()]),
        model=ModelProbe(),
    )
    state = AgentState(
        goal="Open the browser",
        plan=task_plan(),
        observation=observation(),
        decisions=(),
        results=(),
    )

    with pytest.raises(PlannerError, match="AgentDecision") as captured:
        planner.next_action(state)

    assert isinstance(captured.value.__cause__, ValueError)
    assert "between 0 and 999" in str(captured.value.__cause__)


@pytest.mark.parametrize(
    "action",
    [ScrollAction(clicks=-3, x=500), ScrollAction(clicks=-3, y=500)],
)
def test_qwen_planner_rejects_partial_scroll_coordinates(
    action: ScrollAction,
) -> None:
    decision = AgentDecision(
        current_step_id="step-1",
        rationale_summary="The scroll area is visible",
        action=action,
        expected_outcome="The visible content moves",
    )
    planner = QwenTransformersPlanner(
        processor=ProcessorProbe([decision.model_dump_json()]),
        model=ModelProbe(),
    )
    state = AgentState(
        goal="Open the browser",
        plan=task_plan(),
        observation=observation(),
        decisions=(),
        results=(),
    )

    with pytest.raises(PlannerError, match="AgentDecision") as captured:
        planner.next_action(state)

    assert isinstance(captured.value.__cause__, ValueError)
    assert "scroll x and y must be provided together" in str(
        captured.value.__cause__
    )


def test_qwen_planner_loads_model_lazily_without_requiring_torch_for_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []
    processor = ProcessorProbe([task_plan().model_dump_json()])
    model = ModelProbe()
    dtype_marker = object()
    monkeypatch.setitem(cast(dict[str, Any], sys.modules), "torch", None)

    def processor_loader(model_name: str, **kwargs: object) -> object:
        calls.append(("processor", model_name, dict(kwargs)))
        return processor

    def model_loader(model_name: str, **kwargs: object) -> object:
        calls.append(("model", model_name, dict(kwargs)))
        return model

    planner = QwenTransformersPlanner(
        processor_loader=processor_loader,
        model_loader=model_loader,
        model_dtype=dtype_marker,
    )
    assert calls == []

    assert planner.create_plan("Open the browser", observation()) == task_plan()
    assert calls[0] == ("processor", "Qwen/Qwen3-VL-4B-Instruct", {})
    assert calls[1][0:2] == ("model", "Qwen/Qwen3-VL-4B-Instruct")
    assert calls[1][2]["device_map"] == "auto"
    assert calls[1][2]["dtype"] is dtype_marker


def _adapter_output(
    tmp_path: Path,
    *,
    base_model: str = "Qwen/Qwen3-VL-4B-Instruct",
    prompt_profile: str = "week5-grounded",
    coordinate_grid_size: int = 1000,
) -> Path:
    output = tmp_path / "training-output"
    adapter = output / "adapter"
    adapter.mkdir(parents=True)
    weight = b"adapter"
    config = json.dumps({"base_model_name_or_path": base_model}).encode()
    (adapter / "adapter_model.safetensors").write_bytes(weight)
    (adapter / "adapter_config.json").write_bytes(config)
    manifest = {
        "kind": "gui-agent-week5-training-run",
        "base_model": base_model,
        "prompt_profile": prompt_profile,
        "coordinate_grid_size": coordinate_grid_size,
        "output_sha256": {
            "adapter/adapter_model.safetensors": hashlib.sha256(weight).hexdigest(),
            "adapter/adapter_config.json": hashlib.sha256(config).hexdigest(),
        },
    }
    (output / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return output


def test_qwen_adapter_path_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="adapter path"):
        QwenTransformersPlanner(adapter_path=tmp_path / "missing")


def test_qwen_adapter_manifest_must_match_the_base_model(tmp_path: Path) -> None:
    output = _adapter_output(tmp_path, base_model="Qwen/different-model")

    with pytest.raises(ValueError, match="base model"):
        QwenTransformersPlanner(adapter_path=output)


def test_qwen_adapter_manifest_must_match_grid_and_explicit_prompt(tmp_path: Path) -> None:
    wrong_grid = _adapter_output(tmp_path / "grid", coordinate_grid_size=999)
    with pytest.raises(ValueError, match="coordinate grid"):
        QwenTransformersPlanner(adapter_path=wrong_grid)

    output = _adapter_output(tmp_path / "prompt")
    with pytest.raises(ValueError, match="prompt profile"):
        QwenTransformersPlanner(
            adapter_path=output,
            prompt_profile="week4-baseline",
        )


def test_qwen_adapter_manifest_detects_weight_tampering(tmp_path: Path) -> None:
    output = _adapter_output(tmp_path)
    (output / "adapter" / "adapter_model.safetensors").write_bytes(b"modified")

    with pytest.raises(ValueError, match="hash mismatch"):
        QwenTransformersPlanner(adapter_path=output)


def test_qwen_loads_validated_adapter_after_the_base_model(tmp_path: Path) -> None:
    output = _adapter_output(tmp_path)
    plan = task_plan()
    processor = ProcessorProbe([plan.model_dump_json()])
    base_model = ModelProbe()
    adapted_model = ModelProbe()
    calls: list[str] = []

    def processor_loader(_model_name: str, **_kwargs: object) -> object:
        calls.append("processor")
        return processor

    def model_loader(_model_name: str, **_kwargs: object) -> object:
        calls.append("base")
        return base_model

    def adapter_loader(model: object, path: Path) -> object:
        calls.append("adapter")
        assert model is base_model
        assert path == output / "adapter"
        return adapted_model

    planner = QwenTransformersPlanner(
        adapter_path=output,
        processor_loader=processor_loader,
        model_loader=model_loader,
        adapter_loader=adapter_loader,
        model_dtype=object(),
    )

    assert planner.create_plan("Open the browser", observation()) == plan
    assert calls == ["processor", "base", "adapter"]
    messages = cast(list[dict[str, Any]], processor.messages[0])
    assert "week5-grounded" in cast(str, messages[0]["content"])


def test_qwen_preserves_adapter_loading_error_as_cause(tmp_path: Path) -> None:
    output = _adapter_output(tmp_path)
    adapter_error = RuntimeError("incompatible adapter tensors")

    def adapter_loader(_model: object, _path: Path) -> object:
        raise adapter_error

    planner = QwenTransformersPlanner(
        adapter_path=output,
        processor_loader=lambda *_args, **_kwargs: ProcessorProbe([]),
        model_loader=lambda *_args, **_kwargs: ModelProbe(),
        adapter_loader=adapter_loader,
        model_dtype=object(),
    )

    with pytest.raises(PlannerError, match="adapter") as captured:
        planner.create_plan("Open the browser", observation())

    assert captured.value.__cause__ is adapter_error


def test_qwen_planner_preserves_malformed_json_as_cause() -> None:
    planner = QwenTransformersPlanner(
        processor=ProcessorProbe(["not json"]),
        model=ModelProbe(),
    )

    with pytest.raises(PlannerError, match="TaskPlan") as captured:
        planner.create_plan("Open the browser", observation())

    assert captured.value.__cause__ is not None


def test_model_smoke_fake_provider_uses_only_synthetic_observation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = model_smoke.main(["--provider", "fake", "--synthetic"])

    output = capsys.readouterr().out
    assert result == 0
    assert '"goal": "Open the synthetic browser"' in output
    assert '"kind": "click"' in output


def test_model_smoke_requires_synthetic_mode() -> None:
    with pytest.raises(SystemExit) as captured:
        model_smoke.main(["--provider", "fake"])

    assert captured.value.code == 2
