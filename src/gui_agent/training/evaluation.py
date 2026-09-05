import hashlib
import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from gui_agent.agent.coordinates import action_to_grid
from gui_agent.agent.prompts import get_prompt_profile
from gui_agent.agent.qwen import QwenTransformersPlanner
from gui_agent.agent.types import (
    AgentAction,
    AgentState,
    ClickAction,
    DragAction,
    FinishAction,
    HotkeyAction,
    Observation,
    ScrollAction,
    TaskPlan,
    TypeTextAction,
    WaitAction,
)
from gui_agent.training.config import validate_training_output_path
from gui_agent.types import BoundingBox, OCRDetection, Point, ScreenRegion, ScreenshotResult

_POINTER_TOLERANCE = 50


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvaluationElement(_StrictFrozenModel):
    label: str = Field(min_length=1, max_length=120)
    box: tuple[int, int, int, int]
    fill: str = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _positive_box(self) -> "EvaluationElement":
        left, top, right, bottom = self.box
        if right <= left or bottom <= top:
            raise ValueError("element box must have positive width and height")
        return self


class EvaluationCase(_StrictFrozenModel):
    id: str = Field(min_length=1, max_length=80)
    canvas: tuple[int, int]
    instruction: str = Field(min_length=1, max_length=1000)
    plan_keywords: tuple[str, ...] = Field(min_length=1)
    elements: tuple[EvaluationElement, ...]
    expected_action: AgentAction
    target_box: tuple[int, int, int, int] | None = None

    @field_validator("plan_keywords")
    @classmethod
    def _valid_keywords(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(keyword.strip().lower() for keyword in value)
        if any(not keyword for keyword in normalized):
            raise ValueError("plan keywords must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("plan keywords must be unique")
        return normalized

    @model_validator(mode="after")
    def _valid_geometry(self) -> "EvaluationCase":
        width, height = self.canvas
        if width <= 0 or height <= 0:
            raise ValueError("canvas dimensions must be positive")
        for element in self.elements:
            left, top, right, bottom = element.box
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError("element box must be inside the canvas")
        if self.target_box is not None:
            left, top, right, bottom = self.target_box
            if not (0 <= left <= right <= 999 and 0 <= top <= bottom <= 999):
                raise ValueError("target box must be inside the 0-999 coordinate grid")
        if isinstance(self.expected_action, ClickAction) != (self.target_box is not None):
            raise ValueError("exactly click cases require a target box")
        return self


class EvaluationCaseSet(_StrictFrozenModel):
    schema_version: Literal[1]
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def _unique_case_ids(cls, value: tuple[EvaluationCase, ...]) -> tuple[EvaluationCase, ...]:
        ids = [case.id for case in value]
        if len(set(ids)) != len(ids):
            raise ValueError("evaluation case IDs must be unique")
        return value


class EvaluationCondition(_StrictFrozenModel):
    id: str = Field(default="", max_length=120)
    model: str = Field(min_length=1, max_length=500)
    prompt_profile: str = Field(min_length=1, max_length=80)
    adapter_label: str | None = Field(default=None, max_length=500)
    adapter_manifest_sha256: str | None = None
    adapter_weight_sha256: str | None = None

    @field_validator("adapter_manifest_sha256", "adapter_weight_sha256")
    @classmethod
    def _optional_sha256(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("adapter hashes must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def _complete_adapter_provenance(self) -> "EvaluationCondition":
        provenance = (
            self.adapter_label,
            self.adapter_manifest_sha256,
            self.adapter_weight_sha256,
        )
        if any(value is not None for value in provenance) and any(
            value is None for value in provenance
        ):
            raise ValueError("adapter provenance must be complete")
        object.__setattr__(self, "id", self._condition_id())
        return self

    def _condition_id(self) -> str:
        payload = json.dumps(
            {
                "adapter_manifest_sha256": self.adapter_manifest_sha256,
                "adapter_weight_sha256": self.adapter_weight_sha256,
                "model": self.model,
                "prompt_profile": self.prompt_profile,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        prefix = "adapter" if self.adapter_manifest_sha256 is not None else "base"
        digest = hashlib.sha256(payload).hexdigest()
        return f"{prefix}-{self.prompt_profile}-{digest}"


class EvaluationPrediction(_StrictFrozenModel):
    condition_id: str = Field(min_length=1, max_length=120)
    case_id: str = Field(min_length=1, max_length=80)
    plan: TaskPlan | None = None
    action: AgentAction | None = None
    failure_type: str | None = None
    latency_ms: float = Field(ge=0.0)
    peak_vram_mib: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _consistent_result(self) -> "EvaluationPrediction":
        successful = self.plan is not None and self.action is not None
        if successful == (self.failure_type is not None):
            raise ValueError("prediction must contain either a result or a failure")
        return self


class EvaluationMetrics(_StrictFrozenModel):
    schema_valid_rate: float = Field(ge=0.0, le=1.0)
    plan_requirement_recall: float = Field(ge=0.0, le=1.0)
    action_kind_accuracy: float = Field(ge=0.0, le=1.0)
    action_parameter_accuracy: float = Field(ge=0.0, le=1.0)
    click_hit_rate: float = Field(ge=0.0, le=1.0)
    median_latency_ms: float = Field(ge=0.0)
    peak_vram_mib: float = Field(ge=0.0)


class EvaluationOutcome(_StrictFrozenModel):
    case_id: str
    schema_valid: bool
    plan_requirement_recall: float
    action_kind_correct: bool
    action_parameters_correct: bool
    click_hit: bool | None
    latency_ms: float
    peak_vram_mib: float
    failure_type: str | None
    predicted_action: AgentAction | None


class EvaluationReport(_StrictFrozenModel):
    kind: Literal["gui-agent-week5-evaluation"] = "gui-agent-week5-evaluation"
    cases_sha256: str
    condition: EvaluationCondition
    case_count: int = Field(ge=1)
    pointer_tolerance_grid_units: Literal[50] = 50
    metrics: EvaluationMetrics
    failures: dict[str, int]
    outcomes: tuple[EvaluationOutcome, ...]

    @field_validator("cases_sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("cases_sha256 must be a lowercase SHA-256 digest")
        return value


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"could not read evaluation cases: {path}") from error
    try:
        case_set = EvaluationCaseSet.model_validate_json(raw)
    except ValueError as error:
        raise ValueError(f"invalid evaluation cases: {path}") from error
    return case_set.cases


def _file_sha256(path: Path, *, label: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"could not read {label}: {path}") from error


def _adapter_provenance(adapter_path: Path) -> tuple[str, str, str]:
    resolved = adapter_path.resolve()
    if (resolved / "run-manifest.json").is_file():
        output_root = resolved
        adapter_dir = resolved / "adapter"
    elif resolved.name == "adapter" and (resolved.parent / "run-manifest.json").is_file():
        output_root = resolved.parent
        adapter_dir = resolved
    else:
        raise ValueError("adapter path requires a sibling or child run-manifest.json")
    try:
        label = output_root.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        label = output_root.name
    return (
        label,
        _file_sha256(output_root / "run-manifest.json", label="adapter run manifest"),
        _file_sha256(
            adapter_dir / "adapter_model.safetensors",
            label="adapter weights",
        ),
    )


def build_evaluation_condition(
    *,
    model_name: str,
    prompt_profile: str,
    adapter_path: Path | None,
) -> EvaluationCondition:
    selected_profile = get_prompt_profile(prompt_profile)
    if adapter_path is None:
        return EvaluationCondition(
            model=model_name,
            prompt_profile=selected_profile.id,
        )
    label, manifest_sha256, weight_sha256 = _adapter_provenance(adapter_path)
    return EvaluationCondition(
        model=model_name,
        prompt_profile=selected_profile.id,
        adapter_label=label,
        adapter_manifest_sha256=manifest_sha256,
        adapter_weight_sha256=weight_sha256,
    )


def _same_pointer(first: int, second: int) -> bool:
    return abs(first - second) <= _POINTER_TOLERANCE


def _parameters_match(expected: AgentAction, actual: AgentAction | None) -> bool:
    if actual is None or expected.kind != actual.kind:
        return False
    if isinstance(expected, ClickAction) and isinstance(actual, ClickAction):
        return (
            _same_pointer(expected.x, actual.x)
            and _same_pointer(expected.y, actual.y)
            and expected.button == actual.button
            and expected.clicks == actual.clicks
        )
    if isinstance(expected, DragAction) and isinstance(actual, DragAction):
        return (
            _same_pointer(expected.start_x, actual.start_x)
            and _same_pointer(expected.start_y, actual.start_y)
            and _same_pointer(expected.end_x, actual.end_x)
            and _same_pointer(expected.end_y, actual.end_y)
            and expected.duration == actual.duration
        )
    if isinstance(expected, ScrollAction) and isinstance(actual, ScrollAction):
        if expected.clicks != actual.clicks:
            return False
        if expected.x is None or expected.y is None:
            return actual.x is None and actual.y is None
        if actual.x is None or actual.y is None:
            return False
        return _same_pointer(expected.x, actual.x) and _same_pointer(expected.y, actual.y)
    if isinstance(expected, TypeTextAction) and isinstance(actual, TypeTextAction):
        return expected.text == actual.text
    if isinstance(expected, HotkeyAction) and isinstance(actual, HotkeyAction):
        return expected.keys == actual.keys
    if isinstance(expected, WaitAction) and isinstance(actual, WaitAction):
        return expected.seconds == actual.seconds
    if isinstance(expected, FinishAction) and isinstance(actual, FinishAction):
        return expected.success == actual.success and expected.summary == actual.summary
    return False


def _plan_recall(case: EvaluationCase, plan: TaskPlan | None) -> float:
    if plan is None:
        return 0.0
    text = " ".join((plan.goal, *(step.description for step in plan.steps))).lower()
    return sum(keyword in text for keyword in case.plan_keywords) / len(case.plan_keywords)


def _macro_average(scores: dict[str, list[float]]) -> float:
    if not scores:
        return 0.0
    return sum(sum(values) / len(values) for values in scores.values()) / len(scores)


def evaluate_predictions(
    cases: Iterable[EvaluationCase],
    condition: EvaluationCondition,
    predictions: Iterable[EvaluationPrediction],
    *,
    cases_sha256: str,
) -> EvaluationReport:
    ordered_cases = tuple(sorted(cases, key=lambda case: case.id))
    ordered_predictions = tuple(sorted(predictions, key=lambda item: item.case_id))
    if not ordered_cases:
        raise ValueError("evaluation requires at least one case")
    if any(prediction.condition_id != condition.id for prediction in ordered_predictions):
        raise ValueError("predictions contain a different evaluation condition")
    prediction_ids = [prediction.case_id for prediction in ordered_predictions]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("evaluation predictions contain duplicate case IDs")
    case_ids = [case.id for case in ordered_cases]
    if prediction_ids != case_ids:
        raise ValueError("evaluation predictions must cover every case exactly once")

    kind_scores: dict[str, list[float]] = defaultdict(list)
    parameter_scores: dict[str, list[float]] = defaultdict(list)
    failures: Counter[str] = Counter()
    outcomes: list[EvaluationOutcome] = []
    click_scores: list[float] = []
    for case, prediction in zip(ordered_cases, ordered_predictions, strict=True):
        schema_valid = (
            prediction.failure_type is None
            and prediction.plan is not None
            and prediction.action is not None
        )
        kind_correct = bool(
            schema_valid and prediction.action is not None
            and prediction.action.kind == case.expected_action.kind
        )
        parameters_correct = schema_valid and _parameters_match(
            case.expected_action,
            prediction.action,
        )
        kind_scores[case.expected_action.kind].append(float(kind_correct))
        parameter_scores[case.expected_action.kind].append(float(parameters_correct))
        recall = _plan_recall(case, prediction.plan) if schema_valid else 0.0
        click_hit: bool | None = None
        if isinstance(case.expected_action, ClickAction):
            assert case.target_box is not None
            click_hit = False
            if schema_valid and isinstance(prediction.action, ClickAction):
                left, top, right, bottom = case.target_box
                click_hit = (
                    left <= prediction.action.x <= right
                    and top <= prediction.action.y <= bottom
                )
            click_scores.append(float(click_hit))
        if prediction.failure_type is not None:
            failures[prediction.failure_type] += 1
        outcomes.append(
            EvaluationOutcome(
                case_id=case.id,
                schema_valid=schema_valid,
                plan_requirement_recall=recall,
                action_kind_correct=kind_correct,
                action_parameters_correct=parameters_correct,
                click_hit=click_hit,
                latency_ms=prediction.latency_ms,
                peak_vram_mib=prediction.peak_vram_mib,
                failure_type=prediction.failure_type,
                predicted_action=prediction.action,
            )
        )

    count = len(ordered_cases)
    metrics = EvaluationMetrics(
        schema_valid_rate=sum(outcome.schema_valid for outcome in outcomes) / count,
        plan_requirement_recall=(
            sum(outcome.plan_requirement_recall for outcome in outcomes) / count
        ),
        action_kind_accuracy=_macro_average(kind_scores),
        action_parameter_accuracy=_macro_average(parameter_scores),
        click_hit_rate=sum(click_scores) / len(click_scores) if click_scores else 0.0,
        median_latency_ms=statistics.median(item.latency_ms for item in ordered_predictions),
        peak_vram_mib=max(item.peak_vram_mib for item in ordered_predictions),
    )
    return EvaluationReport(
        cases_sha256=cases_sha256,
        condition=condition,
        case_count=count,
        metrics=metrics,
        failures=dict(sorted(failures.items())),
        outcomes=tuple(outcomes),
    )


def write_evaluation_report(
    report: EvaluationReport,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("evaluation output must be a JSON file")
    existed = path.exists()
    if existed:
        if not overwrite:
            raise ValueError(f"evaluation output already exists: {path}")
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("refusing to overwrite an unowned evaluation output") from error
        if not isinstance(stored, dict) or stored.get("kind") != report.kind:
            raise ValueError("refusing to overwrite an unowned evaluation output")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    if not existed:
        try:
            with path.open("xb") as output:
                output.write(encoded)
        except FileExistsError as error:
            raise ValueError(f"evaluation output already exists: {path}") from error
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(encoded)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _render_case(case: EvaluationCase) -> Observation:
    width, height = case.canvas
    image = Image.new("RGB", (width, height), "#F5F7FA")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), case.instruction, fill="#111827")
    detections: list[OCRDetection] = []
    for element in case.elements:
        left, top, right, bottom = element.box
        draw.rectangle(element.box, fill=element.fill, outline="#374151", width=2)
        draw.text((left + 8, top + 8), element.label, fill="#111827")
        detections.append(
            OCRDetection(
                text=element.label,
                confidence=1.0,
                box=BoundingBox(left, top, right, bottom),
            )
        )
    rgb = np.asarray(image, dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return Observation(
        screenshot=ScreenshotResult(
            image=bgr,
            monitor_index=None,
            captured_at=datetime.now(UTC),
            origin=Point(0, 0),
        ),
        detections=tuple(detections),
        step_index=0,
    )


def _reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        return


def _peak_vram_mib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except (ImportError, RuntimeError):
        pass
    return 0.0


def run_evaluation(
    *,
    cases_path: Path,
    model_name: str,
    adapter_path: Path | None,
    prompt_profile: str,
    output_path: Path,
    overwrite: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> EvaluationReport:
    selected_profile = get_prompt_profile(prompt_profile)
    resolved_output = validate_training_output_path(output_path, project_root=Path.cwd())
    raw_cases = cases_path.read_bytes()
    cases = load_evaluation_cases(cases_path)
    condition = build_evaluation_condition(
        model_name=model_name,
        prompt_profile=selected_profile.id,
        adapter_path=adapter_path,
    )
    planner = QwenTransformersPlanner(
        model_name=model_name,
        prompt_profile=selected_profile,
        adapter_path=adapter_path,
    )
    predictions: list[EvaluationPrediction] = []
    for case in cases:
        observation = _render_case(case)
        _reset_cuda_peak()
        started = clock()
        try:
            plan = planner.create_plan(case.instruction, observation)
            decision = planner.next_action(
                AgentState(
                    goal=case.instruction,
                    plan=plan,
                    observation=observation,
                    decisions=(),
                    results=(),
                )
            )
            bounds = ScreenRegion(0, 0, observation.screenshot.width, observation.screenshot.height)
            action = action_to_grid(decision.action, bounds=bounds)
            prediction = EvaluationPrediction(
                condition_id=condition.id,
                case_id=case.id,
                plan=plan,
                action=action,
                latency_ms=(clock() - started) * 1000,
                peak_vram_mib=_peak_vram_mib(),
            )
        except Exception as error:
            prediction = EvaluationPrediction(
                condition_id=condition.id,
                case_id=case.id,
                failure_type=type(error).__name__,
                latency_ms=(clock() - started) * 1000,
                peak_vram_mib=_peak_vram_mib(),
            )
        predictions.append(prediction)
    report = evaluate_predictions(
        cases,
        condition,
        predictions,
        cases_sha256=hashlib.sha256(raw_cases).hexdigest(),
    )
    write_evaluation_report(report, resolved_output, overwrite=overwrite)
    return report


__all__ = [
    "EvaluationCase",
    "EvaluationCondition",
    "EvaluationPrediction",
    "EvaluationReport",
    "build_evaluation_condition",
    "evaluate_predictions",
    "load_evaluation_cases",
    "run_evaluation",
    "write_evaluation_report",
]
