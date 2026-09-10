"""Expand and run reproducible Week 7 simulated evaluation conditions."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias, cast

from pydantic import Field, field_validator, model_validator

from gui_agent._models import StrictFrozenModel
from gui_agent.agent.planner import FakePlanner
from gui_agent.agent.prompts import PROMPT_PROFILES
from gui_agent.agent.qwen import QwenTransformersPlanner
from gui_agent.evaluation.report import EvaluationContext, EvaluationReport
from gui_agent.evaluation.runner import PlannerFactory, run_evaluation
from gui_agent.evaluation.suite import (
    EvaluationTask,
    build_reference_planner,
    load_task_suite,
)
from gui_agent.perception.ocr import EasyOCRBackend
from gui_agent.perception.preprocessing import OCR_PROFILES
from gui_agent.provenance import adapter_provenance, file_sha256
from gui_agent.simulation.harness import ObservationMode

Provider: TypeAlias = Literal["fake", "qwen"]


class EvaluationConditionTemplate(StrictFrozenModel):
    """Define one condition before expanding its requested resolutions."""

    name: str = Field(min_length=1, max_length=120)
    model: str = Field(min_length=1, max_length=500)
    prompt_profile: str
    adapter_path: str | None = Field(default=None, min_length=1, max_length=1000)
    ocr_profile: str
    observation_mode: ObservationMode
    resolutions: tuple[tuple[int, int], ...] = Field(min_length=1)
    seed: int = Field(ge=0)
    per_task_estimated_seconds: float = Field(gt=0.0)

    @field_validator("prompt_profile")
    @classmethod
    def _known_prompt_profile(cls, value: str) -> str:
        if value not in PROMPT_PROFILES:
            raise ValueError(f"unknown prompt profile: {value}")
        return value

    @field_validator("ocr_profile")
    @classmethod
    def _known_ocr_profile(cls, value: str) -> str:
        if value not in OCR_PROFILES:
            raise ValueError(f"unknown OCR profile: {value}")
        return value

    @field_validator("resolutions")
    @classmethod
    def _valid_resolutions(
        cls,
        value: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        if any(width < 320 or height < 180 for width, height in value):
            raise ValueError("condition resolutions must be at least 320x180")
        if len(value) != len(set(value)):
            raise ValueError("condition resolutions must be unique")
        return value

    def effective_key(self) -> tuple[object, ...]:
        """Return fields that determine behavior, excluding the display name."""
        return (
            self.model,
            self.prompt_profile,
            self.adapter_path,
            self.ocr_profile,
            self.observation_mode,
            self.resolutions,
            self.seed,
        )


class EvaluationConditionSet(StrictFrozenModel):
    """Store uniquely named and behaviorally distinct condition templates."""

    schema_version: Literal[1] = 1
    kind: Literal["gui-agent-week7-condition-set"] = "gui-agent-week7-condition-set"
    conditions: tuple[EvaluationConditionTemplate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_conditions(self) -> "EvaluationConditionSet":
        names = [condition.name for condition in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("condition names must be unique")
        keys = [condition.effective_key() for condition in self.conditions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate effective conditions are not allowed")
        return self


class ExpandedCondition(StrictFrozenModel):
    """Bind a deterministic condition identifier to one resolution."""

    id: str = Field(min_length=1, max_length=120)
    name: str
    model: str
    prompt_profile: str
    adapter_path: str | None
    ocr_profile: str
    observation_mode: ObservationMode
    resolution: tuple[int, int]
    seed: int
    per_task_estimated_seconds: float


class EvaluationRunSpec(StrictFrozenModel):
    """Pair one expanded condition with one task and time estimate."""

    id: str = Field(min_length=1, max_length=240)
    condition: ExpandedCondition
    task: EvaluationTask
    estimated_seconds: float = Field(gt=0.0)


def _condition_id(
    condition: EvaluationConditionTemplate,
    resolution: tuple[int, int],
) -> str:
    payload = json.dumps(
        {
            "adapter_path": condition.adapter_path,
            "model": condition.model,
            "observation_mode": condition.observation_mode,
            "ocr_profile": condition.ocr_profile,
            "prompt_profile": condition.prompt_profile,
            "resolution": resolution,
            "seed": condition.seed,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    width, height = resolution
    return f"{condition.name}-{width}x{height}-{digest}"


def expand_conditions(
    condition_set: EvaluationConditionSet,
) -> tuple[ExpandedCondition, ...]:
    """Expand every template-resolution pair with a stable identifier."""
    expanded = tuple(
        ExpandedCondition(
            id=_condition_id(condition, resolution),
            name=condition.name,
            model=condition.model,
            prompt_profile=condition.prompt_profile,
            adapter_path=condition.adapter_path,
            ocr_profile=condition.ocr_profile,
            observation_mode=condition.observation_mode,
            resolution=resolution,
            seed=condition.seed,
            per_task_estimated_seconds=condition.per_task_estimated_seconds,
        )
        for condition in condition_set.conditions
        for resolution in condition.resolutions
    )
    ids = [condition.id for condition in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded condition IDs must be unique")
    return expanded


def expand_run_matrix(
    tasks: Sequence[EvaluationTask],
    condition_set: EvaluationConditionSet,
) -> tuple[EvaluationRunSpec, ...]:
    """Build the deterministic Cartesian product of tasks and conditions."""
    runs = tuple(
        EvaluationRunSpec(
            id=f"{condition.id}:{task.id}",
            condition=condition,
            task=task,
            estimated_seconds=condition.per_task_estimated_seconds,
        )
        for condition in expand_conditions(condition_set)
        for task in tasks
    )
    ids = [run.id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded evaluation run IDs must be unique")
    return runs


def load_condition_set(path: Path) -> EvaluationConditionSet:
    """Load and strictly validate a versioned evaluation condition file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"could not read evaluation conditions: {path}") from error
    try:
        return EvaluationConditionSet.model_validate_json(raw)
    except ValueError as error:
        raise ValueError(f"invalid evaluation conditions: {path}") from error


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "unknown"


def _environment() -> dict[str, str]:
    packages = {
        "easyocr": "easyocr",
        "opencv": "opencv-python-headless",
        "torch": "torch",
        "transformers": "transformers",
    }
    versions = {"python": sys.version.split()[0]}
    for label, distribution in packages.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "not-installed"
    return versions


def _context(
    condition: ExpandedCondition,
    *,
    provider: Provider,
    suite_path: Path,
    conditions_path: Path,
) -> EvaluationContext:
    adapter_label: str | None = None
    manifest_hash: str | None = None
    weight_hash: str | None = None
    if provider == "qwen" and condition.adapter_path is not None:
        adapter_label, manifest_hash, weight_hash = adapter_provenance(
            Path(condition.adapter_path)
        )
    return EvaluationContext(
        suite_sha256=file_sha256(suite_path, label="evaluation task suite"),
        conditions_sha256=file_sha256(
            conditions_path,
            label="evaluation conditions",
        ),
        git_revision=_git_revision(),
        model="fake-reference" if provider == "fake" else condition.model,
        adapter_label=adapter_label,
        adapter_manifest_sha256=manifest_hash,
        adapter_weight_sha256=weight_hash,
        prompt_profile=condition.prompt_profile,
        ocr_profile=condition.ocr_profile,
        observation_mode=("oracle" if provider == "fake" else condition.observation_mode),
        seed=condition.seed,
        environment=_environment(),
    )


def _execute_condition(
    tasks: tuple[EvaluationTask, ...],
    condition: ExpandedCondition,
    *,
    provider: Provider,
    suite_path: Path,
    conditions_path: Path,
    output_dir: Path,
    resume: bool,
) -> EvaluationReport:
    context = _context(
        condition,
        provider=provider,
        suite_path=suite_path,
        conditions_path=conditions_path,
    )
    if provider == "fake":
        def fake_planner_factory(
            task: EvaluationTask,
            resolution: tuple[int, int],
        ) -> FakePlanner:
            return build_reference_planner(task, canvas=resolution)

        planner_factory: PlannerFactory = fake_planner_factory
        ocr = None
    else:
        planner = QwenTransformersPlanner(
            model_name=condition.model,
            prompt_profile=condition.prompt_profile,
            adapter_path=(Path(condition.adapter_path) if condition.adapter_path else None),
        )
        def qwen_planner_factory(
            _task: EvaluationTask,
            _resolution: tuple[int, int],
        ) -> QwenTransformersPlanner:
            return planner

        planner_factory = qwen_planner_factory
        ocr = (
            EasyOCRBackend(profile=condition.ocr_profile)
            if condition.observation_mode == "ocr"
            else None
        )
    return run_evaluation(
        tasks,
        context=context,
        resolution=condition.resolution,
        planner_factory=planner_factory,
        output_dir=output_dir / condition.id,
        ocr=ocr,
        resume=resume,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the simulated evaluation orchestration CLI."""
    parser = argparse.ArgumentParser(description="Run the Week 7 simulated desktop evaluation")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("fake", "qwen"), default="fake")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Plan or execute the condition matrix with resumable owned outputs."""
    parser = build_parser()
    args = parser.parse_args(argv)
    suite_path = cast(Path, args.suite)
    conditions_path = cast(Path, args.conditions)
    output_dir = cast(Path, args.output)
    provider = cast(Provider, args.provider)
    try:
        tasks = load_task_suite(suite_path)
        condition_set = load_condition_set(conditions_path)
        matrix = expand_run_matrix(tasks, condition_set)
    except ValueError as error:
        parser.error(str(error))
    if cast(bool, args.dry_run_plan):
        summary: dict[str, object] = {
            "run_count": len(matrix),
            "condition_count": len(expand_conditions(condition_set)),
            "estimated_minutes": round(
                sum(run.estimated_seconds for run in matrix) / 60,
                2,
            ),
            "executed": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    reports: list[EvaluationReport] = []
    try:
        for condition in expand_conditions(condition_set):
            reports.append(
                _execute_condition(
                    tasks,
                    condition,
                    provider=provider,
                    suite_path=suite_path,
                    conditions_path=conditions_path,
                    output_dir=output_dir,
                    resume=cast(bool, args.resume),
                )
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "run_count": sum(len(report.outcomes) for report in reports),
        "condition_count": len(reports),
        "provider": provider,
        "output": output_dir.as_posix(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "EvaluationConditionSet",
    "EvaluationConditionTemplate",
    "EvaluationRunSpec",
    "ExpandedCondition",
    "build_parser",
    "expand_conditions",
    "expand_run_matrix",
    "load_condition_set",
    "main",
]
