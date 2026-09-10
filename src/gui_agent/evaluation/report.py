"""Build, validate, load, and atomically write Week 7 evaluation reports."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from gui_agent._models import StrictFrozenModel
from gui_agent.evaluation.metrics import SuiteMetrics, TaskOutcome, calculate_suite_metrics
from gui_agent.provenance import atomic_write_owned_json
from gui_agent.simulation.harness import ObservationMode


class EvaluationContext(StrictFrozenModel):
    """Capture all hashes, settings, and environment needed for resume safety."""

    suite_sha256: str
    conditions_sha256: str
    git_revision: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=500)
    adapter_label: str | None = Field(default=None, max_length=500)
    adapter_manifest_sha256: str | None = None
    adapter_weight_sha256: str | None = None
    prompt_profile: str = Field(min_length=1, max_length=80)
    ocr_profile: str = Field(min_length=1, max_length=80)
    observation_mode: ObservationMode
    seed: int = Field(ge=0)
    environment: dict[str, str]

    @field_validator(
        "suite_sha256",
        "conditions_sha256",
        "adapter_manifest_sha256",
        "adapter_weight_sha256",
    )
    @classmethod
    def _valid_sha256(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("provenance hashes must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def _complete_adapter_provenance(self) -> "EvaluationContext":
        values = (
            self.adapter_label,
            self.adapter_manifest_sha256,
            self.adapter_weight_sha256,
        )
        if any(value is not None for value in values) and any(
            value is None for value in values
        ):
            raise ValueError("adapter provenance must be complete")
        return self


class EvaluationReport(StrictFrozenModel):
    """Store reproducible provenance, metrics, and ordered task outcomes."""

    kind: Literal["gui-agent-week7-evaluation"] = "gui-agent-week7-evaluation"
    schema_version: Literal[1] = 1
    suite_sha256: str
    conditions_sha256: str
    git_revision: str
    model: str
    adapter_label: str | None
    adapter_manifest_sha256: str | None
    adapter_weight_sha256: str | None
    prompt_profile: str
    ocr_profile: str
    observation_mode: ObservationMode
    seed: int
    environment: dict[str, str]
    metrics: SuiteMetrics
    outcomes: tuple[TaskOutcome, ...]

    def matches_context(self, context: EvaluationContext) -> bool:
        """Return whether every provenance field matches a requested context."""
        return all(
            getattr(self, field_name) == value
            for field_name, value in context.model_dump(mode="python").items()
        )


def build_evaluation_report(
    context: EvaluationContext,
    outcomes: tuple[TaskOutcome, ...],
) -> EvaluationReport:
    """Sort outcomes and calculate a report for one immutable context."""
    ordered = tuple(
        sorted(
            outcomes,
            key=lambda outcome: (
                outcome.task_id,
                outcome.resolution[0],
                outcome.resolution[1],
            ),
        )
    )
    return EvaluationReport(
        **context.model_dump(mode="python"),
        metrics=calculate_suite_metrics(ordered),
        outcomes=ordered,
    )


def load_evaluation_report(path: Path) -> EvaluationReport:
    """Load and strictly validate a Week 7 evaluation report."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"could not read evaluation report: {path}") from error
    try:
        return EvaluationReport.model_validate_json(raw)
    except ValueError as error:
        raise ValueError(f"invalid evaluation report: {path}") from error


def write_evaluation_report(
    report: EvaluationReport,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write only a caller-owned Week 7 report file."""
    atomic_write_owned_json(
        report.model_dump(mode="json"),
        path,
        owned_kind=report.kind,
        overwrite=overwrite,
    )


__all__ = [
    "EvaluationContext",
    "EvaluationReport",
    "build_evaluation_report",
    "load_evaluation_report",
    "write_evaluation_report",
]
