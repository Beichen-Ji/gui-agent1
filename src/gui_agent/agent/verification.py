import hashlib
import re
from collections.abc import Iterable
from typing import Protocol

import numpy as np

from gui_agent.agent.types import (
    AgentDecision,
    FinishAction,
    Observation,
    StepResult,
    VerificationResult,
)

_QUOTED_TEXT = re.compile(r"['\"]([^'\"]+)['\"]")
_ABSENCE_MARKERS = (
    "no longer visible",
    "not visible",
    "is absent",
    "does not display",
)
_DETERMINISTIC_FAILURES = frozenset(
    {"execution_error", "policy_denied", "confirmation_rejected"}
)


class OutcomeVerifier(Protocol):
    def verify(
        self,
        before: Observation,
        decision: AgentDecision,
        execution: StepResult,
        after: Observation,
    ) -> VerificationResult: ...


def _normalized_text(observation: Observation) -> tuple[str, ...]:
    return tuple(
        " ".join(detection.text.split()).casefold()
        for detection in observation.detections
        if detection.text.strip()
    )


def _frame_fingerprint(observation: Observation) -> str:
    image = np.ascontiguousarray(observation.screenshot.image)
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


class RuleBasedOutcomeVerifier:
    def __init__(self, *, success_criteria: str | None = None) -> None:
        self._success_criteria = success_criteria

    def verify(
        self,
        before: Observation,
        decision: AgentDecision,
        execution: StepResult,
        after: Observation,
    ) -> VerificationResult:
        if execution.status == "denied":
            return VerificationResult(
                passed=False,
                reason_code="policy_denied",
                summary="The local safety policy denied the action",
            )
        if execution.status == "failed":
            return VerificationResult(
                passed=False,
                reason_code="execution_error",
                summary="The authorized action failed during execution",
                retryable=True,
            )
        if execution.status == "dry_run":
            return VerificationResult(
                passed=True,
                summary="The action was validated as a dry-run preview",
                evidence=("dry_run_preview",),
            )

        if isinstance(decision.action, FinishAction):
            criteria = self._success_criteria
            expected = (
                tuple(_QUOTED_TEXT.findall(criteria))
                if criteria
                else ()
            )
            observed = " ".join(_normalized_text(after))
            if criteria and not expected:
                return VerificationResult(
                    passed=False,
                    reason_code="expected_text_missing",
                    summary="Success criteria require quoted text for deterministic verification",
                    retryable=False,
                )
            requires_absence = criteria is not None and any(
                marker in criteria.casefold()
                for marker in _ABSENCE_MARKERS
            )
            expected_found = tuple(
                text.casefold() in observed for text in expected
            )
            criteria_satisfied = (
                not any(expected_found)
                if requires_absence
                else all(expected_found)
            )
            if expected and not criteria_satisfied:
                return VerificationResult(
                    passed=False,
                    reason_code="expected_text_missing",
                    summary="Quoted success text does not match the latest observation",
                    retryable=True,
                )
            finish_evidence = (
                "expected_text_absent"
                if expected and requires_absence
                else "expected_text_present"
                if expected
                else "finish_signal"
            )
            return VerificationResult(
                passed=True,
                summary="The finish signal is consistent with the latest observation",
                evidence=(finish_evidence,),
            )

        before_screen = before.screenshot
        after_screen = after.screenshot
        geometry_changed = (
            before_screen.width != after_screen.width
            or before_screen.height != after_screen.height
            or before_screen.origin != after_screen.origin
        )
        frame_changed = geometry_changed or (
            _frame_fingerprint(before) != _frame_fingerprint(after)
        )
        ocr_changed = set(_normalized_text(before)) != set(_normalized_text(after))
        evidence: list[str] = []
        if frame_changed:
            evidence.append("frame_changed")
        if ocr_changed:
            evidence.append("ocr_changed")
        if evidence:
            return VerificationResult(
                passed=True,
                summary="The latest observation contains deterministic change evidence",
                evidence=tuple(evidence),
            )
        return VerificationResult(
            passed=False,
            reason_code="no_visual_change",
            summary="The screenshot and OCR text did not change after the action",
            retryable=True,
        )


class CompositeOutcomeVerifier:
    def __init__(self, verifiers: Iterable[OutcomeVerifier]) -> None:
        self._verifiers = tuple(verifiers)
        if not self._verifiers:
            raise ValueError("at least one outcome verifier is required")

    def verify(
        self,
        before: Observation,
        decision: AgentDecision,
        execution: StepResult,
        after: Observation,
    ) -> VerificationResult:
        results = tuple(
            verifier.verify(before, decision, execution, after)
            for verifier in self._verifiers
        )
        for result in results:
            if result.reason_code in _DETERMINISTIC_FAILURES:
                return result
        passed = tuple(result for result in results if result.passed)
        if passed:
            return VerificationResult(
                passed=True,
                summary="; ".join(result.summary for result in passed)[:500],
                evidence=tuple(
                    evidence
                    for result in passed
                    for evidence in result.evidence
                )[:20],
            )
        return results[0]


__all__ = [
    "CompositeOutcomeVerifier",
    "OutcomeVerifier",
    "RuleBasedOutcomeVerifier",
]
