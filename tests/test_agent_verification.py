from datetime import UTC, datetime

import numpy as np

from gui_agent.agent.types import (
    AgentDecision,
    ClickAction,
    FinishAction,
    Observation,
    StepResult,
    VerificationResult,
)
from gui_agent.agent.verification import (
    CompositeOutcomeVerifier,
    RuleBasedOutcomeVerifier,
)
from gui_agent.types import BoundingBox, OCRDetection, Point, ScreenshotResult


def observation(value: int, *texts: str) -> Observation:
    image = np.full((20, 30, 3), value, dtype=np.uint8)
    detections = tuple(
        OCRDetection(text, 0.99, BoundingBox(1, 1 + index * 3, 20, 3 + index * 3))
        for index, text in enumerate(texts)
    )
    return Observation(
        screenshot=ScreenshotResult(
            image=image,
            monitor_index=1,
            captured_at=datetime(2026, 9, 5, tzinfo=UTC),
            origin=Point(0, 0),
        ),
        detections=detections,
        step_index=value,
    )


def decision(action: ClickAction | FinishAction) -> AgentDecision:
    return AgentDecision(
        current_step_id="step-1",
        rationale_summary="Use the visible control",
        action=action,
        expected_outcome="The target becomes visible",
    )


def result(
    action: ClickAction | FinishAction,
    *,
    status: str = "executed",
) -> StepResult:
    return StepResult(
        step_index=0,
        action=action,
        status=status,  # type: ignore[arg-type]
        message=f"action {status}",
    )


def test_rule_verifier_accepts_visual_or_ocr_change() -> None:
    action = ClickAction(x=10, y=10)
    verifier = RuleBasedOutcomeVerifier()

    visual = verifier.verify(
        observation(0, "Before"),
        decision(action),
        result(action),
        observation(1, "Before"),
    )
    ocr = verifier.verify(
        observation(0, "Before"),
        decision(action),
        result(action),
        observation(0, "After"),
    )

    assert visual.passed is True
    assert "frame_changed" in visual.evidence
    assert ocr.passed is True
    assert "ocr_changed" in ocr.evidence


def test_rule_verifier_detects_unchanged_screen_as_retryable() -> None:
    action = ClickAction(x=10, y=10)

    verified = RuleBasedOutcomeVerifier().verify(
        observation(0, "Same"),
        decision(action),
        result(action),
        observation(0, "Same"),
    )

    assert verified.passed is False
    assert verified.reason_code == "no_visual_change"
    assert verified.retryable is True


def test_rule_verifier_treats_dry_run_as_safe_preview_evidence() -> None:
    action = ClickAction(x=10, y=10)

    verified = RuleBasedOutcomeVerifier().verify(
        observation(0),
        decision(action),
        result(action, status="dry_run"),
        observation(0),
    )

    assert verified.passed is True
    assert verified.evidence == ("dry_run_preview",)


def test_finish_checks_quoted_success_text_when_present() -> None:
    action = FinishAction(success=True, summary="Done")
    verifier = RuleBasedOutcomeVerifier(
        success_criteria="The panel displays 'READY FOR WEEK 6'."
    )

    missing = verifier.verify(
        observation(0),
        decision(action),
        result(action),
        observation(0, "Still loading"),
    )
    found = verifier.verify(
        observation(0),
        decision(action),
        result(action),
        observation(0, "ready for week 6"),
    )

    assert missing.reason_code == "expected_text_missing"
    assert missing.retryable is True
    assert found.passed is True
    assert "expected_text_present" in found.evidence


def test_finish_rejects_unquoted_success_criteria_as_unverifiable() -> None:
    action = FinishAction(success=True, summary="Done")
    verifier = RuleBasedOutcomeVerifier(
        success_criteria="The requested area is visible and ready."
    )

    verified = verifier.verify(
        observation(0),
        decision(action),
        result(action),
        observation(1, "The requested area is visible and ready"),
    )

    assert verified.passed is False
    assert verified.reason_code == "expected_text_missing"
    assert verified.retryable is False


def test_finish_supports_quoted_text_that_must_no_longer_be_visible() -> None:
    action = FinishAction(success=True, summary="Closed")
    verifier = RuleBasedOutcomeVerifier(
        success_criteria=(
            "The text 'Local GUI Agent Testbed' is no longer visible."
        )
    )

    still_visible = verifier.verify(
        observation(0),
        decision(action),
        result(action),
        observation(1, "Local GUI Agent Testbed"),
    )
    absent = verifier.verify(
        observation(0),
        decision(action),
        result(action),
        observation(1, "Unrelated desktop"),
    )

    assert still_visible.reason_code == "expected_text_missing"
    assert absent.passed is True
    assert "expected_text_absent" in absent.evidence


class PassingSemanticVerifier:
    def verify(
        self,
        before: Observation,
        proposed: AgentDecision,
        execution: StepResult,
        after: Observation,
    ) -> VerificationResult:
        del before, proposed, execution, after
        return VerificationResult(
            passed=True,
            summary="A model guessed success",
            evidence=("semantic_guess",),
        )


def test_composite_does_not_allow_semantic_success_to_override_execution_failure() -> None:
    action = ClickAction(x=10, y=10)
    failed = result(action, status="failed")
    verifier = CompositeOutcomeVerifier(
        (RuleBasedOutcomeVerifier(), PassingSemanticVerifier())
    )

    verified = verifier.verify(
        observation(0),
        decision(action),
        failed,
        observation(1),
    )

    assert verified.passed is False
    assert verified.reason_code == "execution_error"
