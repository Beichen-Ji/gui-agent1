import pytest

from gui_agent.agent.retry import RetryPolicy
from gui_agent.agent.types import VerificationResult


def failure(
    reason_code: str = "no_visual_change",
    *,
    retryable: bool = True,
) -> VerificationResult:
    return VerificationResult(
        passed=False,
        reason_code=reason_code,  # type: ignore[arg-type]
        summary="The attempt did not produce the expected outcome",
        retryable=retryable,
    )


def test_retry_policy_uses_two_bounded_backoff_delays() -> None:
    policy = RetryPolicy()

    assert policy.decide(failure(), attempt=1).delay_seconds == 0.5
    assert policy.decide(failure(), attempt=2).delay_seconds == 1.0
    exhausted = policy.decide(failure(), attempt=3)
    assert exhausted.retry is False
    assert exhausted.reason_code == "retry_exhausted"


@pytest.mark.parametrize(
    "reason_code",
    [
        "policy_denied",
        "confirmation_rejected",
        "planner_output_invalid",
        "repeated_action",
    ],
)
def test_retry_policy_never_retries_unsafe_or_nonrecoverable_failures(
    reason_code: str,
) -> None:
    decision = RetryPolicy().decide(failure(reason_code), attempt=1)

    assert decision.retry is False
    assert decision.reason_code == reason_code


def test_retry_policy_respects_explicit_nonretryable_failure() -> None:
    decision = RetryPolicy().decide(failure(retryable=False), attempt=1)

    assert decision.retry is False


@pytest.mark.parametrize("limit", [-1, 3])
def test_retry_policy_rejects_limits_outside_the_safe_range(limit: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 2"):
        RetryPolicy(max_retries_per_step=limit)

