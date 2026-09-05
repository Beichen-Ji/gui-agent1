from gui_agent.agent.types import RetryDecision, VerificationResult

_NEVER_RETRY = frozenset(
    {
        "policy_denied",
        "confirmation_rejected",
        "planner_output_invalid",
        "repeated_action",
        "retry_exhausted",
    }
)


class RetryPolicy:
    """Turn typed failures into bounded, deterministic retry decisions."""

    def __init__(
        self,
        *,
        max_retries_per_step: int = 2,
        backoff_seconds: tuple[float, ...] = (0.5, 1.0),
    ) -> None:
        if (
            isinstance(max_retries_per_step, bool)
            or not isinstance(max_retries_per_step, int)
            or not 0 <= max_retries_per_step <= 2
        ):
            raise ValueError("max_retries_per_step must be between 0 and 2")
        if len(backoff_seconds) < max_retries_per_step or any(
            isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not 0.0 <= delay <= 5.0
            for delay in backoff_seconds
        ):
            raise ValueError("backoff_seconds must provide safe delays for every retry")
        self.max_retries_per_step = max_retries_per_step
        self.backoff_seconds = tuple(float(delay) for delay in backoff_seconds)

    def decide(self, failure: VerificationResult, *, attempt: int) -> RetryDecision:
        if failure.passed or failure.reason_code is None:
            raise ValueError("retry decisions require a failed verification")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if not failure.retryable or failure.reason_code in _NEVER_RETRY:
            return RetryDecision(retry=False, reason_code=failure.reason_code)
        if attempt <= self.max_retries_per_step:
            return RetryDecision(
                retry=True,
                delay_seconds=self.backoff_seconds[attempt - 1],
                reason_code=failure.reason_code,
            )
        return RetryDecision(retry=False, reason_code="retry_exhausted")


__all__ = ["RetryPolicy"]
