import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from gui_agent.agent.planner import MultimodalPlanner
from gui_agent.agent.policy import ActionDeniedError
from gui_agent.agent.retry import RetryPolicy
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    AgentState,
    FailureReason,
    FinishAction,
    Observation,
    PlanProgress,
    ReplanContext,
    StepResult,
    TaskPlan,
    VerificationResult,
    reconcile_revised_plan,
)
from gui_agent.agent.verification import (
    CompositeOutcomeVerifier,
    OutcomeVerifier,
    RuleBasedOutcomeVerifier,
)

RunStatus: TypeAlias = Literal["succeeded", "failed", "stopped"]
FailureStage: TypeAlias = Literal[
    "observation",
    "planning",
    "policy",
    "execution",
    "verification",
    "task",
]


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    goal: str
    status: RunStatus
    message: str
    plan: TaskPlan | None
    observation: Observation | None
    decisions: tuple[AgentDecision, ...]
    results: tuple[StepResult, ...]
    progress: PlanProgress | None = None
    failure_stage: FailureStage | None = None
    reason_code: FailureReason | None = None
    verifications: tuple[VerificationResult, ...] = ()


class ObservationSource(Protocol):
    def observe(self, step_index: int) -> Observation: ...


class ActionPolicy(Protocol):
    def authorize(
        self,
        action: AgentAction,
        observation: Observation,
        *,
        expected_outcome: str,
    ) -> None: ...


class PlannedActionExecutor(Protocol):
    def execute(self, action: AgentAction, *, step_index: int) -> StepResult: ...


class GUIAgent:
    """Run a bounded observe-plan-authorize-execute-verify-recover loop."""

    def __init__(
        self,
        observer: ObservationSource,
        planner: MultimodalPlanner,
        policy: ActionPolicy,
        executor: PlannedActionExecutor,
        *,
        verifier: OutcomeVerifier | None = None,
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[float], None] = time.sleep,
        max_replans: int = 1,
    ) -> None:
        if isinstance(max_replans, bool) or max_replans not in {0, 1}:
            raise ValueError("max_replans must be 0 or 1")
        self._observer = observer
        self._planner = planner
        self._policy = policy
        self._executor = executor
        self._verifier = verifier
        self._retry_policy = retry_policy or RetryPolicy()
        self._clock = clock
        self._max_replans = max_replans

    def run(
        self,
        goal: str,
        *,
        success_criteria: str | None = None,
        max_steps: int = 10,
    ) -> AgentRunResult:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must not be blank")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if success_criteria is not None and not success_criteria.strip():
            raise ValueError("success_criteria must not be blank")

        decisions: list[AgentDecision] = []
        results: list[StepResult] = []
        verifications: list[VerificationResult] = []
        plan: TaskPlan | None = None
        observation: Observation | None = None
        progress: PlanProgress | None = None
        replan_context: ReplanContext | None = None
        has_verification_evidence = False
        verifier = self._verifier or CompositeOutcomeVerifier(
            (RuleBasedOutcomeVerifier(success_criteria=success_criteria),)
        )

        observation, observation_failure = self._observe_with_retry(0)
        if observation_failure is not None:
            return self._result(
                goal,
                status="failed",
                message=observation_failure.summary,
                failure_stage="observation",
                reason_code=observation_failure.reason_code,
                plan=None,
                observation=None,
                progress=None,
                decisions=decisions,
                results=results,
                verifications=verifications,
            )
        assert observation is not None

        try:
            plan = self._planner.create_plan(goal, observation)
            progress = PlanProgress.from_plan(plan)
        except Exception as error:
            return self._failure(
                goal,
                "planning",
                error,
                plan,
                observation,
                progress,
                decisions,
                results,
                reason_code="planner_output_invalid",
                verifications=verifications,
            )

        step_index = 0

        def recover(failure: VerificationResult) -> AgentRunResult | None:
            nonlocal plan, progress, replan_context
            assert plan is not None
            assert progress is not None
            assert observation is not None
            assert failure.reason_code is not None
            active = next(
                step
                for step in progress.steps
                if step.step_id == progress.active_step_id
            )
            retry = self._retry_policy.decide(failure, attempt=active.attempts)
            replan_context = ReplanContext(
                reason_code=retry.reason_code,
                summary=failure.summary,
            )
            if retry.retry:
                self._clock(retry.delay_seconds)
                return None
            if failure.retryable and progress.replan_count < self._max_replans:
                revision_state = AgentState(
                    goal=goal,
                    plan=plan,
                    progress=progress,
                    observation=observation,
                    decisions=tuple(decisions),
                    results=tuple(results),
                    replan_context=replan_context,
                )
                try:
                    proposed = self._planner.revise_plan(
                        revision_state,
                        replan_context,
                    )
                    plan, progress = reconcile_revised_plan(
                        plan,
                        progress,
                        proposed,
                    )
                    return None
                except Exception as error:
                    return self._failure(
                        goal,
                        "planning",
                        error,
                        plan,
                        observation,
                        progress.fail_active(),
                        decisions,
                        results,
                        reason_code="planner_output_invalid",
                        verifications=verifications,
                    )
            progress = progress.fail_active()
            return self._result(
                goal,
                status="failed",
                message=failure.summary,
                failure_stage=self._stage_for_reason(failure.reason_code),
                reason_code=retry.reason_code,
                plan=plan,
                observation=observation,
                progress=progress,
                decisions=decisions,
                results=results,
                verifications=verifications,
            )

        while step_index < max_steps:
            state = AgentState(
                goal=goal,
                plan=plan,
                progress=progress,
                observation=observation,
                decisions=tuple(decisions),
                results=tuple(results),
                replan_context=replan_context,
            )
            try:
                next_decision = self._planner.next_action(state)
            except Exception as error:
                return self._failure(
                    goal,
                    "planning",
                    error,
                    plan,
                    observation,
                    progress.fail_active(),
                    decisions,
                    results,
                    reason_code="planner_output_invalid",
                    verifications=verifications,
                )
            decisions.append(next_decision)
            try:
                progress = progress.select_step(
                    next_decision.current_step_id
                ).record_attempt(next_decision.current_step_id)
            except ValueError as error:
                return self._failure(
                    goal,
                    "planning",
                    error,
                    plan,
                    observation,
                    progress.fail_active(),
                    decisions,
                    results,
                    reason_code="planner_output_invalid",
                    verifications=verifications,
                )

            if len(decisions) >= 2 and decisions[-2].action == next_decision.action:
                progress = progress.fail_active()
                return self._result(
                    goal,
                    status="stopped",
                    message="stopped before executing a repeated action",
                    failure_stage="verification",
                    reason_code="repeated_action",
                    plan=plan,
                    observation=observation,
                    progress=progress,
                    decisions=decisions,
                    results=results,
                    verifications=verifications,
                )

            try:
                self._policy.authorize(
                    next_decision.action,
                    observation,
                    expected_outcome=next_decision.expected_outcome,
                )
            except Exception as error:
                progress = progress.fail_active()
                reason_code: FailureReason = (
                    "confirmation_rejected"
                    if isinstance(error, ActionDeniedError)
                    and (
                        "not confirmed" in str(error)
                        or "confirmation could not" in str(error)
                    )
                    else "policy_denied"
                )
                results.append(
                    StepResult(
                        step_index=step_index,
                        action=next_decision.action,
                        status="denied",
                        message="action denied by the local safety policy",
                    )
                )
                return self._result(
                    goal,
                    status="failed",
                    message="safety policy denied the proposed action",
                    failure_stage="policy",
                    reason_code=reason_code,
                    plan=plan,
                    observation=observation,
                    progress=progress,
                    decisions=decisions,
                    results=results,
                    verifications=verifications,
                )

            try:
                step_result = self._executor.execute(
                    next_decision.action,
                    step_index=step_index,
                )
            except Exception as error:
                step_result = StepResult(
                    step_index=step_index,
                    action=next_decision.action,
                    status="failed",
                    message=f"action execution failed ({type(error).__name__})",
                )
            results.append(step_result)

            before = observation
            step_index += 1
            after, observation_failure = self._observe_with_retry(step_index)
            if observation_failure is not None:
                verifications.append(observation_failure)
                recovered = recover(observation_failure)
                if recovered is not None:
                    return recovered
                continue
            assert after is not None
            observation = after
            verification = verifier.verify(before, next_decision, step_result, after)
            verifications.append(verification)
            if not verification.passed:
                recovered = recover(verification)
                if recovered is not None:
                    return recovered
                continue

            replan_context = None
            if not isinstance(next_decision.action, FinishAction):
                has_verification_evidence = (
                    has_verification_evidence or bool(verification.evidence)
                )
                continue

            if not next_decision.action.success:
                progress = progress.fail_active()
                return self._result(
                    goal,
                    status="failed",
                    message=next_decision.action.summary,
                    failure_stage="task",
                    reason_code="expected_text_missing",
                    plan=plan,
                    observation=observation,
                    progress=progress,
                    decisions=decisions,
                    results=results,
                    verifications=verifications,
                )

            has_expected_text = "expected_text_present" in verification.evidence
            active_is_last = all(
                step.status == "completed" or step.step_id == progress.active_step_id
                for step in progress.steps
            )
            if not active_is_last or not (
                has_verification_evidence or has_expected_text
            ):
                progress = progress.fail_active()
                return self._result(
                    goal,
                    status="failed",
                    message="finish requested without verified completion evidence",
                    failure_stage="verification",
                    reason_code="expected_text_missing",
                    plan=plan,
                    observation=observation,
                    progress=progress,
                    decisions=decisions,
                    results=results,
                    verifications=verifications,
                )
            progress = progress.complete_active()
            return self._result(
                goal,
                status="succeeded",
                message=next_decision.action.summary,
                plan=plan,
                observation=observation,
                progress=progress,
                decisions=decisions,
                results=results,
                verifications=verifications,
            )

        return self._result(
            goal,
            status="stopped",
            message=f"stopped after reaching maximum step count ({max_steps})",
            plan=plan,
            observation=observation,
            progress=progress,
            decisions=decisions,
            results=results,
            verifications=verifications,
        )

    def _observe_with_retry(
        self,
        step_index: int,
    ) -> tuple[Observation | None, VerificationResult | None]:
        attempt = 1
        while True:
            try:
                return self._observer.observe(step_index), None
            except Exception as error:
                failure = VerificationResult(
                    passed=False,
                    reason_code="observation_error",
                    summary=f"observation failed ({type(error).__name__})",
                    retryable=True,
                )
                retry = self._retry_policy.decide(failure, attempt=attempt)
                if not retry.retry:
                    return None, failure.model_copy(update={"retryable": False})
                self._clock(retry.delay_seconds)
                attempt += 1

    @classmethod
    def _failure(
        cls,
        goal: str,
        stage: FailureStage,
        error: Exception,
        plan: TaskPlan | None,
        observation: Observation | None,
        progress: PlanProgress | None,
        decisions: Sequence[AgentDecision],
        results: Sequence[StepResult],
        *,
        reason_code: FailureReason,
        verifications: Sequence[VerificationResult],
    ) -> AgentRunResult:
        return cls._result(
            goal,
            status="failed",
            message=f"{stage} failed ({type(error).__name__})",
            failure_stage=stage,
            reason_code=reason_code,
            plan=plan,
            observation=observation,
            progress=progress,
            decisions=decisions,
            results=results,
            verifications=verifications,
        )

    @staticmethod
    def _stage_for_reason(reason_code: FailureReason) -> FailureStage:
        if reason_code == "execution_error":
            return "execution"
        if reason_code == "observation_error":
            return "observation"
        if reason_code == "planner_output_invalid":
            return "planning"
        if reason_code in {"policy_denied", "confirmation_rejected"}:
            return "policy"
        return "verification"

    @staticmethod
    def _result(
        goal: str,
        *,
        status: RunStatus,
        message: str,
        plan: TaskPlan | None,
        observation: Observation | None,
        progress: PlanProgress | None,
        decisions: Sequence[AgentDecision],
        results: Sequence[StepResult],
        verifications: Sequence[VerificationResult],
        failure_stage: FailureStage | None = None,
        reason_code: FailureReason | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            goal=goal,
            status=status,
            message=message,
            plan=plan,
            observation=observation,
            progress=progress,
            decisions=tuple(decisions),
            results=tuple(results),
            failure_stage=failure_stage,
            reason_code=reason_code,
            verifications=tuple(verifications),
        )


__all__ = [
    "ActionPolicy",
    "AgentRunResult",
    "FailureStage",
    "GUIAgent",
    "ObservationSource",
    "PlannedActionExecutor",
    "RunStatus",
]
