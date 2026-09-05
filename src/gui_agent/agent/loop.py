import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, TypeAlias

from gui_agent.agent.events import (
    EventEmitter,
    EventKind,
    EventSink,
    NullEventSink,
    action_metadata,
    goal_metadata,
    observation_metadata,
)
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
        event_sink: EventSink | None = None,
        run_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        event_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
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
        self._event_sink = event_sink or NullEventSink()
        self._run_id_factory = run_id_factory
        self._event_clock = event_clock
        self._emitter: EventEmitter | None = None

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

        self._emitter = EventEmitter(
            self._event_sink,
            run_id=self._run_id_factory(),
            clock=self._event_clock,
        )
        self._emit(
            "run_started",
            {**goal_metadata(goal), "max_steps": max_steps},
        )

        decisions: list[AgentDecision] = []
        results: list[StepResult] = []
        verifications: list[VerificationResult] = []
        plan: TaskPlan | None = None
        observation: Observation | None = None
        progress: PlanProgress | None = None
        replan_context: ReplanContext | None = None
        has_verification_evidence = False
        verified_step_id: str | None = None
        deterministic_verifier = RuleBasedOutcomeVerifier(
            success_criteria=success_criteria
        )
        verifier = CompositeOutcomeVerifier(
            (deterministic_verifier, self._verifier)
            if self._verifier is not None
            else (deterministic_verifier,)
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
            self._emit("plan_created", {"step_count": len(plan.steps)})
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
            nonlocal plan, progress, replan_context, verified_step_id
            assert plan is not None
            assert progress is not None
            assert observation is not None
            assert failure.reason_code is not None
            verified_step_id = None
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
                self._emit(
                    "retry_scheduled",
                    {
                        "reason_code": retry.reason_code,
                        "delay_seconds": retry.delay_seconds,
                        "attempt": active.attempts,
                    },
                )
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
                    self._emit(
                        "plan_revised",
                        {
                            "reason_code": replan_context.reason_code,
                            "step_count": len(plan.steps),
                            "replan_count": progress.replan_count,
                        },
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
            self._emit(
                "step_started",
                {
                    "step_index": step_index,
                    "active_step_id": progress.active_step_id,
                },
            )
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
            self._emit(
                "action_proposed",
                {
                    "current_step_id": next_decision.current_step_id,
                    **action_metadata(next_decision.action),
                },
            )
            try:
                previous_active_step_id = progress.active_step_id
                progress = progress.select_step(
                    next_decision.current_step_id,
                    verified_step_id=verified_step_id,
                ).record_attempt(next_decision.current_step_id)
                if next_decision.current_step_id != previous_active_step_id:
                    verified_step_id = None
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
                self._emit(
                    "action_authorized",
                    action_metadata(next_decision.action),
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
            self._emit(
                "action_executed",
                {
                    **action_metadata(next_decision.action),
                    "status": step_result.status,
                    "step_index": step_result.step_index,
                },
            )

            before = observation
            step_index += 1
            after, observation_failure = self._observe_with_retry(step_index)
            if observation_failure is not None:
                verifications.append(observation_failure)
                self._emit(
                    "verification_completed",
                    {
                        "passed": False,
                        "reason_code": observation_failure.reason_code,
                    },
                )
                recovered = recover(observation_failure)
                if recovered is not None:
                    return recovered
                continue
            assert after is not None
            observation = after
            try:
                verification = verifier.verify(before, next_decision, step_result, after)
            except Exception:
                verification = VerificationResult(
                    passed=False,
                    reason_code="expected_text_missing",
                    summary="outcome verification failed",
                    retryable=False,
                )
            verifications.append(verification)
            self._emit(
                "verification_completed",
                {
                    "passed": verification.passed,
                    "reason_code": verification.reason_code,
                    "evidence_count": len(verification.evidence),
                },
            )
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
                verified_step_id = progress.active_step_id
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

            has_expected_text = bool(
                {"expected_text_present", "expected_text_absent"}
                & set(verification.evidence)
            )
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
                observed = self._observer.observe(step_index)
                self._emit("observation_completed", observation_metadata(observed))
                return observed, None
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
                self._emit(
                    "retry_scheduled",
                    {
                        "reason_code": retry.reason_code,
                        "delay_seconds": retry.delay_seconds,
                        "attempt": attempt,
                    },
                )
                self._clock(retry.delay_seconds)
                attempt += 1

    def _failure(
        self,
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
        return self._result(
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

    def _result(
        self,
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
        result = AgentRunResult(
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
        self._emit(
            "run_finished",
            {
                "status": status,
                "failure_stage": failure_stage,
                "reason_code": reason_code,
                "decision_count": len(decisions),
                "result_count": len(results),
            },
        )
        return result

    def _emit(self, kind: EventKind, payload: dict[str, object]) -> None:
        if self._emitter is None:
            return
        try:
            self._emitter.emit(kind, payload)
        except Exception:
            self._emitter = None


__all__ = [
    "ActionPolicy",
    "AgentRunResult",
    "FailureStage",
    "GUIAgent",
    "ObservationSource",
    "PlannedActionExecutor",
    "RunStatus",
]
