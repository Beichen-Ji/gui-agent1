from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from gui_agent.agent.planner import MultimodalPlanner
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    AgentState,
    FinishAction,
    Observation,
    StepResult,
    TaskPlan,
)

RunStatus: TypeAlias = Literal["succeeded", "failed", "stopped"]
FailureStage: TypeAlias = Literal[
    "observation",
    "planning",
    "policy",
    "execution",
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
    failure_stage: FailureStage | None = None


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
    """Run a bounded observe-plan-authorize-execute-feedback loop."""

    def __init__(
        self,
        observer: ObservationSource,
        planner: MultimodalPlanner,
        policy: ActionPolicy,
        executor: PlannedActionExecutor,
    ) -> None:
        self._observer = observer
        self._planner = planner
        self._policy = policy
        self._executor = executor

    def run(self, goal: str, *, max_steps: int = 10) -> AgentRunResult:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must not be blank")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")

        decisions: list[AgentDecision] = []
        results: list[StepResult] = []
        plan: TaskPlan | None = None
        observation: Observation | None = None

        try:
            observation = self._observer.observe(0)
        except Exception as error:
            return self._failure(
                goal,
                "observation",
                error,
                plan,
                observation,
                decisions,
                results,
            )

        try:
            plan = self._planner.create_plan(goal, observation)
        except Exception as error:
            return self._failure(
                goal,
                "planning",
                error,
                plan,
                observation,
                decisions,
                results,
            )

        for step_index in range(max_steps):
            state = AgentState(
                goal=goal,
                plan=plan,
                observation=observation,
                decisions=tuple(decisions),
                results=tuple(results),
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
                    decisions,
                    results,
                )
            decisions.append(next_decision)

            if (
                len(decisions) >= 2
                and decisions[-2].action == next_decision.action
            ):
                return self._result(
                    goal,
                    status="stopped",
                    message="stopped before executing a repeated action",
                    plan=plan,
                    observation=observation,
                    decisions=decisions,
                    results=results,
                )

            try:
                self._policy.authorize(
                    next_decision.action,
                    observation,
                    expected_outcome=next_decision.expected_outcome,
                )
            except Exception:
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
                    plan=plan,
                    observation=observation,
                    decisions=decisions,
                    results=results,
                )

            try:
                step_result = self._executor.execute(
                    next_decision.action,
                    step_index=step_index,
                )
            except Exception as error:
                results.append(
                    StepResult(
                        step_index=step_index,
                        action=next_decision.action,
                        status="failed",
                        message=f"action execution failed ({type(error).__name__})",
                    )
                )
                return self._result(
                    goal,
                    status="failed",
                    message="action execution failed",
                    failure_stage="execution",
                    plan=plan,
                    observation=observation,
                    decisions=decisions,
                    results=results,
                )
            results.append(step_result)
            if step_result.status in {"denied", "failed"}:
                failure_stage: FailureStage = (
                    "policy" if step_result.status == "denied" else "execution"
                )
                return self._result(
                    goal,
                    status="failed",
                    message=step_result.message,
                    failure_stage=failure_stage,
                    plan=plan,
                    observation=observation,
                    decisions=decisions,
                    results=results,
                )

            try:
                observation = self._observer.observe(step_index + 1)
            except Exception as error:
                return self._failure(
                    goal,
                    "observation",
                    error,
                    plan,
                    observation,
                    decisions,
                    results,
                )

            if isinstance(next_decision.action, FinishAction):
                return self._result(
                    goal,
                    status=(
                        "succeeded" if next_decision.action.success else "failed"
                    ),
                    message=next_decision.action.summary,
                    failure_stage=(None if next_decision.action.success else "task"),
                    plan=plan,
                    observation=observation,
                    decisions=decisions,
                    results=results,
                )

        return self._result(
            goal,
            status="stopped",
            message=f"stopped after reaching maximum step count ({max_steps})",
            plan=plan,
            observation=observation,
            decisions=decisions,
            results=results,
        )

    @classmethod
    def _failure(
        cls,
        goal: str,
        stage: FailureStage,
        error: Exception,
        plan: TaskPlan | None,
        observation: Observation | None,
        decisions: Sequence[AgentDecision],
        results: Sequence[StepResult],
    ) -> AgentRunResult:
        return cls._result(
            goal,
            status="failed",
            message=f"{stage} failed ({type(error).__name__})",
            failure_stage=stage,
            plan=plan,
            observation=observation,
            decisions=decisions,
            results=results,
        )

    @staticmethod
    def _result(
        goal: str,
        *,
        status: RunStatus,
        message: str,
        plan: TaskPlan | None,
        observation: Observation | None,
        decisions: Sequence[AgentDecision],
        results: Sequence[StepResult],
        failure_stage: FailureStage | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            goal=goal,
            status=status,
            message=message,
            plan=plan,
            observation=observation,
            decisions=tuple(decisions),
            results=tuple(results),
            failure_stage=failure_stage,
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
