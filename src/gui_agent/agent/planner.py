from collections import deque
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from gui_agent.agent.types import AgentDecision, AgentState, Observation, TaskPlan


class PlannerError(RuntimeError):
    """A planner could not produce a validated plan or decision."""


@runtime_checkable
class MultimodalPlanner(Protocol):
    def create_plan(self, goal: str, observation: Observation) -> TaskPlan: ...

    def next_action(self, state: AgentState) -> AgentDecision: ...


class FakePlanner:
    def __init__(
        self,
        *,
        plan: TaskPlan,
        decisions: Iterable[AgentDecision],
    ) -> None:
        self._plan = plan
        self._decisions = deque(decisions)

    def create_plan(self, goal: str, observation: Observation) -> TaskPlan:
        if not goal.strip():
            raise PlannerError("goal must not be blank")
        del observation
        return self._plan

    def next_action(self, state: AgentState) -> AgentDecision:
        del state
        if not self._decisions:
            raise PlannerError("fake planner has no configured decision remaining")
        return self._decisions.popleft()


__all__ = ["FakePlanner", "MultimodalPlanner", "PlannerError"]
