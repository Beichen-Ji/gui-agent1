"""Validated fake, local, and remote multimodal planner adapters."""

import base64
import json
from collections import deque
from collections.abc import Iterable
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import cv2
from pydantic import BaseModel, SecretStr

from gui_agent.agent.prompts import (
    build_action_prompt,
    build_plan_prompt,
    build_replan_prompt,
)
from gui_agent.agent.types import (
    AgentDecision,
    AgentState,
    Observation,
    ReplanContext,
    TaskPlan,
)


class PlannerError(RuntimeError):
    """A planner could not produce a validated plan or decision."""


class RemoteImagePermissionError(PlannerError):
    """A remote planner was asked to receive a screenshot without consent."""


@runtime_checkable
class MultimodalPlanner(Protocol):
    """Create, advance, and revise structured plans from visual state."""

    def create_plan(self, goal: str, observation: Observation) -> TaskPlan:
        """Create an initial structured plan for a goal and observation."""
        ...

    def next_action(self, state: AgentState) -> AgentDecision:
        """Choose one validated action for the active plan step."""
        ...

    def revise_plan(self, state: AgentState, failure: ReplanContext) -> TaskPlan:
        """Revise unfinished steps using redacted failure context."""
        ...


class FakePlanner:
    """Return deterministic plans and decisions for tests and offline demos."""

    def __init__(
        self,
        *,
        plan: TaskPlan,
        decisions: Iterable[AgentDecision],
        revised_plans: Iterable[TaskPlan] = (),
    ) -> None:
        """Configure finite queues of deterministic planner outputs."""
        self._plan = plan
        self._decisions = deque(decisions)
        self._revised_plans = deque(revised_plans)

    def create_plan(self, goal: str, observation: Observation) -> TaskPlan:
        """Return the configured initial plan for a non-blank goal."""
        if not goal.strip():
            raise PlannerError("goal must not be blank")
        del observation
        return self._plan

    def next_action(self, state: AgentState) -> AgentDecision:
        """Consume the next configured decision."""
        del state
        if not self._decisions:
            raise PlannerError("fake planner has no configured decision remaining")
        return self._decisions.popleft()

    def revise_plan(self, state: AgentState, failure: ReplanContext) -> TaskPlan:
        """Consume the next configured revised plan."""
        del state, failure
        if not self._revised_plans:
            raise PlannerError("fake planner has no configured revised plan remaining")
        return self._revised_plans.popleft()


ModelSchema = TypeVar("ModelSchema", bound=BaseModel)


def _png_data_url(observation: Observation) -> str:
    encoded_ok, encoded = cv2.imencode(".png", observation.screenshot.image)
    if not encoded_ok:
        raise PlannerError("could not encode screenshot as PNG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


class LangChainPlanner:
    """Call a remote structured-output model only with explicit image consent."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        allow_remote_image: bool = False,
        chat_model: object | None = None,
    ) -> None:
        """Configure a remote chat model and its screenshot permission boundary."""
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if chat_model is None:
            from langchain_openai import ChatOpenAI

            chat_model = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=SecretStr(api_key) if api_key is not None else None,
                temperature=0,
                max_retries=1,
            )
        self._chat_model = chat_model
        self._allow_remote_image = allow_remote_image

    def _invoke(
        self,
        schema: type[ModelSchema],
        prompt: str,
        observation: Observation,
    ) -> ModelSchema:
        if not self._allow_remote_image:
            raise RemoteImagePermissionError(
                "remote screenshots require allow_remote_image=True"
            )
        messages = [
            {
                "role": "system",
                "content": "Return only the requested validated GUI-agent structure.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(observation)},
                    },
                ],
            },
        ]
        try:
            model = cast(Any, self._chat_model).with_structured_output(
                schema,
                method="json_schema",
                include_raw=False,
            )
            raw = model.invoke(messages)
            if isinstance(raw, schema):
                return raw
            return schema.model_validate_json(json.dumps(raw))
        except Exception as exc:
            raise PlannerError(f"remote model failed to produce {schema.__name__}") from exc

    def create_plan(self, goal: str, observation: Observation) -> TaskPlan:
        """Request and validate an initial task plan."""
        return self._invoke(TaskPlan, build_plan_prompt(goal, observation), observation)

    def next_action(self, state: AgentState) -> AgentDecision:
        """Request and validate one action for the current state."""
        return self._invoke(
            AgentDecision,
            build_action_prompt(state),
            state.observation,
        )

    def revise_plan(self, state: AgentState, failure: ReplanContext) -> TaskPlan:
        """Request and validate a revision of unfinished plan steps."""
        return self._invoke(
            TaskPlan,
            build_replan_prompt(state, failure),
            state.observation,
        )


__all__ = [
    "FakePlanner",
    "LangChainPlanner",
    "MultimodalPlanner",
    "PlannerError",
    "RemoteImagePermissionError",
]
