import json
from collections.abc import Callable
from typing import Any, TypeVar, cast

from PIL import Image
from pydantic import BaseModel

from gui_agent.agent.planner import PlannerError
from gui_agent.agent.prompts import build_action_prompt, build_plan_prompt
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    AgentState,
    ClickAction,
    DragAction,
    Observation,
    ScrollAction,
    TaskPlan,
)

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_COORDINATE_GRID_SIZE = 1000
ModelSchema = TypeVar("ModelSchema", bound=BaseModel)

_QWEN_COORDINATE_INSTRUCTION = (
    "For click, scroll, and drag actions, return image-relative integer coordinates "
    "on a 1000x1000 grid. (0,0) is the image's top-left pixel and (999,999) "
    "is its bottom-right pixel. Do not add the desktop origin; the application "
    "converts these coordinates to absolute desktop pixels."
)


def _default_processor_loader(model_name: str, **kwargs: object) -> object:
    from transformers import AutoProcessor

    loader = cast(Callable[..., object], AutoProcessor.from_pretrained)
    return loader(model_name, **kwargs)


def _default_model_loader(model_name: str, **kwargs: object) -> object:
    from transformers import AutoModelForMultimodalLM

    return AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)


def _json_object(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model output does not contain a JSON object")
    return stripped[start : end + 1]


def _desktop_coordinate(value: int, *, origin: int, extent: int) -> int:
    if not 0 <= value < QWEN_COORDINATE_GRID_SIZE:
        raise ValueError("Qwen pointer coordinates must be between 0 and 999")
    return origin + (value * extent) // QWEN_COORDINATE_GRID_SIZE


def _desktop_action(action: AgentAction, observation: Observation) -> AgentAction:
    screenshot = observation.screenshot
    origin = screenshot.origin

    if isinstance(action, ClickAction):
        return action.model_copy(
            update={
                "x": _desktop_coordinate(
                    action.x,
                    origin=origin.x,
                    extent=screenshot.width,
                ),
                "y": _desktop_coordinate(
                    action.y,
                    origin=origin.y,
                    extent=screenshot.height,
                ),
            }
        )

    if isinstance(action, ScrollAction):
        if action.x is None:
            if action.y is None:
                return action
            raise ValueError("scroll x and y must be provided together")
        if action.y is None:
            raise ValueError("scroll x and y must be provided together")
        return action.model_copy(
            update={
                "x": _desktop_coordinate(
                    action.x,
                    origin=origin.x,
                    extent=screenshot.width,
                ),
                "y": _desktop_coordinate(
                    action.y,
                    origin=origin.y,
                    extent=screenshot.height,
                ),
            }
        )

    if isinstance(action, DragAction):
        return action.model_copy(
            update={
                "start_x": _desktop_coordinate(
                    action.start_x,
                    origin=origin.x,
                    extent=screenshot.width,
                ),
                "start_y": _desktop_coordinate(
                    action.start_y,
                    origin=origin.y,
                    extent=screenshot.height,
                ),
                "end_x": _desktop_coordinate(
                    action.end_x,
                    origin=origin.x,
                    extent=screenshot.width,
                ),
                "end_y": _desktop_coordinate(
                    action.end_y,
                    origin=origin.y,
                    extent=screenshot.height,
                ),
            }
        )

    return action


def _desktop_decision(
    decision: AgentDecision,
    observation: Observation,
) -> AgentDecision:
    return decision.model_copy(
        update={"action": _desktop_action(decision.action, observation)}
    )


class QwenTransformersPlanner:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN_MODEL,
        processor: object | None = None,
        model: object | None = None,
        processor_loader: Callable[..., object] = _default_processor_loader,
        model_loader: Callable[..., object] = _default_model_loader,
        model_dtype: object | None = None,
        max_image_side: int = 1280,
        max_new_tokens: int = 512,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if isinstance(max_image_side, bool) or max_image_side < 64:
            raise ValueError("max_image_side must be an integer of at least 64")
        if isinstance(max_new_tokens, bool) or not 1 <= max_new_tokens <= 2048:
            raise ValueError("max_new_tokens must be between 1 and 2048")
        if (processor is None) != (model is None):
            raise ValueError("processor and model must be provided together")
        self.model_name = model_name
        self._processor = processor
        self._model = model
        self._processor_loader = processor_loader
        self._model_loader = model_loader
        self._model_dtype = model_dtype
        self._max_image_side = max_image_side
        self._max_new_tokens = max_new_tokens

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            dtype = self._model_dtype
            if dtype is None:
                import torch

                dtype = torch.bfloat16

            try:
                self._processor = self._processor_loader(self.model_name)
                self._model = self._model_loader(
                    self.model_name,
                    dtype=dtype,
                    device_map="auto",
                )
            except Exception as exc:
                raise PlannerError(f"could not load local model {self.model_name}") from exc
        return cast(Any, self._processor), cast(Any, self._model)

    def _image(self, observation: Observation) -> Image.Image:
        rgb = observation.screenshot.image[:, :, ::-1]
        image = Image.fromarray(rgb)
        longest = max(image.size)
        if longest <= self._max_image_side:
            return image
        scale = self._max_image_side / longest
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size, Image.Resampling.LANCZOS)

    def _invoke(
        self,
        schema: type[ModelSchema],
        prompt: str,
        observation: Observation,
    ) -> ModelSchema:
        processor, model = self._ensure_loaded()
        schema_json = json.dumps(
            schema.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        coordinate_instruction = (
            f"\n{_QWEN_COORDINATE_INSTRUCTION}" if schema is AgentDecision else ""
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._image(observation)},
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}{coordinate_instruction}\n"
                            "Return exactly one JSON object matching this schema:\n"
                            f"{schema_json}"
                        ),
                    },
                ],
            }
        ]
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            input_length = int(inputs["input_ids"].shape[-1])
            generated = model.generate(
                **dict(inputs),
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
            trimmed = [row[input_length:] for row in generated]
            decoded = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            parsed = schema.model_validate_json(_json_object(decoded[0]))
            if isinstance(parsed, AgentDecision):
                return cast(ModelSchema, _desktop_decision(parsed, observation))
            return parsed
        except Exception as exc:
            raise PlannerError(f"local model failed to produce {schema.__name__}") from exc

    def create_plan(self, goal: str, observation: Observation) -> TaskPlan:
        return self._invoke(TaskPlan, build_plan_prompt(goal, observation), observation)

    def next_action(self, state: AgentState) -> AgentDecision:
        return self._invoke(
            AgentDecision,
            build_action_prompt(state),
            state.observation,
        )


__all__ = ["DEFAULT_QWEN_MODEL", "QwenTransformersPlanner"]
