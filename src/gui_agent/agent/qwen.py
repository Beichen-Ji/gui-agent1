import json
from collections.abc import Callable
from typing import Any, TypeVar, cast

from PIL import Image
from pydantic import BaseModel

from gui_agent.agent.planner import PlannerError
from gui_agent.agent.prompts import build_action_prompt, build_plan_prompt
from gui_agent.agent.types import AgentDecision, AgentState, Observation, TaskPlan

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
ModelSchema = TypeVar("ModelSchema", bound=BaseModel)


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


class QwenTransformersPlanner:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN_MODEL,
        processor: object | None = None,
        model: object | None = None,
        processor_loader: Callable[..., object] = _default_processor_loader,
        model_loader: Callable[..., object] = _default_model_loader,
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
        self._max_image_side = max_image_side
        self._max_new_tokens = max_new_tokens

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            import torch

            try:
                self._processor = self._processor_loader(self.model_name)
                self._model = self._model_loader(
                    self.model_name,
                    dtype=torch.bfloat16,
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._image(observation)},
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}\nReturn exactly one JSON object matching this schema:\n"
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
            return schema.model_validate_json(_json_object(decoded[0]))
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
