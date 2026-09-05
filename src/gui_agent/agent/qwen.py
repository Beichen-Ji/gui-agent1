import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from PIL import Image
from pydantic import BaseModel

from gui_agent.agent.coordinates import action_from_grid
from gui_agent.agent.planner import PlannerError
from gui_agent.agent.prompts import (
    PromptProfile,
    build_action_prompt,
    build_plan_prompt,
    build_replan_prompt,
    get_prompt_profile,
)
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    AgentState,
    Observation,
    ReplanContext,
    TaskPlan,
)
from gui_agent.types import ScreenRegion

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
ModelSchema = TypeVar("ModelSchema", bound=BaseModel)
AdapterLoader = Callable[[object, Path], object]


@dataclass(frozen=True, slots=True)
class _AdapterSpec:
    adapter_dir: Path
    prompt_profile: PromptProfile


def _default_processor_loader(model_name: str, **kwargs: object) -> object:
    from transformers import AutoProcessor

    loader = cast(Callable[..., object], AutoProcessor.from_pretrained)
    return loader(model_name, **kwargs)


def _default_model_loader(model_name: str, **kwargs: object) -> object:
    from transformers import AutoModelForMultimodalLM

    return AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)


def _default_adapter_loader(model: object, adapter_dir: Path) -> object:
    from peft import PeftModel

    loader = cast(Callable[..., object], PeftModel.from_pretrained)
    return loader(model, adapter_dir, is_trainable=False)


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return cast(dict[str, object], value)


def _validated_adapter(
    adapter_path: Path,
    *,
    model_name: str,
    requested_profile: str | PromptProfile | None,
) -> _AdapterSpec:
    resolved = adapter_path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"adapter path is not a directory: {adapter_path}")
    if (resolved / "run-manifest.json").is_file():
        output_root = resolved
        adapter_dir = output_root / "adapter"
    elif resolved.name == "adapter" and (resolved.parent / "run-manifest.json").is_file():
        output_root = resolved.parent
        adapter_dir = resolved
    else:
        raise ValueError("adapter path requires a sibling or child run-manifest.json")
    manifest = _read_json_object(
        output_root / "run-manifest.json",
        label="adapter run manifest",
    )
    if manifest.get("kind") != "gui-agent-week5-training-run":
        raise ValueError("adapter run manifest has an unsupported kind")
    if manifest.get("base_model") != model_name:
        raise ValueError("adapter base model does not match the requested model")
    if manifest.get("coordinate_grid_size") != 1000:
        raise ValueError("adapter coordinate grid does not match the Qwen runtime")
    manifest_profile = manifest.get("prompt_profile")
    if not isinstance(manifest_profile, str):
        raise ValueError("adapter run manifest has no prompt profile")
    selected_profile = get_prompt_profile(manifest_profile)
    if requested_profile is not None:
        requested = get_prompt_profile(requested_profile)
        if requested.id != selected_profile.id:
            raise ValueError("adapter prompt profile does not match the requested profile")

    adapter_config = _read_json_object(
        adapter_dir / "adapter_config.json",
        label="adapter config",
    )
    if adapter_config.get("base_model_name_or_path") != model_name:
        raise ValueError("adapter config base model does not match the requested model")
    raw_hashes = manifest.get("output_sha256")
    if not isinstance(raw_hashes, dict):
        raise ValueError("adapter run manifest has no output hashes")
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        path = adapter_dir / filename
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"adapter file is missing: {path}") from error
        expected = raw_hashes.get(f"adapter/{filename}")
        if expected != actual:
            raise ValueError(f"adapter file hash mismatch: {filename}")
    return _AdapterSpec(adapter_dir=adapter_dir, prompt_profile=selected_profile)


def _json_object(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model output does not contain a JSON object")
    return stripped[start : end + 1]


def _desktop_action(action: AgentAction, observation: Observation) -> AgentAction:
    screenshot = observation.screenshot
    bounds = ScreenRegion(
        screenshot.origin.x,
        screenshot.origin.y,
        screenshot.width,
        screenshot.height,
    )
    return action_from_grid(action, bounds=bounds)


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
        prompt_profile: str | PromptProfile | None = None,
        adapter_path: Path | None = None,
        adapter_loader: AdapterLoader = _default_adapter_loader,
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
        self._adapter = (
            _validated_adapter(
                adapter_path,
                model_name=model_name,
                requested_profile=prompt_profile,
            )
            if adapter_path is not None
            else None
        )
        self._prompt_profile = (
            self._adapter.prompt_profile
            if self._adapter is not None
            else get_prompt_profile(prompt_profile or "week4-baseline")
        )
        self._adapter_loader = adapter_loader
        self._adapter_loaded = False

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
        if self._adapter is not None and not self._adapter_loaded:
            try:
                self._model = self._adapter_loader(
                    self._model,
                    self._adapter.adapter_dir,
                )
                self._adapter_loaded = True
            except Exception as exc:
                raise PlannerError(
                    f"could not load LoRA adapter {self._adapter.adapter_dir}"
                ) from exc
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
            f"\n{self._prompt_profile.coordinate_instruction}"
            if schema is AgentDecision
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self._prompt_profile.system_prompt}\n"
                    f"Prompt profile: {self._prompt_profile.id}."
                ),
            },
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
        return self._invoke(
            TaskPlan,
            build_plan_prompt(goal, observation, profile=self._prompt_profile),
            observation,
        )

    def next_action(self, state: AgentState) -> AgentDecision:
        return self._invoke(
            AgentDecision,
            build_action_prompt(state, profile=self._prompt_profile),
            state.observation,
        )

    def revise_plan(self, state: AgentState, failure: ReplanContext) -> TaskPlan:
        return self._invoke(
            TaskPlan,
            build_replan_prompt(state, failure, profile=self._prompt_profile),
            state.observation,
        )


__all__ = ["DEFAULT_QWEN_MODEL", "QwenTransformersPlanner"]
