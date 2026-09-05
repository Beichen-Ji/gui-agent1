import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoRATrainingConfig(BaseModel):
    """Validated, immutable settings for one QLoRA training run."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    base_model: str = Field(min_length=1)
    seed: int = Field(ge=0)
    validation_ratio: float = Field(gt=0.0, lt=1.0)
    prompt_profile: str = Field(min_length=1)
    coordinate_grid_size: Literal[1000]
    load_in_4bit: Literal[True]
    bnb_quant_type: Literal["nf4", "fp4"]
    bnb_compute_dtype: Literal["bfloat16", "float16", "float32"]
    lora_r: int = Field(ge=1, le=256)
    lora_alpha: int = Field(ge=1, le=1024)
    lora_dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = Field(min_length=1)
    freeze_vision_tower: Literal[True]
    per_device_train_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    gradient_checkpointing: bool
    learning_rate: float = Field(gt=0.0, le=1.0)
    num_train_epochs: float = Field(gt=0.0)
    max_sequence_length: int = Field(ge=128)
    max_image_pixels: int = Field(ge=1)
    save_total_limit: int = Field(ge=1)

    @field_validator("target_modules", mode="before")
    @classmethod
    def _tuple_target_modules(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("target_modules")
    @classmethod
    def _unique_target_modules(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("target_modules must not contain blank names")
        if len(set(value)) != len(value):
            raise ValueError("target_modules must not contain duplicates")
        return value


def load_training_config(path: Path) -> LoRATrainingConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return LoRATrainingConfig.model_validate(raw)


def validate_training_output_path(path: Path, *, project_root: Path) -> Path:
    resolved = path.resolve()
    root = project_root.resolve()
    allowed_roots = (root / "artifacts", root / "checkpoints")
    if not any(resolved.is_relative_to(candidate) for candidate in allowed_roots):
        raise ValueError("training output must be inside artifacts or checkpoints")
    return resolved


__all__ = [
    "LoRATrainingConfig",
    "load_training_config",
    "validate_training_output_path",
]
