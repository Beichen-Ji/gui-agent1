from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from gui_agent.training.config import LoRATrainingConfig

Loader = Callable[..., object]
_VISION_PATH_PARTS = ("visual", "vision", "merger")


@dataclass(frozen=True, slots=True)
class ParameterSummary:
    trainable: int
    total: int

    @property
    def ratio(self) -> float:
        return self.trainable / self.total if self.total else 0.0


def _torch_dtype(name: str) -> object:
    import torch

    return getattr(torch, name)


def _default_quantization_factory(**kwargs: object) -> object:
    from transformers import BitsAndBytesConfig

    factory = cast(Loader, BitsAndBytesConfig)
    return factory(**kwargs)


def _default_processor_loader(model_name: str, **kwargs: object) -> object:
    from transformers import AutoProcessor

    loader = cast(Loader, AutoProcessor.from_pretrained)
    return loader(model_name, **kwargs)


def _default_model_loader(model_name: str, **kwargs: object) -> object:
    from transformers import AutoModelForMultimodalLM

    return AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)


def load_qlora_model(
    config: LoRATrainingConfig,
    *,
    processor_loader: Loader = _default_processor_loader,
    model_loader: Loader = _default_model_loader,
    quantization_factory: Loader = _default_quantization_factory,
) -> tuple[object, object]:
    dtype = _torch_dtype(config.bnb_compute_dtype)
    quantization = quantization_factory(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    processor = processor_loader(config.base_model, max_pixels=config.max_image_pixels)
    model = model_loader(
        config.base_model,
        quantization_config=quantization,
        device_map="auto",
        dtype=dtype,
    )
    return model, processor


def _is_vision_path(name: str) -> bool:
    lowered = name.casefold()
    return any(part in lowered for part in _VISION_PATH_PARTS)


def _expanded_target_modules(model: object, suffixes: tuple[str, ...]) -> list[str]:
    named_modules = cast(Any, model).named_modules()
    names = [name for name, _module in named_modules]
    selected = sorted(
        name
        for name in names
        if not _is_vision_path(name) and name.rsplit(".", 1)[-1] in suffixes
    )
    matched_suffixes = {name.rsplit(".", 1)[-1] for name in selected}
    missing = set(suffixes) - matched_suffixes
    if missing:
        raise ValueError(f"LoRA target modules were not found: {', '.join(sorted(missing))}")
    return selected


def _freeze_vision_parameters(model: object) -> None:
    for name, parameter in cast(Any, model).named_parameters():
        if _is_vision_path(name):
            parameter.requires_grad = False


def _assert_vision_is_frozen(model: object) -> None:
    trainable_vision = [
        name
        for name, parameter in cast(Any, model).named_parameters()
        if _is_vision_path(name) and parameter.requires_grad
    ]
    if trainable_vision:
        raise RuntimeError("LoRA created trainable parameters inside the frozen vision tower")


def attach_lora(
    model: object,
    config: LoRATrainingConfig,
    *,
    prepare_model: Loader | None = None,
    lora_config_factory: Loader | None = None,
    peft_model_factory: Callable[[object, object], object] | None = None,
) -> object:
    prepare = prepare_model
    config_factory = lora_config_factory
    model_factory = peft_model_factory
    if prepare is None or config_factory is None or model_factory is None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        prepare = prepare or cast(Loader, prepare_model_for_kbit_training)
        config_factory = config_factory or cast(Loader, LoraConfig)
        model_factory = model_factory or cast(
            Callable[[object, object], object],
            get_peft_model,
        )

    targets = _expanded_target_modules(model, config.target_modules)
    prepared = prepare(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
    )
    lora_config = config_factory(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    attached = model_factory(prepared, lora_config)
    cast(Any, attached).config.use_cache = False
    _freeze_vision_parameters(attached)
    _assert_vision_is_frozen(attached)
    return attached


def trainable_parameter_summary(model: object) -> ParameterSummary:
    parameters = [parameter for _name, parameter in cast(Any, model).named_parameters()]
    return ParameterSummary(
        trainable=sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        total=sum(parameter.numel() for parameter in parameters),
    )


__all__ = [
    "ParameterSummary",
    "attach_lora",
    "load_qlora_model",
    "trainable_parameter_summary",
]
