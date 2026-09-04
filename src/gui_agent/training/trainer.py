import gc
import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from gui_agent.agent.prompts import get_prompt_profile
from gui_agent.agent.types import AgentAction
from gui_agent.training.collator import QwenSFTCollator
from gui_agent.training.config import LoRATrainingConfig, validate_training_output_path
from gui_agent.training.formatting import format_training_messages
from gui_agent.training.lora import (
    attach_lora,
    load_qlora_model,
    trainable_parameter_summary,
)
from gui_agent.training.schema import TrainingExample, TrainingManifest

HistoryValue = int | float | str | bool | None


@dataclass(frozen=True, slots=True)
class BackendResult:
    history: tuple[dict[str, HistoryValue], ...]
    trainable_parameters: int
    total_parameters: int
    peak_memory_bytes: int
    environment: dict[str, str]
    adapter_reloaded: bool
    structured_generation: bool
    effective_max_image_pixels: int | None = None


class TrainingRunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    kind: Literal["gui-agent-week5-training-run"] = "gui-agent-week5-training-run"
    mode: Literal["check", "train"]
    base_model: str
    prompt_profile: str
    seed: int = Field(ge=0)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trainable_parameters: int = Field(ge=0)
    total_parameters: int = Field(ge=0)
    trainable_ratio: float = Field(ge=0.0, le=1.0)
    peak_memory_bytes: int = Field(ge=0)
    effective_max_image_pixels: int = Field(ge=1)
    adapter_reloaded: bool
    structured_generation: bool
    resume_from_checkpoint: str | None
    environment: dict[str, str]
    history: tuple[dict[str, HistoryValue], ...]
    output_sha256: dict[str, str]


TrainingBackend = Callable[
    [LoRATrainingConfig, Path, Path],
    BackendResult,
]


def _canonical_config(config: LoRATrainingConfig) -> bytes:
    return json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validated_data_manifest(data_dir: Path) -> tuple[TrainingManifest, bytes]:
    path = data_dir / "manifest.json"
    try:
        raw = path.read_bytes()
        manifest = TrainingManifest.model_validate_json(raw)
        for filename in ("train.jsonl", "validation.jsonl"):
            actual = hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
            if manifest.output_sha256.get(filename) != actual:
                raise ValueError(f"training data hash mismatch: {filename}")
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid Week 5 training data: {data_dir}") from error
    return manifest, raw


def _load_examples(path: Path) -> list[TrainingExample]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        examples = [
            TrainingExample.model_validate_json(line) for line in lines if line.strip()
        ]
    except (OSError, ValueError) as error:
        raise ValueError(f"could not load training examples: {path}") from error
    if not examples:
        raise ValueError("training split must contain at least one example")
    return examples


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _model_device(model: object) -> object:
    return cast(Any, model).device


def _structured_adapter_generation(
    config: LoRATrainingConfig,
    adapter_dir: Path,
    example: TrainingExample,
) -> tuple[bool, bool]:
    import torch
    from peft import PeftModel

    base, processor = load_qlora_model(config)
    peft_loader = cast(Any, PeftModel.from_pretrained)
    reloaded = peft_loader(base, adapter_dir, is_trainable=False)
    processor_api = cast(Any, processor)
    messages = format_training_messages(
        example,
        get_prompt_profile(config.prompt_profile),
    )[:-1]
    image_content = cast(list[dict[str, object]], messages[1]["content"])
    with Image.open(example.image_path) as source:
        image_content[0]["image"] = source.convert("RGB")
        inputs = processor_api.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(_model_device(reloaded))
    input_length = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = cast(Any, reloaded).generate(
            **dict(inputs),
            max_new_tokens=256,
            do_sample=False,
        )
    decoded = processor_api.batch_decode(
        [row[input_length:] for row in generated],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    start, end = decoded.find("{"), decoded.rfind("}")
    if start < 0 or end < start:
        return True, False
    try:
        TypeAdapter(AgentAction).validate_json(decoded[start : end + 1])
    except ValueError:
        return True, False
    return True, True


def _default_training_backend(
    config: LoRATrainingConfig,
    data_dir: Path,
    output_dir: Path,
    *,
    check_only: bool,
    resume_from_checkpoint: Path | None,
) -> BackendResult:
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(config.seed)
    examples = _load_examples(data_dir / "train.jsonl")
    model, processor = load_qlora_model(config)
    model = attach_lora(model, config)
    summary = trainable_parameter_summary(model)
    if summary.trainable == 0:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=1 if check_only else -1,
        save_strategy="no" if check_only else "steps",
        save_steps=50,
        save_total_limit=config.save_total_limit,
        logging_steps=1,
        bf16=config.bnb_compute_dtype == "bfloat16",
        fp16=config.bnb_compute_dtype == "float16",
        remove_unused_columns=False,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Trainer(
        model=cast(Any, model),
        args=arguments,
        train_dataset=examples,
        data_collator=QwenSFTCollator(
            processor,
            profile=get_prompt_profile(config.prompt_profile),
        ),
        processing_class=cast(Any, processor),
    )
    result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    adapter_dir = output_dir / "adapter"
    cast(Any, model).save_pretrained(adapter_dir, safe_serialization=True)
    cast(Any, processor).save_pretrained(adapter_dir)
    history = tuple(
        {
            key: cast(HistoryValue, value)
            for key, value in row.items()
            if isinstance(value, (int, float, str, bool)) or value is None
        }
        for row in trainer.state.log_history
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    environment = _package_versions()
    environment["cuda_available"] = str(torch.cuda.is_available())
    environment["cuda_version"] = str(torch.version.cuda)
    environment["device"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    del trainer, model, processor, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if check_only:
        reloaded, structured = _structured_adapter_generation(
            config,
            adapter_dir,
            examples[0],
        )
    else:
        reloaded, structured = False, False
    return BackendResult(
        history=history,
        trainable_parameters=summary.trainable,
        total_parameters=summary.total,
        peak_memory_bytes=peak_memory,
        environment=environment,
        adapter_reloaded=reloaded,
        structured_generation=structured,
        effective_max_image_pixels=config.max_image_pixels,
    )


def _output_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in {"run-manifest.json", "failure-report.json"}
    }


def _write_failure_report(
    output_dir: Path,
    error: Exception,
    *,
    max_image_pixels: int,
) -> None:
    (output_dir / "failure-report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "gui-agent-week5-training-failure",
                "error_type": type(error).__name__,
                "message": str(error)[:2000],
                "effective_max_image_pixels": max_image_pixels,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_training(
    config: LoRATrainingConfig,
    data_dir: Path,
    output_dir: Path,
    *,
    resume_from_checkpoint: Path | None = None,
    check_only: bool = False,
    project_root: Path | None = None,
    training_backend: Callable[..., BackendResult] = _default_training_backend,
) -> TrainingRunManifest:
    root = (project_root or Path.cwd()).resolve()
    resolved_output = validate_training_output_path(output_dir, project_root=root)
    if resolved_output.exists():
        raise ValueError(f"training output already exists: {resolved_output}")
    _data_manifest, data_manifest_bytes = _validated_data_manifest(data_dir)
    resolved_output.mkdir(parents=True)
    try:
        result = training_backend(
            config,
            data_dir,
            resolved_output,
            check_only=check_only,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        required_files = (
            resolved_output / "adapter" / "adapter_model.safetensors",
            resolved_output / "adapter" / "adapter_config.json",
            resolved_output / "adapter" / "processor_config.json",
        )
        missing_files = [path.name for path in required_files if not path.is_file()]
        if missing_files:
            raise RuntimeError(
                f"training backend did not save required files: {', '.join(missing_files)}"
            )
        if check_only and not result.adapter_reloaded:
            raise RuntimeError("feasibility check could not reload the saved adapter")
        if check_only and not result.structured_generation:
            raise RuntimeError(
                "feasibility check did not generate a structured AgentAction"
            )
    except Exception as error:
        _write_failure_report(
            resolved_output,
            error,
            max_image_pixels=config.max_image_pixels,
        )
        raise
    ratio = result.trainable_parameters / result.total_parameters
    manifest = TrainingRunManifest(
        mode="check" if check_only else "train",
        base_model=config.base_model,
        prompt_profile=config.prompt_profile,
        seed=config.seed,
        config_sha256=hashlib.sha256(_canonical_config(config)).hexdigest(),
        data_manifest_sha256=hashlib.sha256(data_manifest_bytes).hexdigest(),
        trainable_parameters=result.trainable_parameters,
        total_parameters=result.total_parameters,
        trainable_ratio=ratio,
        peak_memory_bytes=result.peak_memory_bytes,
        effective_max_image_pixels=(
            result.effective_max_image_pixels or config.max_image_pixels
        ),
        adapter_reloaded=result.adapter_reloaded,
        structured_generation=result.structured_generation,
        resume_from_checkpoint=(
            resume_from_checkpoint.as_posix()
            if resume_from_checkpoint is not None
            else None
        ),
        environment=result.environment,
        history=result.history,
        output_sha256=_output_hashes(resolved_output),
    )
    (resolved_output / "run-manifest.json").write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["BackendResult", "TrainingRunManifest", "run_training"]
