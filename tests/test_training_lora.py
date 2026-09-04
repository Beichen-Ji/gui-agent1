import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from gui_agent.training.config import LoRATrainingConfig


def _config() -> LoRATrainingConfig:
    return LoRATrainingConfig(
        base_model="Qwen/Qwen3-VL-4B-Instruct",
        seed=20260904,
        validation_ratio=0.1,
        prompt_profile="week5-grounded",
        coordinate_grid_size=1000,
        load_in_4bit=True,
        bnb_quant_type="nf4",
        bnb_compute_dtype="bfloat16",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        freeze_vision_tower=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=0.0001,
        num_train_epochs=1.0,
        max_sequence_length=2048,
        max_image_pixels=401408,
        save_total_limit=2,
    )


def test_load_qlora_model_uses_approved_four_bit_parameters() -> None:
    from gui_agent.training.lora import load_qlora_model

    calls: dict[str, Any] = {}
    processor = object()
    model = object()

    def quantization_factory(**kwargs: object) -> object:
        calls["quantization"] = kwargs
        return "quantization-probe"

    def processor_loader(model_name: str, **kwargs: object) -> object:
        calls["processor"] = (model_name, kwargs)
        return processor

    def model_loader(model_name: str, **kwargs: object) -> object:
        calls["model"] = (model_name, kwargs)
        return model

    loaded_model, loaded_processor = load_qlora_model(
        _config(),
        processor_loader=processor_loader,
        model_loader=model_loader,
        quantization_factory=quantization_factory,
    )

    assert (loaded_model, loaded_processor) == (model, processor)
    assert calls["quantization"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    }
    assert calls["processor"] == (
        "Qwen/Qwen3-VL-4B-Instruct",
        {"max_pixels": 401408},
    )
    model_call = calls["model"]
    assert model_call[0] == "Qwen/Qwen3-VL-4B-Instruct"
    assert model_call[1]["quantization_config"] == "quantization-probe"
    assert model_call[1]["device_map"] == "auto"
    assert model_call[1]["dtype"] is torch.bfloat16


class ParameterProbe:
    def __init__(self, size: int, *, requires_grad: bool = True) -> None:
        self._size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._size


class ModuleTreeProbe:
    def __init__(self) -> None:
        suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        self.modules = {
            **{f"model.language.layers.0.{suffix}": object() for suffix in suffixes},
            "model.visual.layers.0.q_proj": object(),
            "model.vision.merger.o_proj": object(),
        }
        self.parameters = {
            "model.language.layers.0.q_proj.weight": ParameterProbe(100, requires_grad=False),
            "model.visual.layers.0.q_proj.weight": ParameterProbe(200),
            "model.vision.merger.o_proj.weight": ParameterProbe(50),
            "adapter.language.weight": ParameterProbe(10, requires_grad=False),
        }
        self.config = type("Config", (), {"use_cache": True})()

    def named_modules(self) -> object:
        return self.modules.items()

    def named_parameters(self) -> object:
        return self.parameters.items()


def test_attach_lora_expands_language_targets_and_keeps_vision_frozen() -> None:
    from gui_agent.training.lora import attach_lora, trainable_parameter_summary

    model = ModuleTreeProbe()
    captured: dict[str, object] = {}

    def prepare(candidate: object, **kwargs: object) -> object:
        captured["prepare"] = kwargs
        return candidate

    def lora_config_factory(**kwargs: object) -> object:
        captured["lora"] = kwargs
        return "lora-config"

    def peft_model_factory(candidate: object, lora_config: object) -> object:
        assert lora_config == "lora-config"
        model.parameters["adapter.language.weight"].requires_grad = True
        return candidate

    attached = attach_lora(
        model,
        _config(),
        prepare_model=prepare,
        lora_config_factory=lora_config_factory,
        peft_model_factory=peft_model_factory,
    )

    assert attached is model
    assert captured["prepare"] == {"use_gradient_checkpointing": True}
    lora_kwargs = cast(dict[str, Any], captured["lora"])
    target_modules = lora_kwargs["target_modules"]
    assert len(target_modules) == 7
    assert all("language" in name for name in target_modules)
    assert all(
        excluded not in name
        for name in target_modules
        for excluded in ("visual", "vision", "merger")
    )
    assert model.parameters["model.visual.layers.0.q_proj.weight"].requires_grad is False
    assert model.parameters["model.vision.merger.o_proj.weight"].requires_grad is False
    assert model.config.use_cache is False
    summary = trainable_parameter_summary(model)
    assert summary.trainable == 10
    assert summary.total == 360
    assert summary.ratio == pytest.approx(10 / 360)


def _training_data(path: Path) -> None:
    path.mkdir(parents=True)
    train = b'{"schema_version":1}\n'
    validation = b""
    (path / "train.jsonl").write_bytes(train)
    (path / "validation.jsonl").write_bytes(validation)
    manifest = {
        "schema_version": 1,
        "kind": "gui-agent-week5-training",
        "seed": 20260904,
        "validation_ratio": 0.1,
        "records_seen": 1,
        "records_accepted": 1,
        "records_skipped": 0,
        "train_examples": 1,
        "validation_examples": 0,
        "source_counts": {},
        "skip_reasons": {},
        "input_sha256": {"screenagent": hashlib.sha256(b"input").hexdigest()},
        "source_revisions": {"screenagent": ["fixture-v1"]},
        "source_licenses": {"screenagent": "Apache-2.0"},
        "output_sha256": {
            "train.jsonl": hashlib.sha256(train).hexdigest(),
            "validation.jsonl": hashlib.sha256(validation).hexdigest(),
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_run_training_writes_hashed_manifest_and_refuses_existing_output(
    tmp_path: Path,
) -> None:
    from gui_agent.training.trainer import BackendResult, run_training

    data_dir = tmp_path / "data"
    output_dir = tmp_path / "artifacts" / "adapter"
    _training_data(data_dir)
    calls: list[tuple[bool, Path | None]] = []

    def backend(
        config: LoRATrainingConfig,
        data: Path,
        output: Path,
        *,
        check_only: bool,
        resume_from_checkpoint: Path | None,
    ) -> BackendResult:
        assert config == _config()
        assert data == data_dir
        calls.append((check_only, resume_from_checkpoint))
        adapter = output / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "processor_config.json").write_text("{}", encoding="utf-8")
        return BackendResult(
            history=({"loss": 0.5, "step": 1},),
            trainable_parameters=10,
            total_parameters=360,
            peak_memory_bytes=1024,
            environment={"device": "fake-gpu", "torch": "test"},
            adapter_reloaded=True,
            structured_generation=True,
        )

    manifest = run_training(
        _config(),
        data_dir,
        output_dir,
        project_root=tmp_path,
        check_only=True,
        training_backend=backend,
    )

    assert calls == [(True, None)]
    assert manifest.kind == "gui-agent-week5-training-run"
    assert manifest.mode == "check"
    assert manifest.adapter_reloaded is True
    assert manifest.structured_generation is True
    assert manifest.output_sha256["adapter/adapter_model.safetensors"] == hashlib.sha256(
        b"adapter"
    ).hexdigest()
    stored = json.loads((output_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert stored["prompt_profile"] == "week5-grounded"

    with pytest.raises(ValueError, match="already exists"):
        run_training(
            _config(),
            data_dir,
            output_dir,
            project_root=tmp_path,
            training_backend=backend,
        )


def test_feasibility_check_fails_closed_when_reloaded_generation_is_invalid(
    tmp_path: Path,
) -> None:
    from gui_agent.training.trainer import BackendResult, run_training

    data_dir = tmp_path / "data"
    output_dir = tmp_path / "artifacts" / "invalid-check"
    _training_data(data_dir)

    def backend(
        _config: LoRATrainingConfig,
        _data: Path,
        output: Path,
        **_kwargs: object,
    ) -> BackendResult:
        adapter = output / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "processor_config.json").write_text("{}", encoding="utf-8")
        return BackendResult(
            history=(),
            trainable_parameters=10,
            total_parameters=100,
            peak_memory_bytes=1024,
            environment={"device": "fake-gpu"},
            adapter_reloaded=True,
            structured_generation=False,
        )

    with pytest.raises(RuntimeError, match="structured AgentAction"):
        run_training(
            _config(),
            data_dir,
            output_dir,
            project_root=tmp_path,
            check_only=True,
            training_backend=backend,
        )

    failure = json.loads(
        (output_dir / "failure-report.json").read_text(encoding="utf-8")
    )
    assert failure["error_type"] == "RuntimeError"


def test_training_cli_routes_check_and_explicit_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gui_agent.training import cli as training_cli

    captured: list[dict[str, object]] = []

    class ManifestProbe:
        def model_dump(self, **_kwargs: object) -> dict[str, object]:
            return {"kind": "probe"}

    def fake_run_training(
        config: LoRATrainingConfig,
        data_dir: Path,
        output_dir: Path,
        **kwargs: object,
    ) -> ManifestProbe:
        captured.append(
            {"config": config, "data": data_dir, "output": output_dir, **kwargs}
        )
        return ManifestProbe()

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)
    config_path = Path("configs/week5_qwen3vl_qlora.toml")

    assert training_cli.main(
        [
            "check",
            "--config",
            str(config_path),
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "check"),
        ]
    ) == 0
    assert training_cli.main(
        [
            "train",
            "--config",
            str(config_path),
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "train"),
            "--resume-from-checkpoint",
            str(tmp_path / "checkpoint-2"),
        ]
    ) == 0

    assert captured[0]["check_only"] is True
    assert captured[0]["resume_from_checkpoint"] is None
    assert captured[1]["check_only"] is False
    assert captured[1]["resume_from_checkpoint"] == tmp_path / "checkpoint-2"
