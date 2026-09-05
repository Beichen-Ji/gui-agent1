from pathlib import Path

import pytest
from pydantic import ValidationError

VALID_CONFIG_TOML = """
base_model = "Qwen/Qwen3-VL-4B-Instruct"
seed = 20260904
validation_ratio = 0.10
prompt_profile = "week5-grounded"
coordinate_grid_size = 1000
load_in_4bit = true
bnb_quant_type = "nf4"
bnb_compute_dtype = "bfloat16"
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
freeze_vision_tower = true
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
gradient_checkpointing = true
learning_rate = 0.0001
num_train_epochs = 1.0
max_sequence_length = 2048
max_image_pixels = 401408
save_total_limit = 2
""".strip()


def _write_config(tmp_path: Path, text: str = VALID_CONFIG_TOML) -> Path:
    path = tmp_path / "training.toml"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_load_training_config_accepts_the_approved_week5_values(tmp_path: Path) -> None:
    from gui_agent.training.config import load_training_config

    config = load_training_config(_write_config(tmp_path))

    assert config.base_model == "Qwen/Qwen3-VL-4B-Instruct"
    assert config.seed == 20260904
    assert config.validation_ratio == 0.10
    assert config.prompt_profile == "week5-grounded"
    assert config.coordinate_grid_size == 1000
    assert config.load_in_4bit is True
    assert config.target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert config.max_image_pixels == 401408


def test_load_training_config_rejects_unknown_fields(tmp_path: Path) -> None:
    from gui_agent.training.config import load_training_config

    path = _write_config(tmp_path, VALID_CONFIG_TOML + "\nunknown_option = 3")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_training_config(path)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("per_device_train_batch_size = 1", "per_device_train_batch_size = 0"),
        ("validation_ratio = 0.10", "validation_ratio = 0.0"),
        ("validation_ratio = 0.10", "validation_ratio = 1.0"),
    ],
)
def test_load_training_config_rejects_unsafe_numeric_values(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    from gui_agent.training.config import load_training_config

    path = _write_config(tmp_path, VALID_CONFIG_TOML.replace(original, replacement))

    with pytest.raises(ValidationError):
        load_training_config(path)


def test_validate_training_output_path_allows_only_artifact_roots(tmp_path: Path) -> None:
    from gui_agent.training.config import validate_training_output_path

    project_root = tmp_path / "project"

    assert validate_training_output_path(
        project_root / "artifacts" / "week5" / "adapter",
        project_root=project_root,
    ) == (project_root / "artifacts" / "week5" / "adapter").resolve()
    with pytest.raises(ValueError, match="artifacts or checkpoints"):
        validate_training_output_path(
            project_root / "src" / "adapter",
            project_root=project_root,
        )


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ('base_model = "Qwen/Qwen3-VL-4B-Instruct"', 'base_model = " "'),
        ("seed = 20260904", "seed = -1"),
        ("coordinate_grid_size = 1000", "coordinate_grid_size = 999"),
        ("load_in_4bit = true", "load_in_4bit = false"),
        ("lora_r = 8", "lora_r = 0"),
        ("lora_alpha = 16", "lora_alpha = 0"),
        ("lora_dropout = 0.05", "lora_dropout = 1.0"),
        ("freeze_vision_tower = true", "freeze_vision_tower = false"),
        ("gradient_accumulation_steps = 8", "gradient_accumulation_steps = 0"),
        ("learning_rate = 0.0001", "learning_rate = 0.0"),
        ("num_train_epochs = 1.0", "num_train_epochs = 0.0"),
        ("max_sequence_length = 2048", "max_sequence_length = 127"),
        ("max_image_pixels = 401408", "max_image_pixels = 0"),
        ("save_total_limit = 2", "save_total_limit = 0"),
    ],
)
def test_load_training_config_rejects_values_that_break_the_training_contract(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    from gui_agent.training.config import load_training_config

    path = _write_config(tmp_path, VALID_CONFIG_TOML.replace(original, replacement))

    with pytest.raises(ValidationError):
        load_training_config(path)


def test_load_training_config_rejects_duplicate_target_modules(tmp_path: Path) -> None:
    from gui_agent.training.config import load_training_config

    path = _write_config(
        tmp_path,
        VALID_CONFIG_TOML.replace(
            'target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", '
            '"gate_proj", "up_proj", "down_proj"]',
            'target_modules = ["q_proj", "q_proj"]',
        ),
    )

    with pytest.raises(ValidationError, match="target_modules"):
        load_training_config(path)
