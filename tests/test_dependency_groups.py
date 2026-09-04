import tomllib
from pathlib import Path


def test_training_extra_declares_its_runtime_dependencies() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    training_dependencies = set(
        project["project"]["optional-dependencies"]["training"]
    )

    assert "pydantic>=2,<3" in training_dependencies
