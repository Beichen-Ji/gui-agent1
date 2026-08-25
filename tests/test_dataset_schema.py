import pytest
from pydantic import ValidationError

from gui_agent.agent.types import ClickAction
from gui_agent.datasets.schema import DatasetManifest, NormalizedGUIRecord


def test_valid_trajectory_record_parses_nested_action() -> None:
    record = NormalizedGUIRecord.model_validate(
        {
            "source": "screenagent",
            "record_type": "trajectory_step",
            "split": "train",
            "episode_id": "example-session",
            "step_index": 0,
            "instruction": "Open the browser",
            "image_path": "images/example.jpg",
            "action": {"kind": "click", "x": 410, "y": 220},
            "source_revision": "abc123",
        }
    )

    assert record.schema_version == 1
    assert record.action == ClickAction(x=410, y=220)


def test_trajectory_record_requires_an_action() -> None:
    with pytest.raises(ValidationError, match="action"):
        NormalizedGUIRecord(
            source="mind2web",
            record_type="trajectory_step",
            split="train",
            episode_id="episode-1",
            step_index=0,
            instruction="Select the result",
            text_observation="A search result is visible",
            source_revision="main",
        )


def test_trajectory_record_requires_image_or_text_observation() -> None:
    with pytest.raises(ValidationError, match="observation"):
        NormalizedGUIRecord(
            source="mind2web",
            record_type="trajectory_step",
            split="train",
            episode_id="episode-1",
            step_index=0,
            instruction="Select the result",
            action=ClickAction(x=4, y=5),
            source_revision="main",
        )


def test_webarena_task_requires_success_criteria() -> None:
    with pytest.raises(ValidationError, match="success_criteria"):
        NormalizedGUIRecord(
            source="webarena",
            record_type="task",
            split="test",
            episode_id="task-1",
            step_index=0,
            instruction="Find an item",
            source_revision="main",
        )


def test_dataset_record_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="private_note"):
        NormalizedGUIRecord.model_validate(
            {
                "source": "webarena",
                "record_type": "task",
                "split": "test",
                "episode_id": "task-1",
                "step_index": 0,
                "instruction": "Find an item",
                "success_criteria": "The item name is shown",
                "source_revision": "main",
                "private_note": "do not retain",
            }
        )


def test_manifest_accepts_a_real_sha256_and_is_frozen() -> None:
    manifest = DatasetManifest(
        source="screenagent",
        source_url="https://github.com/niuzaisheng/ScreenAgent",
        source_revision="abc123",
        license="repository metadata; verify before redistribution",
        records_written=10,
        records_skipped=2,
        output_file="records.jsonl",
        sha256="a" * 64,
    )

    assert manifest.schema_version == 1
    with pytest.raises(ValidationError, match="frozen"):
        manifest.records_written = 11


def test_manifest_rejects_invalid_sha256() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        DatasetManifest(
            source="screenagent",
            source_url="https://github.com/niuzaisheng/ScreenAgent",
            source_revision="abc123",
            license="unknown",
            records_written=1,
            records_skipped=0,
            output_file="records.jsonl",
            sha256="not-a-sha",
        )
