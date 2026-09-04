import hashlib

import pytest
from pydantic import ValidationError

from gui_agent.agent.types import ClickAction


def test_training_example_preserves_provenance_and_typed_action() -> None:
    from gui_agent.training.schema import TrainingExample

    example = TrainingExample(
        sample_id="screenagent:episode-1:0",
        source="screenagent",
        episode_id="episode-1",
        instruction="Open settings",
        image_path="C:/dataset/images/step-0.png",
        text_observation="Settings button",
        target_action=ClickAction(x=20, y=30),
        source_revision="fixture-v1",
    )

    assert example.target_action == ClickAction(x=20, y=30)
    assert example.model_dump(mode="json")["target_action"]["kind"] == "click"


def test_training_manifest_rejects_malformed_output_hashes() -> None:
    from gui_agent.training.schema import TrainingManifest

    with pytest.raises(ValidationError, match="output_sha256"):
        TrainingManifest(
            seed=7,
            validation_ratio=0.2,
            records_seen=1,
            records_accepted=1,
            records_skipped=0,
            train_examples=1,
            validation_examples=0,
            source_counts={},
            skip_reasons={},
            input_sha256={"screenagent": hashlib.sha256(b"input").hexdigest()},
            source_revisions={"screenagent": ("fixture-v1",)},
            source_licenses={"screenagent": "Apache-2.0"},
            output_sha256={"train.jsonl": "not-a-sha"},
        )
