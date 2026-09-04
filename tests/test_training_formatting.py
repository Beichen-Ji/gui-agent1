import json
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.agent.types import ClickAction
from gui_agent.training.schema import TrainingExample


def _example(
    image_path: Path,
    *,
    instruction: str = "Click the bottom-right item",
) -> TrainingExample:
    return TrainingExample(
        sample_id="screenagent:example",
        source="screenagent",
        episode_id="episode-1",
        instruction=instruction,
        image_path=image_path.as_posix(),
        text_observation="The item is visible in the lower-right corner",
        target_action=ClickAction(x=100, y=50),
        source_revision="fixture-v1",
    )


def test_prompt_profiles_are_immutable_and_keep_versioned_constraints() -> None:
    from gui_agent.agent.prompts import PROMPT_PROFILES, get_prompt_profile

    baseline = get_prompt_profile("week4-baseline")
    grounded = get_prompt_profile("week5-grounded")

    assert baseline.id == "week4-baseline"
    assert grounded.id == "week5-grounded"
    assert "exactly one" in baseline.action_instruction
    assert "visible image evidence" in grounded.action_instruction
    assert "0 through 999" in grounded.coordinate_instruction
    with pytest.raises(TypeError):
        PROMPT_PROFILES["modified"] = baseline  # type: ignore[index]


def test_training_messages_use_real_image_size_and_grid_target(tmp_path: Path) -> None:
    from gui_agent.agent.prompts import get_prompt_profile
    from gui_agent.training.formatting import format_training_messages

    image_path = tmp_path / "screen.png"
    Image.new("RGB", (101, 51), "white").save(image_path)
    profile = get_prompt_profile("week5-grounded")

    messages = format_training_messages(_example(image_path), profile)

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    system_content = messages[0]["content"]
    assert isinstance(system_content, str)
    assert "week5-grounded" in system_content
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0] == {"type": "image", "image": image_path.as_posix()}
    target_content = messages[2]["content"]
    assert isinstance(target_content, str)
    target = json.loads(target_content)
    assert target == {
        "button": "left",
        "clicks": 1,
        "kind": "click",
        "x": 999,
        "y": 999,
    }
