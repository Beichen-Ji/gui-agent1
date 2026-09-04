import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from gui_agent.agent.coordinates import action_to_grid
from gui_agent.agent.prompts import PromptProfile
from gui_agent.training.schema import TrainingExample
from gui_agent.types import ScreenRegion


def _image_bounds(path: Path) -> ScreenRegion:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"could not read training image: {path}") from error
    return ScreenRegion(0, 0, width, height)


def format_training_messages(
    example: TrainingExample,
    profile: PromptProfile,
) -> list[dict[str, object]]:
    image_path = Path(example.image_path)
    target = action_to_grid(
        example.target_action,
        bounds=_image_bounds(image_path),
    )
    observation = example.text_observation or "No text observation was supplied."
    system_text = (
        f"{profile.system_prompt}\n"
        f"Prompt profile: {profile.id}.\n"
        f"{profile.action_instruction}\n"
        f"{profile.coordinate_instruction}\n"
        "Return only one structured action JSON object."
    )
    user_text = f"Task: {example.instruction}\nText observation: {observation}"
    target_text = json.dumps(
        target.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.as_posix()},
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": target_text},
    ]


__all__ = ["format_training_messages"]
