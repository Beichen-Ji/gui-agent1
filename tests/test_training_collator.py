from pathlib import Path
from typing import Any

import torch
from PIL import Image

from gui_agent.agent.prompts import get_prompt_profile
from gui_agent.agent.types import ClickAction
from gui_agent.training.schema import TrainingExample


class CharacterTokenizer:
    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) + 1 for character in text]}


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = CharacterTokenizer()

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **_kwargs: object,
    ) -> str:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"<{message['role']}>")
            content = message["content"]
            if isinstance(content, str):
                rendered.append(content)
            else:
                assert isinstance(content, list)
                for item in content:
                    assert isinstance(item, dict)
                    rendered.append("<image>" if item["type"] == "image" else str(item["text"]))
        return "".join(rendered)

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        **_kwargs: object,
    ) -> dict[str, Any]:
        assert len(text) == len(images)
        encoded = [self.tokenizer(value)["input_ids"] for value in text]
        width = max(len(row) for row in encoded) + 3
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(encoded):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.ones((len(encoded), 3, 2, 2)),
        }


def _example(image: Path, sample_id: str, instruction: str) -> TrainingExample:
    return TrainingExample(
        sample_id=sample_id,
        source="screenagent",
        episode_id=sample_id,
        instruction=instruction,
        image_path=image.as_posix(),
        text_observation="Visible button",
        target_action=ClickAction(x=10, y=10),
        source_revision="fixture-v1",
    )


def test_collator_masks_everything_except_assistant_json(tmp_path: Path) -> None:
    from gui_agent.training.collator import QwenSFTCollator
    from gui_agent.training.formatting import format_training_messages

    image = tmp_path / "screen.png"
    Image.new("RGB", (100, 50), "white").save(image)
    examples = [
        _example(image, "long", "Click the long visible button description"),
        _example(image, "short", "Click"),
    ]
    processor = FakeProcessor()
    profile = get_prompt_profile("week5-grounded")
    collator = QwenSFTCollator(processor, profile=profile)

    batch = collator(examples)

    labels = batch["labels"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    for index, example in enumerate(examples):
        messages = format_training_messages(example, profile)
        target = messages[-1]["content"]
        assert isinstance(target, str)
        target_ids = processor.tokenizer(target)["input_ids"]
        kept = labels[index][labels[index] != -100].tolist()
        assert kept == target_ids
        padding_labels = labels[index][attention_mask[index] == 0]
        assert torch.equal(padding_labels, torch.full_like(padding_labels, -100))
        trained_tokens = labels[index] != -100
        assert torch.equal(labels[index][trained_tokens], input_ids[index][trained_tokens])
