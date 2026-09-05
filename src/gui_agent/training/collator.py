from collections.abc import Sequence
from typing import Any, cast

import torch
from PIL import Image

from gui_agent.agent.prompts import PromptProfile
from gui_agent.training.formatting import format_training_messages
from gui_agent.training.schema import TrainingExample


def _subsequence_start(row: list[int], target: list[int]) -> int:
    for start in range(len(row) - len(target), -1, -1):
        if row[start : start + len(target)] == target:
            return start
    raise ValueError("assistant target tokens were not found in the formatted batch")


class QwenSFTCollator:
    def __init__(self, processor: object, *, profile: PromptProfile) -> None:
        self._processor = cast(Any, processor)
        self._profile = profile

    def __call__(self, examples: Sequence[TrainingExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("at least one training example is required")
        messages = [format_training_messages(example, self._profile) for example in examples]
        rendered = [
            self._processor.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=False,
            )
            for item in messages
        ]
        images = [Image.open(example.image_path).convert("RGB") for example in examples]
        try:
            batch = cast(
                dict[str, Any],
                self._processor(
                    text=rendered,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                ),
            )
        finally:
            for image in images:
                image.close()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = torch.full_like(input_ids, -100)
        for index, item in enumerate(messages):
            target = item[-1]["content"]
            if not isinstance(target, str):
                raise TypeError("assistant training content must be text")
            encoded = self._processor.tokenizer(target, add_special_tokens=False)["input_ids"]
            target_ids = cast(list[int], encoded)
            active_length = int(attention_mask[index].sum().item())
            row = cast(list[int], input_ids[index, :active_length].tolist())
            start = _subsequence_start(row, target_ids)
            labels[index, start : start + len(target_ids)] = input_ids[
                index, start : start + len(target_ids)
            ]
        batch["labels"] = labels
        return batch


__all__ = ["QwenSFTCollator"]
