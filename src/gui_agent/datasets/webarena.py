import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

from gui_agent.datasets.schema import NormalizedGUIRecord
from gui_agent.datasets.screenagent import DatasetAdapterError


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetAdapterError(f"{context}: expected an object")
    return cast(Mapping[str, object], value)


def _configs(path: Path) -> Iterator[tuple[str, Mapping[str, object]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetAdapterError(f"{path}: cannot read WebArena JSON: {exc}") from exc
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            context = f"{path}[{index}]"
            yield context, _mapping(item, context=context)
        return
    yield str(path), _mapping(payload, context=str(path))


def iter_webarena(
    config_dir: Path,
    *,
    source_revision: str = "unknown",
) -> Iterator[NormalizedGUIRecord]:
    if not config_dir.is_dir():
        raise DatasetAdapterError(f"{config_dir}: WebArena config directory does not exist")
    raw_config = config_dir / "test.raw.json"
    paths = [raw_config] if raw_config.is_file() else sorted(
        config_dir.rglob("*.json"), key=lambda path: path.as_posix().casefold()
    )
    if not paths:
        raise DatasetAdapterError(f"{config_dir}: no WebArena JSON configs found")

    for path in paths:
        for context, config in _configs(path):
            task_id = config.get("task_id")
            if isinstance(task_id, bool) or not isinstance(task_id, int | str):
                raise DatasetAdapterError(f"{context}: task_id must be an integer or string")
            instruction = config.get("intent") or config.get("intent_template")
            if not isinstance(instruction, str) or not instruction.strip():
                raise DatasetAdapterError(f"{context}: intent must be non-empty text")
            evaluation = config.get("eval")
            if not isinstance(evaluation, Mapping) or not evaluation:
                raise DatasetAdapterError(f"{context}: eval must define success criteria")
            success_criteria = json.dumps(
                evaluation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )[:4000]
            yield NormalizedGUIRecord(
                source="webarena",
                record_type="task",
                split="test",
                episode_id=str(task_id),
                step_index=0,
                instruction=instruction.strip(),
                success_criteria=success_criteria,
                source_revision=source_revision,
            )


__all__ = ["iter_webarena"]
