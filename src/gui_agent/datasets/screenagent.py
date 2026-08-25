import json
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from gui_agent.agent.types import (
    AgentAction,
    ClickAction,
    HotkeyAction,
    ScrollAction,
    TypeTextAction,
    WaitAction,
)
from gui_agent.datasets.pipeline import AdapterReport
from gui_agent.datasets.schema import NormalizedGUIRecord


class DatasetAdapterError(ValueError):
    """A public dataset record cannot be mapped without guessing."""


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetAdapterError(f"{context}: expected an object")
    return cast(Mapping[str, object], value)


def _required_text(record: Mapping[str, object], *names: str, context: str) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise DatasetAdapterError(f"{context}: missing non-empty field {names!r}")


def _coordinate(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetAdapterError(f"{context}: coordinate must be an integer")
    return value


def _keyboard_keys(value: object) -> tuple[str, ...] | None:
    aliases = {
        "alt_l": "alt",
        "backspace": "backspace",
        "control_l": "ctrl",
        "escape": "esc",
        "return": "enter",
        "shift_l": "shift",
        "super_l": "win",
    }
    parts: Sequence[str]
    if isinstance(value, str):
        parts = value.split("+")
    elif isinstance(value, Sequence) and not isinstance(value, str) and all(
        isinstance(part, str) for part in value
    ):
        parts = cast(Sequence[str], value)
    else:
        return None
    keys = tuple(aliases.get(part.casefold(), part.casefold()) for part in parts)
    supported = {
        "alt",
        "backspace",
        "ctrl",
        "delete",
        "enter",
        "esc",
        "shift",
        "space",
        "tab",
        "win",
    } | {chr(code) for code in range(ord("a"), ord("z") + 1)} | {
        str(number) for number in range(10)
    }
    return keys if keys and len(keys) <= 4 and all(key in supported for key in keys) else None


def _adapt_action(raw: object, *, context: str) -> AgentAction | None:
    action = _mapping(raw, context=context)
    action_type = action.get("action_type")
    try:
        if action_type == "MouseAction":
            subtype = action.get("mouse_action_type")
            if subtype in {"click", "double_click"}:
                position = _mapping(action.get("mouse_position"), context=context)
                raw_button = action.get("mouse_button", "left")
                if raw_button not in {"left", "middle", "right"}:
                    raise DatasetAdapterError(f"{context}: unsupported mouse button")
                button = cast(Literal["left", "middle", "right"], raw_button)
                return ClickAction(
                    x=_coordinate(position.get("width"), context=context),
                    y=_coordinate(position.get("height"), context=context),
                    button=button,
                    clicks=2 if subtype == "double_click" else 1,
                )
            if subtype in {"scroll_down", "scroll_up"}:
                repeat = action.get("scroll_repeat", 1)
                clicks = min(20, max(1, _coordinate(repeat, context=context)))
                return ScrollAction(clicks=-clicks if subtype == "scroll_down" else clicks)
            return None
        if action_type == "KeyboardAction":
            subtype = action.get("keyboard_action_type")
            if subtype == "text":
                return TypeTextAction(
                    text=_required_text(action, "keyboard_text", context=context)
                )
            if subtype == "press":
                raw_key = action.get("keyboard_key")
                if raw_key is None:
                    raise DatasetAdapterError(f"{context}: missing keyboard_key")
                keys = _keyboard_keys(raw_key)
                return HotkeyAction(keys=keys) if keys is not None else None
            return None
        if action_type == "WaitAction":
            seconds = action.get("seconds", 1.0)
            if isinstance(seconds, bool) or not isinstance(seconds, int | float):
                raise DatasetAdapterError(f"{context}: wait seconds must be numeric")
            return WaitAction(seconds=min(5.0, max(0.0, float(seconds))))
        return None
    except ValidationError as exc:
        raise DatasetAdapterError(f"{context}: invalid action: {exc}") from exc


def _load_record(path: Path) -> Mapping[str, object]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), context=str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetAdapterError(f"{path}: cannot read ScreenAgent JSON: {exc}") from exc


def iter_screenagent(
    root: Path,
    *,
    split: str,
    source_revision: str = "unknown",
    report: AdapterReport | None = None,
) -> Iterator[NormalizedGUIRecord]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise DatasetAdapterError(f"{split_dir}: ScreenAgent split directory does not exist")

    session_dirs = sorted(
        (path for path in split_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    for session_dir in session_dirs:
        step_index = 0
        json_paths = sorted(
            (
                path
                for path in session_dir.glob("*_translate.json")
                if "_neg_" not in path.name
            ),
            key=lambda path: path.name.casefold(),
        )
        for path in json_paths:
            source = _load_record(path)
            context = str(path)
            instruction = _required_text(
                source,
                "task_prompt_en",
                "task_prompt",
                "task_prompt_zh",
                context=context,
            )
            session_id = _required_text(source, "session_id", context=context)
            image_name = _required_text(source, "saved_image_name", context=context)
            actions = source.get("actions")
            if not isinstance(actions, list):
                raise DatasetAdapterError(f"{path}: actions must be a list")
            for action_index, raw_action in enumerate(actions):
                action = _adapt_action(raw_action, context=f"{path} action[{action_index}]")
                if action is None:
                    raw_mapping = _mapping(
                        raw_action,
                        context=f"{path} action[{action_index}]",
                    )
                    issue = (
                        f"{path} action[{action_index}]: unsupported ScreenAgent action "
                        f"{raw_mapping.get('action_type')!r}"
                    )
                    if report is None:
                        warnings.warn(issue, RuntimeWarning, stacklevel=2)
                    else:
                        report.skip(issue)
                    continue
                image_path = (Path(split) / session_dir.name / "images" / image_name).as_posix()
                yield NormalizedGUIRecord(
                    source="screenagent",
                    record_type="trajectory_step",
                    split=split,
                    episode_id=session_id,
                    step_index=step_index,
                    instruction=instruction,
                    image_path=image_path,
                    action=action,
                    source_revision=source_revision,
                )
                step_index += 1


__all__ = ["DatasetAdapterError", "iter_screenagent"]
