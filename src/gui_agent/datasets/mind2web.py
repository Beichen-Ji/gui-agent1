import json
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import cast

from gui_agent.agent.types import AgentAction, ClickAction, TypeTextAction
from gui_agent.datasets.pipeline import AdapterReport
from gui_agent.datasets.schema import NormalizedGUIRecord
from gui_agent.datasets.screenagent import DatasetAdapterError


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetAdapterError(f"{context}: expected an object")
    return cast(Mapping[str, object], value)


def _text(value: object, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetAdapterError(f"{context}: {field} must be non-empty text")
    return value.strip()


def _target_center(action: Mapping[str, object], *, context: str) -> tuple[int, int]:
    candidates = action.get("pos_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, str) or not candidates:
        raise DatasetAdapterError(f"{context}: CLICK has no positive target candidate")
    candidate = _mapping(candidates[0], context=context)
    attributes_value = candidate.get("attributes")
    if isinstance(attributes_value, str):
        try:
            attributes = _mapping(json.loads(attributes_value), context=context)
        except json.JSONDecodeError as exc:
            raise DatasetAdapterError(f"{context}: invalid candidate attributes JSON") from exc
    else:
        attributes = _mapping(attributes_value, context=context)
    rect = attributes.get("bounding_box_rect")
    if not isinstance(rect, str):
        raise DatasetAdapterError(f"{context}: target has no bounding_box_rect")
    try:
        left, top, width, height = (float(part.strip()) for part in rect.split(","))
    except (TypeError, ValueError) as exc:
        raise DatasetAdapterError(f"{context}: malformed bounding_box_rect {rect!r}") from exc
    return round(left + width / 2), round(top + height / 2)


def _adapt_action(action: Mapping[str, object], *, context: str) -> AgentAction:
    operation = _mapping(action.get("operation"), context=context)
    op = _text(operation.get("op"), field="operation.op", context=context).upper()
    if op == "CLICK":
        x, y = _target_center(action, context=context)
        return ClickAction(x=x, y=y)
    if op == "TYPE":
        return TypeTextAction(
            text=_text(operation.get("value"), field="operation.value", context=context)
        )
    raise DatasetAdapterError(f"{context}: unsupported Mind2Web operation {op!r}")


def iter_mind2web(
    rows: Iterable[Mapping[str, object]],
    *,
    split: str = "train",
    source_revision: str = "main",
    report: AdapterReport | None = None,
) -> Iterator[NormalizedGUIRecord]:
    for row_index, row in enumerate(rows):
        context = f"Mind2Web row[{row_index}]"
        episode_id = _text(row.get("annotation_id"), field="annotation_id", context=context)
        instruction = _text(row.get("confirmed_task"), field="confirmed_task", context=context)
        actions = row.get("actions")
        if not isinstance(actions, Sequence) or isinstance(actions, str):
            raise DatasetAdapterError(f"{context}: actions must be a sequence")
        for step_index, raw_action in enumerate(actions):
            action_record = _mapping(raw_action, context=f"{context} action[{step_index}]")
            action_context = f"{context} action[{step_index}]"
            try:
                action = _adapt_action(action_record, context=action_context)
            except DatasetAdapterError as exc:
                if report is None:
                    warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
                else:
                    report.skip(str(exc))
                continue
            cleaned_html = action_record.get("cleaned_html")
            observation = (
                cleaned_html.strip()[:8000]
                if isinstance(cleaned_html, str) and cleaned_html.strip()
                else "Mind2Web DOM observation unavailable"
            )
            yield NormalizedGUIRecord(
                source="mind2web",
                record_type="trajectory_step",
                split=split,
                episode_id=episode_id,
                step_index=step_index,
                instruction=instruction,
                text_observation=observation,
                action=action,
                source_revision=source_revision,
            )


__all__ = ["iter_mind2web"]
