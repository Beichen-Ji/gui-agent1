"""Emit ordered, redacted lifecycle events for GUI agent runs."""

import hashlib
import json
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, TextIO

from gui_agent.agent.types import AgentAction, Observation, TypeTextAction
from gui_agent.perception.text import normalize_text

EventKind = Literal[
    "run_started",
    "plan_created",
    "step_started",
    "observation_completed",
    "action_proposed",
    "action_authorized",
    "action_executed",
    "verification_completed",
    "retry_scheduled",
    "plan_revised",
    "run_finished",
]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Store one immutable, UTC-stamped event with a read-only payload."""

    sequence: int
    timestamp: datetime
    run_id: str
    kind: EventKind
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        """Validate event identity and freeze a defensive payload copy."""
        if isinstance(self.sequence, bool) or self.sequence < 1:
            raise ValueError("event sequence must be a positive integer")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("event timestamp must be UTC")
        if not self.run_id.strip():
            raise ValueError("event run_id must not be blank")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible event representation."""
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
            "run_id": self.run_id,
            "kind": self.kind,
            "payload": dict(self.payload),
        }


class EventSink(Protocol):
    """Consume structured agent events."""

    def emit(self, event: AgentEvent) -> None:
        """Consume one event without modifying it."""
        ...


class NullEventSink:
    """Discard events when persistence and console logging are disabled."""

    def emit(self, event: AgentEvent) -> None:
        """Discard one event."""
        del event


class CompositeEventSink:
    """Fan out events while isolating failures in individual sinks."""

    def __init__(self, sinks: Iterable[EventSink]) -> None:
        """Freeze the ordered sink collection used for fan-out."""
        self._sinks = tuple(sinks)

    def emit(self, event: AgentEvent) -> None:
        """Send an event to every healthy sink in order."""
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                continue


class JSONLEventSink:
    """Append events as UTF-8 JSON Lines records."""

    def __init__(self, path: Path) -> None:
        """Create parent directories for the caller-selected log path."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: AgentEvent) -> None:
        """Append one serialized event to the log."""
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


class ConsoleEventSink:
    """Print compact event metadata without exposing typed text or goals."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        log_level: str = "INFO",
    ) -> None:
        """Configure the destination stream and validated log level."""
        normalized = log_level.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("unsupported log level")
        self._stream = stream or sys.stderr
        self._log_level = normalized

    def emit(self, event: AgentEvent) -> None:
        """Print one compact event line and flush the stream."""
        payload = " ".join(f"{key}={value}" for key, value in event.payload.items())
        print(
            f"[{event.sequence:04d}] {event.kind} {payload}".rstrip(),
            file=self._stream,
            flush=True,
        )


class EventEmitter:
    """Assign a run identity, UTC time, and monotonic sequence to events."""

    def __init__(
        self,
        sink: EventSink,
        *,
        run_id: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Configure a sink and injectable identity and time providers."""
        self._sink = sink
        self.run_id = run_id or uuid.uuid4().hex
        if not self.run_id.strip():
            raise ValueError("run_id must not be blank")
        self._clock = clock
        self._sequence = 0

    def emit(self, kind: EventKind, payload: Mapping[str, object]) -> AgentEvent:
        """Create, forward, and return the next event in this run."""
        self._sequence += 1
        event = AgentEvent(
            sequence=self._sequence,
            timestamp=self._clock(),
            run_id=self.run_id,
            kind=kind,
            payload=payload,
        )
        self._sink.emit(event)
        return event


def goal_metadata(goal: str) -> dict[str, object]:
    """Represent a goal only by its SHA-256 digest."""
    return {"goal_sha256": hashlib.sha256(goal.encode("utf-8")).hexdigest()}


def action_metadata(action: AgentAction) -> dict[str, object]:
    """Return redacted action metadata that never includes typed text."""
    if isinstance(action, TypeTextAction):
        return {"action_kind": action.kind, "text_length": len(action.text)}
    return {"action_kind": action.kind}


def observation_metadata(observation: Observation) -> dict[str, object]:
    """Return geometry, counts, and a digest instead of raw OCR text."""
    normalized = "\n".join(
        normalize_text(detection.text) for detection in observation.detections
    )
    screenshot = observation.screenshot
    return {
        "step_index": observation.step_index,
        "screen_width": screenshot.width,
        "screen_height": screenshot.height,
        "screen_origin_x": screenshot.origin.x,
        "screen_origin_y": screenshot.origin.y,
        "ocr_count": len(observation.detections),
        "ocr_summary_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


__all__ = [
    "AgentEvent",
    "CompositeEventSink",
    "ConsoleEventSink",
    "EventEmitter",
    "EventKind",
    "EventSink",
    "JSONLEventSink",
    "NullEventSink",
    "action_metadata",
    "goal_metadata",
    "observation_metadata",
]
