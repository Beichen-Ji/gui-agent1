"""Sort, serialize, and record provenance for normalized dataset records."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from gui_agent.datasets.schema import DatasetManifest, NormalizedGUIRecord
from gui_agent.datasets.sources import DATASET_SOURCES


@dataclass(slots=True)
class AdapterReport:
    """Count skipped records while bounding retained issue details."""

    records_skipped: int = 0
    issues: list[str] = field(default_factory=list)
    issue_limit: int = 20

    def skip(self, issue: str) -> None:
        """Record one skip and retain its message only within the issue limit."""
        self.records_skipped += 1
        if len(self.issues) < self.issue_limit:
            self.issues.append(issue)

    @property
    def suppressed_issue_count(self) -> int:
        """Return the number of issue messages omitted from the report."""
        return max(0, self.records_skipped - len(self.issues))


def _episode_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdecimal() else (1, value.casefold())


def _record_sort_key(record: NormalizedGUIRecord) -> tuple[object, ...]:
    return (
        record.source,
        record.split.casefold(),
        _episode_sort_key(record.episode_id),
        record.step_index,
        record.instruction.casefold(),
    )


def write_dataset(
    records: Iterable[NormalizedGUIRecord],
    output: Path,
    *,
    limit: int | None = None,
    records_skipped: int = 0,
) -> DatasetManifest:
    """Write one source and revision deterministically with a provenance manifest."""
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    if isinstance(records_skipped, bool) or records_skipped < 0:
        raise ValueError("records_skipped must be a non-negative integer")
    ordered = sorted(records, key=_record_sort_key)
    if not ordered:
        raise ValueError("at least one normalized record is required")
    sources = {record.source for record in ordered}
    revisions = {record.source_revision for record in ordered}
    if len(sources) != 1 or len(revisions) != 1:
        raise ValueError("records must come from one source and one revision")

    selected = ordered if limit is None else ordered[:limit]
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    serialized = "".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in selected
    ).encode("utf-8")
    records_path.write_bytes(serialized)

    source = selected[0].source
    source_metadata = DATASET_SOURCES[source]
    manifest = DatasetManifest(
        source=source,
        source_url=source_metadata.url,
        source_revision=selected[0].source_revision,
        license=source_metadata.license,
        records_written=len(selected),
        records_skipped=records_skipped + len(ordered) - len(selected),
        output_file=records_path.name,
        sha256=hashlib.sha256(serialized).hexdigest(),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["AdapterReport", "write_dataset"]
