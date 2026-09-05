import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from gui_agent.agent.types import ClickAction, DragAction, ScrollAction
from gui_agent.datasets.schema import DatasetSource, NormalizedGUIRecord
from gui_agent.training.schema import (
    SourceSplitCounts,
    TrainingExample,
    TrainingManifest,
    TrainingSplit,
)

_SOURCE_LICENSES: dict[DatasetSource, str] = {
    "screenagent": "Apache-2.0 (dataset); MIT (code)",
    "mind2web": "Creative Commons Attribution 4.0 International",
    "webarena": "Apache-2.0",
}
_OUTPUT_FILES = frozenset({"train.jsonl", "validation.jsonl", "manifest.json"})


def _canonical_record(record: NormalizedGUIRecord) -> bytes:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _example_sort_key(example: TrainingExample) -> tuple[str, str, str]:
    return (example.source, example.episode_id.casefold(), example.sample_id)


def _valid_action_coordinates(record: NormalizedGUIRecord, width: int, height: int) -> bool:
    action = record.action
    points: tuple[tuple[int, int], ...]
    if isinstance(action, ClickAction):
        points = ((action.x, action.y),)
    elif isinstance(action, DragAction):
        points = ((action.start_x, action.start_y), (action.end_x, action.end_y))
    elif isinstance(action, ScrollAction):
        if action.x is None and action.y is None:
            points = ()
        elif action.x is None or action.y is None:
            return False
        else:
            points = ((action.x, action.y),)
    else:
        points = ()
    return all(0 <= x < width and 0 <= y < height for x, y in points)


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError):
        return None


def _episode_rank(seed: int, source: DatasetSource, episode_id: str) -> str:
    value = f"{seed}:{source}:{episode_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _source_hashes(
    records_by_source: Mapping[DatasetSource, list[NormalizedGUIRecord]],
) -> dict[DatasetSource, str]:
    return {
        source: hashlib.sha256(
            b"".join(sorted(_canonical_record(record) for record in records))
        ).hexdigest()
        for source, records in sorted(records_by_source.items())
    }


def build_training_split(
    records: Iterable[NormalizedGUIRecord],
    *,
    image_roots: Mapping[DatasetSource, Path],
    validation_ratio: float,
    seed: int,
    input_sha256: Mapping[DatasetSource, str] | None = None,
) -> TrainingSplit:
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    materialized = list(records)
    by_source: dict[DatasetSource, list[NormalizedGUIRecord]] = defaultdict(list)
    seen_by_source: Counter[DatasetSource] = Counter()
    skipped_by_source: Counter[DatasetSource] = Counter()
    skip_reasons: Counter[str] = Counter()
    source_revisions: dict[DatasetSource, set[str]] = defaultdict(set)
    accepted: list[TrainingExample] = []

    def skip(source: DatasetSource, reason: str) -> None:
        skipped_by_source[source] += 1
        skip_reasons[reason] += 1

    for record in materialized:
        by_source[record.source].append(record)
        seen_by_source[record.source] += 1
        source_revisions[record.source].add(record.source_revision)
        if record.record_type != "trajectory_step":
            skip(record.source, "unsupported_record_type")
            continue
        if record.action is None:
            skip(record.source, "missing_action")
            continue
        if record.image_path is None:
            skip(record.source, "missing_image_path")
            continue
        root = image_roots.get(record.source)
        if root is None:
            raise ValueError(f"missing image root for source: {record.source}")
        resolved_root = root.resolve()
        image_path = (resolved_root / record.image_path).resolve()
        if not image_path.is_relative_to(resolved_root):
            skip(record.source, "image_outside_root")
            continue
        if not image_path.is_file():
            skip(record.source, "image_not_found")
            continue
        size = _image_size(image_path)
        if size is None:
            skip(record.source, "invalid_image")
            continue
        if not _valid_action_coordinates(record, *size):
            skip(record.source, "coordinate_out_of_bounds")
            continue
        sample_digest = hashlib.sha256(_canonical_record(record)).hexdigest()[:24]
        accepted.append(
            TrainingExample(
                sample_id=f"{record.source}:{sample_digest}",
                source=record.source,
                episode_id=record.episode_id,
                instruction=record.instruction,
                image_path=image_path.as_posix(),
                text_observation=record.text_observation,
                target_action=record.action,
                source_revision=record.source_revision,
            )
        )

    if not accepted:
        raise ValueError("no valid training examples were found")

    accepted_groups: dict[DatasetSource, dict[str, list[TrainingExample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in accepted:
        accepted_groups[example.source][example.episode_id].append(example)

    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    for source in sorted(accepted_groups):
        groups = accepted_groups[source]
        ranked_episodes = sorted(
            groups,
            key=lambda episode: (_episode_rank(seed, source, episode), episode.casefold()),
        )
        if len(ranked_episodes) < 2:
            validation_count = 0
        else:
            validation_count = max(1, int(len(ranked_episodes) * validation_ratio + 0.5))
            validation_count = min(validation_count, len(ranked_episodes) - 1)
        validation_episodes = set(ranked_episodes[:validation_count])
        for episode, examples in groups.items():
            target = validation if episode in validation_episodes else train
            target.extend(examples)

    train.sort(key=_example_sort_key)
    validation.sort(key=_example_sort_key)
    source_counts: dict[DatasetSource, SourceSplitCounts] = {}
    all_sources = sorted(seen_by_source)
    for source in all_sources:
        source_counts[source] = SourceSplitCounts(
            seen=seen_by_source[source],
            accepted=sum(item.source == source for item in accepted),
            skipped=skipped_by_source[source],
            train=sum(item.source == source for item in train),
            validation=sum(item.source == source for item in validation),
        )

    hashes = dict(input_sha256) if input_sha256 is not None else _source_hashes(by_source)
    return TrainingSplit(
        train=tuple(train),
        validation=tuple(validation),
        seed=seed,
        validation_ratio=validation_ratio,
        records_seen=len(materialized),
        records_skipped=sum(skip_reasons.values()),
        source_counts=source_counts,
        skip_reasons=dict(sorted(skip_reasons.items())),
        input_sha256=hashes,
        source_revisions={
            source: tuple(sorted(revisions))
            for source, revisions in sorted(source_revisions.items())
        },
        source_licenses={source: _SOURCE_LICENSES[source] for source in all_sources},
    )


def _serialize_examples(examples: tuple[TrainingExample, ...]) -> bytes:
    return b"".join(
        (
            json.dumps(
                example.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for example in examples
    )


def _validate_overwrite_target(output_dir: Path) -> None:
    entries = {path.name for path in output_dir.iterdir()}
    if entries != _OUTPUT_FILES:
        raise ValueError("overwrite requires an intact Week 5 training output directory")
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = TrainingManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        for filename in ("train.jsonl", "validation.jsonl"):
            actual = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
            if manifest.output_sha256.get(filename) != actual:
                raise ValueError(f"generated file hash mismatch: {filename}")
    except (OSError, ValueError) as error:
        raise ValueError("overwrite requires a valid Week 5 manifest") from error


def write_training_split(
    split: TrainingSplit,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> TrainingManifest:
    if output_dir.exists():
        if not overwrite:
            raise ValueError("output directory already exists; pass --overwrite to rebuild")
        _validate_overwrite_target(output_dir)
    else:
        output_dir.mkdir(parents=True)

    train_bytes = _serialize_examples(split.train)
    validation_bytes = _serialize_examples(split.validation)
    output_hashes = {
        "train.jsonl": hashlib.sha256(train_bytes).hexdigest(),
        "validation.jsonl": hashlib.sha256(validation_bytes).hexdigest(),
    }
    manifest = TrainingManifest(
        seed=split.seed,
        validation_ratio=split.validation_ratio,
        records_seen=split.records_seen,
        records_accepted=len(split.train) + len(split.validation),
        records_skipped=split.records_skipped,
        train_examples=len(split.train),
        validation_examples=len(split.validation),
        source_counts=split.source_counts,
        skip_reasons=split.skip_reasons,
        input_sha256=split.input_sha256,
        source_revisions=split.source_revisions,
        source_licenses=split.source_licenses,
        output_sha256=output_hashes,
    )
    (output_dir / "train.jsonl").write_bytes(train_bytes)
    (output_dir / "validation.jsonl").write_bytes(validation_bytes)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["build_training_split", "write_training_split"]
