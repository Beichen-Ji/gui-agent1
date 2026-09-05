import hashlib
from pathlib import Path

from PIL import Image

from gui_agent.agent.types import ClickAction
from gui_agent.datasets.schema import DatasetSource, NormalizedGUIRecord


def _record(
    source: DatasetSource,
    episode_id: str,
    image_path: str | None,
    *,
    action: ClickAction | None = None,
) -> NormalizedGUIRecord:
    values = {
        "source": source,
        "record_type": "trajectory_step",
        "split": "train",
        "episode_id": episode_id,
        "step_index": 0,
        "instruction": f"Complete {episode_id}",
        "image_path": image_path,
        "text_observation": "A visible control",
        "action": action or ClickAction(x=10, y=10),
        "source_revision": "fixture-v1",
    }
    return NormalizedGUIRecord.model_validate(values)


def _image(path: Path, *, size: tuple[int, int] = (100, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def test_split_is_episode_safe_source_stratified_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    from gui_agent.training.dataset import build_training_split, write_training_split

    roots: dict[DatasetSource, Path] = {
        "screenagent": tmp_path / "screen-images",
        "mind2web": tmp_path / "mind-images",
    }
    records: list[NormalizedGUIRecord] = []
    for source in ("screenagent", "mind2web"):
        for episode in ("episode-a", "episode-b", "episode-c"):
            relative = f"{episode}.png"
            _image(roots[source] / relative)
            records.append(_record(source, episode, relative))
            records.append(
                NormalizedGUIRecord.model_validate(
                    {
                        **records[-1].model_dump(mode="python"),
                        "step_index": 1,
                        "action": ClickAction(x=20, y=20),
                    }
                )
            )

    first = build_training_split(
        records,
        image_roots=roots,
        validation_ratio=0.34,
        seed=20260904,
    )
    second = build_training_split(
        reversed(records),
        image_roots=roots,
        validation_ratio=0.34,
        seed=20260904,
    )
    first_manifest = write_training_split(first, tmp_path / "first")
    second_manifest = write_training_split(second, tmp_path / "second")

    train_groups = {(item.source, item.episode_id) for item in first.train}
    validation_groups = {(item.source, item.episode_id) for item in first.validation}
    assert train_groups.isdisjoint(validation_groups)
    assert {item.source for item in first.validation} == {"screenagent", "mind2web"}
    for filename in ("train.jsonl", "validation.jsonl", "manifest.json"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
    assert first_manifest == second_manifest


def test_builder_reports_every_rejected_sample_reason(tmp_path: Path) -> None:
    from gui_agent.training.dataset import build_training_split

    root = tmp_path / "images"
    _image(root / "valid.png")
    _image(root / "bounds.png", size=(40, 30))
    (root / "corrupt.png").write_text("not an image", encoding="utf-8")

    valid = _record("screenagent", "valid", "valid.png")
    missing_path = _record("screenagent", "missing-path", None)
    missing_file = _record("screenagent", "missing-file", "absent.png")
    corrupt = _record("screenagent", "corrupt", "corrupt.png")
    out_of_bounds = _record(
        "screenagent",
        "out-of-bounds",
        "bounds.png",
        action=ClickAction(x=40, y=10),
    )
    no_action = NormalizedGUIRecord.model_construct(
        **{
            **valid.model_dump(mode="python"),
            "episode_id": "no-action",
            "action": None,
        }
    )
    webarena_task = NormalizedGUIRecord(
        source="webarena",
        record_type="task",
        split="test",
        episode_id="web-task",
        step_index=0,
        instruction="Find a product",
        success_criteria="Product is visible",
        source_revision="fixture-v1",
    )

    split = build_training_split(
        [
            valid,
            missing_path,
            missing_file,
            corrupt,
            out_of_bounds,
            no_action,
            webarena_task,
        ],
        image_roots={"screenagent": root, "webarena": root},
        validation_ratio=0.2,
        seed=7,
    )

    assert len(split.train) == 1
    assert split.validation == ()
    assert split.skip_reasons == {
        "coordinate_out_of_bounds": 1,
        "image_not_found": 1,
        "invalid_image": 1,
        "missing_action": 1,
        "missing_image_path": 1,
        "unsupported_record_type": 1,
    }
    assert split.records_seen == 7
    assert split.records_skipped == 6


def test_writer_records_provenance_licenses_and_output_hashes(tmp_path: Path) -> None:
    from gui_agent.training.dataset import build_training_split, write_training_split

    root = tmp_path / "images"
    _image(root / "step.png")
    split = build_training_split(
        [_record("screenagent", "episode", "step.png")],
        image_roots={"screenagent": root},
        validation_ratio=0.2,
        seed=7,
    )

    manifest = write_training_split(split, tmp_path / "output")

    train_bytes = (tmp_path / "output" / "train.jsonl").read_bytes()
    assert manifest.output_sha256["train.jsonl"] == hashlib.sha256(train_bytes).hexdigest()
    assert manifest.source_revisions == {"screenagent": ("fixture-v1",)}
    assert "Apache-2.0" in manifest.source_licenses["screenagent"]
    assert manifest.input_sha256["screenagent"]
