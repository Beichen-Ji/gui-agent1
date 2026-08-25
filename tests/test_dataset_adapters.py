import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from gui_agent.agent.types import ClickAction, HotkeyAction, TypeTextAction
from gui_agent.datasets.mind2web import iter_mind2web
from gui_agent.datasets.pipeline import AdapterReport, write_dataset
from gui_agent.datasets.schema import NormalizedGUIRecord
from gui_agent.datasets.screenagent import DatasetAdapterError, iter_screenagent
from gui_agent.datasets.webarena import iter_webarena
from scripts import prepare_gui_datasets

FIXTURES = Path(__file__).parent / "fixtures" / "gui_datasets"


def test_screenagent_maps_actions_and_sorts_source_files() -> None:
    report = AdapterReport()
    records = list(
        iter_screenagent(FIXTURES / "screenagent", split="train", report=report)
    )

    assert [record.step_index for record in records] == [0, 1, 2]
    assert records[0].action == TypeTextAction(text="weather")
    assert records[1].action == ClickAction(x=410, y=220, clicks=2)
    assert records[1].image_path == "train/session-b/images/002.jpg"
    assert records[2].action == HotkeyAction(keys=("ctrl", "a"))
    assert report.records_skipped == 1
    assert "PlanAction" in report.issues[0]


def test_screenagent_reports_source_path_for_malformed_action() -> None:
    root = FIXTURES / "screenagent-invalid"

    with pytest.raises(DatasetAdapterError, match=r"broken.*001_translate\.json"):
        list(iter_screenagent(root, split="train"))


def test_mind2web_maps_click_coordinates_and_text_actions() -> None:
    rows = cast(
        list[dict[str, object]],
        json.loads((FIXTURES / "mind2web" / "train.json").read_text(encoding="utf-8")),
    )

    report = AdapterReport()
    records = list(
        iter_mind2web(
            rows,
            split="train",
            source_revision="fixture-v1",
            report=report,
        )
    )

    assert [record.action for record in records] == [
        ClickAction(x=30, y=30),
        TypeTextAction(text="Boston"),
    ]
    assert records[0].episode_id == "mind-session-1"
    assert records[0].source_revision == "fixture-v1"
    assert report.records_skipped == 1
    assert "SELECT" in report.issues[0]


def test_webarena_sorts_configs_and_preserves_success_criteria() -> None:
    records = list(
        iter_webarena(FIXTURES / "webarena", source_revision="fixture-v1")
    )

    assert [record.episode_id for record in records] == ["2", "10"]
    assert records[0].record_type == "task"
    assert records[0].action is None
    assert records[0].success_criteria == (
        '{"eval_types":["string_match"],'
        '"reference_answers":{"must_include":["blue shirt"]}}'
    )


def test_writer_applies_limit_after_deterministic_sort_and_reports_skips(
    tmp_path: Path,
) -> None:
    source_records = list(
        iter_webarena(FIXTURES / "webarena", source_revision="fixture-v1")
    )
    records = [source_records[1], source_records[0]]

    manifest = write_dataset(records, tmp_path / "first", limit=1)
    repeated = write_dataset(reversed(records), tmp_path / "second", limit=1)
    first_bytes = (tmp_path / "first" / "records.jsonl").read_bytes()
    second_bytes = (tmp_path / "second" / "records.jsonl").read_bytes()

    assert first_bytes == second_bytes
    assert json.loads(first_bytes)["episode_id"] == "2"
    assert manifest.records_written == 1
    assert manifest.records_skipped == 1
    assert manifest.sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert repeated == manifest


def test_writer_rejects_mixed_dataset_sources(tmp_path: Path) -> None:
    webarena = next(iter_webarena(FIXTURES / "webarena"))
    screenagent = next(
        iter_screenagent(
            FIXTURES / "screenagent",
            split="train",
            report=AdapterReport(),
        )
    )

    with pytest.raises(ValueError, match="one source"):
        write_dataset([webarena, screenagent], tmp_path / "mixed")


def test_dataset_cli_help_lists_all_sources(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as captured:
        prepare_gui_datasets.main(["--help"])

    assert captured.value.code == 0
    output = capsys.readouterr().out
    assert "screenagent" in output
    assert "mind2web" in output
    assert "webarena" in output


def test_dataset_cli_processes_fixture_without_network(tmp_path: Path) -> None:
    output = tmp_path / "processed"

    result = prepare_gui_datasets.main(
        [
            "webarena",
            "--input",
            str(FIXTURES / "webarena"),
            "--output",
            str(output),
            "--revision",
            "fixture-v1",
            "--limit",
            "1",
        ]
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert result == 0
    assert manifest["records_written"] == 1
    assert manifest["records_skipped"] == 1


def test_dataset_cli_does_not_exhaust_stream_beyond_limit_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = list(iter_webarena(FIXTURES / "webarena", source_revision="fixture-v1"))

    def guarded_records(*_args: object, **_kwargs: object) -> object:
        yield records[0]
        yield records[1]
        raise AssertionError("CLI consumed beyond limit + 1")

    monkeypatch.setattr(prepare_gui_datasets, "iter_webarena", guarded_records)

    result = prepare_gui_datasets.main(
        [
            "webarena",
            "--input",
            str(FIXTURES / "webarena"),
            "--output",
            str(tmp_path / "bounded"),
            "--revision",
            "fixture-v1",
            "--limit",
            "1",
        ]
    )

    assert result == 0


def test_writer_requires_at_least_one_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        write_dataset(cast(list[NormalizedGUIRecord], []), tmp_path / "empty")
