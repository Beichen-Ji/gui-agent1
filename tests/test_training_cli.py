import json
from pathlib import Path

import pytest
from PIL import Image

from gui_agent import cli

FIXTURES = Path(__file__).parent / "fixtures" / "training"


def _image_root(tmp_path: Path) -> Path:
    root = tmp_path / "images-root"
    for name in ("episode-a.png", "episode-b.png"):
        path = root / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 60), "white").save(path)
    return root


def _build_args(tmp_path: Path, output: Path) -> list[str]:
    root = _image_root(tmp_path)
    return [
        "training",
        "build",
        "--input",
        f"screenagent={FIXTURES / 'screenagent-records.jsonl'}",
        "--image-root",
        f"screenagent={root}",
        "--validation-ratio",
        "0.5",
        "--seed",
        "20260904",
        "--output",
        str(output),
    ]


def test_training_build_cli_writes_split_and_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "week5"

    result = cli.main(_build_args(tmp_path, output))

    printed = json.loads(capsys.readouterr().out)
    assert result == 0
    assert printed["kind"] == "gui-agent-week5-training"
    assert (output / "train.jsonl").is_file()
    assert (output / "validation.jsonl").is_file()
    assert (output / "manifest.json").is_file()


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (
            ["--input", "unknown=records.jsonl", "--image-root", "unknown=images"],
            "unknown source",
        ),
        (
            ["--input", "screenagent=duplicate.jsonl"],
            "duplicate source",
        ),
    ],
)
def test_training_build_cli_rejects_unknown_or_duplicate_sources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
) -> None:
    args = _build_args(tmp_path, tmp_path / "week5") + extra_args

    with pytest.raises(SystemExit) as captured:
        cli.main(args)

    assert captured.value.code == 2
    assert message in capsys.readouterr().err


def test_training_build_cli_requires_an_image_root_for_every_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _build_args(tmp_path, tmp_path / "week5")
    image_root_index = args.index("--image-root")
    del args[image_root_index : image_root_index + 2]

    with pytest.raises(SystemExit) as captured:
        cli.main(args)

    assert captured.value.code == 2
    assert "required: --image-root" in capsys.readouterr().err


def test_training_build_cli_refuses_unsafe_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")
    args = _build_args(tmp_path, output)

    with pytest.raises(SystemExit):
        cli.main([*args, "--overwrite"])

    assert sentinel.read_text(encoding="utf-8") == "user data"


def test_training_build_cli_overwrites_only_its_own_valid_output(tmp_path: Path) -> None:
    output = tmp_path / "week5"
    args = _build_args(tmp_path, output)
    assert cli.main(args) == 0

    with pytest.raises(SystemExit):
        cli.main(args)

    assert cli.main([*args, "--overwrite"]) == 0


def test_training_build_cli_refuses_overwrite_when_generated_files_were_modified(
    tmp_path: Path,
) -> None:
    output = tmp_path / "week5"
    args = _build_args(tmp_path, output)
    assert cli.main(args) == 0
    train_path = output / "train.jsonl"
    train_path.write_text("modified\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        cli.main([*args, "--overwrite"])

    assert train_path.read_text(encoding="utf-8") == "modified\n"
