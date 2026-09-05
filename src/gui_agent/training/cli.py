import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from gui_agent.agent.prompts import PROMPT_PROFILES
from gui_agent.datasets.schema import DatasetSource, NormalizedGUIRecord
from gui_agent.training.config import load_training_config
from gui_agent.training.dataset import build_training_split, write_training_split
from gui_agent.training.evaluation import run_evaluation
from gui_agent.training.trainer import run_training

_SOURCES = frozenset({"screenagent", "mind2web", "webarena"})


def _validation_ratio(value: str) -> float:
    converted = float(value)
    if not 0.0 < converted < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return converted


def _non_negative_integer(value: str) -> int:
    converted = int(value)
    if converted < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run Week 5 GUI training data")
    commands = parser.add_subparsers(dest="training_command", required=True)
    build = commands.add_parser("build", help="Build deterministic train/validation JSONL")
    build.add_argument("--input", action="append", required=True, metavar="SOURCE=PATH")
    build.add_argument("--image-root", action="append", required=True, metavar="SOURCE=PATH")
    build.add_argument("--validation-ratio", type=_validation_ratio, required=True)
    build.add_argument("--seed", type=_non_negative_integer, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Evaluate a frozen Week 5 condition")
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--adapter", type=Path)
    evaluate.add_argument("--prompt-profile", choices=tuple(PROMPT_PROFILES), required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    for command_name in ("check", "train"):
        command = commands.add_parser(
            command_name,
            help=f"{'Check' if command_name == 'check' else 'Train'} a QLoRA adapter",
        )
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if command_name == "train":
            command.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def _source_paths(values: list[str], *, label: str) -> dict[DatasetSource, Path]:
    parsed: dict[DatasetSource, Path] = {}
    for value in values:
        source_text, separator, path_text = value.partition("=")
        if not separator or not path_text:
            raise ValueError(f"{label} must use source=path")
        if source_text not in _SOURCES:
            raise ValueError(f"unknown source: {source_text}")
        source = cast(DatasetSource, source_text)
        if source in parsed:
            raise ValueError(f"duplicate source in {label}: {source}")
        parsed[source] = Path(path_text)
    return parsed


def _load_records(
    inputs: dict[DatasetSource, Path],
) -> tuple[list[NormalizedGUIRecord], dict[DatasetSource, str]]:
    records: list[NormalizedGUIRecord] = []
    hashes: dict[DatasetSource, str] = {}
    for source, path in sorted(inputs.items()):
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError(f"could not read input for {source}: {path}") from error
        hashes[source] = hashlib.sha256(raw).hexdigest()
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = NormalizedGUIRecord.model_validate_json(line)
            except (ValidationError, ValueError) as error:
                raise ValueError(f"invalid {source} record at line {line_number}") from error
            if record.source != source:
                raise ValueError(
                    f"input source mismatch at {path}:{line_number}: {record.source}"
                )
            records.append(record)
    return records, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.training_command == "evaluate":
        try:
            evaluation = run_evaluation(
                cases_path=cast(Path, args.cases),
                model_name=cast(str, args.model),
                adapter_path=cast(Path | None, args.adapter),
                prompt_profile=cast(str, args.prompt_profile),
                output_path=cast(Path, args.output),
                overwrite=cast(bool, args.overwrite),
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if args.training_command in {"check", "train"}:
        try:
            config = load_training_config(cast(Path, args.config))
            resume = cast(Path | None, getattr(args, "resume_from_checkpoint", None))
            run_manifest = run_training(
                config,
                cast(Path, args.data),
                cast(Path, args.output),
                check_only=args.training_command == "check",
                resume_from_checkpoint=resume,
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                run_manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    try:
        inputs = _source_paths(args.input, label="--input")
        image_roots = _source_paths(args.image_root, label="--image-root")
        missing_roots = set(inputs) - set(image_roots)
        extra_roots = set(image_roots) - set(inputs)
        if missing_roots:
            raise ValueError(f"missing image root for: {', '.join(sorted(missing_roots))}")
        if extra_roots:
            raise ValueError(f"image root has no input: {', '.join(sorted(extra_roots))}")
        for source, root in image_roots.items():
            if not root.is_dir():
                raise ValueError(f"image root for {source} is not a directory: {root}")
        records, hashes = _load_records(inputs)
        split = build_training_split(
            records,
            image_roots=image_roots,
            validation_ratio=cast(float, args.validation_ratio),
            seed=cast(int, args.seed),
            input_sha256=hashes,
        )
        build_manifest = write_training_split(
            split,
            cast(Path, args.output),
            overwrite=cast(bool, args.overwrite),
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(build_manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


__all__ = ["build_parser", "main"]
