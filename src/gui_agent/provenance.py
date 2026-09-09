import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class AdapterLayout:
    output_root: Path
    adapter_dir: Path


def file_sha256(path: Path, *, label: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"could not read {label}: {path}") from error


def read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return cast(dict[str, object], value)


def resolve_adapter_layout(adapter_path: Path) -> AdapterLayout:
    resolved = adapter_path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"adapter path is not a directory: {adapter_path}")
    if (resolved / "run-manifest.json").is_file():
        return AdapterLayout(output_root=resolved, adapter_dir=resolved / "adapter")
    if resolved.name == "adapter" and (resolved.parent / "run-manifest.json").is_file():
        return AdapterLayout(output_root=resolved.parent, adapter_dir=resolved)
    raise ValueError("adapter path requires a sibling or child run-manifest.json")


def adapter_provenance(adapter_path: Path) -> tuple[str, str, str]:
    layout = resolve_adapter_layout(adapter_path)
    try:
        label = layout.output_root.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        label = layout.output_root.name
    return (
        label,
        file_sha256(
            layout.output_root / "run-manifest.json",
            label="adapter run manifest",
        ),
        file_sha256(
            layout.adapter_dir / "adapter_model.safetensors",
            label="adapter weights",
        ),
    )


def atomic_write_owned_json(
    payload: Mapping[str, object],
    path: Path,
    *,
    owned_kind: str,
    overwrite: bool = False,
    output_label: str = "evaluation output",
) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError(f"{output_label} must be a JSON file")
    if payload.get("kind") != owned_kind:
        raise ValueError(f"{output_label} has an unsupported kind")
    existed = path.exists()
    if existed:
        if not overwrite:
            raise ValueError(f"{output_label} already exists: {path}")
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"refusing to overwrite an unowned {output_label}") from error
        if not isinstance(stored, dict) or stored.get("kind") != owned_kind:
            raise ValueError(f"refusing to overwrite an unowned {output_label}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    if not existed:
        try:
            with path.open("xb") as output:
                output.write(encoded)
        except FileExistsError as error:
            raise ValueError(f"{output_label} already exists: {path}") from error
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(encoded)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


__all__ = [
    "AdapterLayout",
    "adapter_provenance",
    "atomic_write_owned_json",
    "file_sha256",
    "read_json_object",
    "resolve_adapter_layout",
]
