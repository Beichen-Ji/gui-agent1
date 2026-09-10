# Codebase Cleanup and Docstrings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate duplicated private helpers, retain every existing public interface, document the production codebase, and push a fully verified cleanup branch.

**Architecture:** Focused internal modules become the single source of truth for strict Pydantic models, CLI integer parsing, model loading, image fingerprints, normalized text, and dataset source metadata. Existing public modules keep their symbols and behavior, while package-by-package docstring work is enforced by Ruff's Google-style `D` rules.

**Tech Stack:** Python 3.11, Pydantic 2, NumPy, argparse, pytest, pytest-cov, Ruff, mypy, uv, Git worktrees.

**Spec:** `docs/superpowers/specs/2026-09-10-codebase-cleanup-docstrings-design.md`

## Global Constraints

- Preserve every existing public class, method, function, constant, import path, callable signature, CLI option, default, exit code, and persisted schema.
- Do not change safety, OCR, planning, retry, verification, training, simulation, or evaluation behavior.
- Do not fix the Week 7 terminal-focus limitation in this cleanup.
- Do not rename week-numbered files, change package version metadata, or restructure user documentation.
- Add no runtime dependencies.
- Keep `ObservationBuilder.clear_cache()` and characterize its public behavior.
- Source docstrings and comments use English; comments explain invariants and boundaries instead of narrating syntax.
- Do not commit `artifacts/`, screenshots, model weights, full OCR text, or local environments.
- Final collection must contain at least the baseline 486 tests; non-integration coverage must remain at least 88%.

---

### Task 1: Protect the Public Cache-Control Contract

**Files:**
- Modify: `tests/test_agent_observation.py`
- Modify: `docs/superpowers/plans/2026-09-10-codebase-cleanup-docstrings.md`

**Interfaces:**
- Consumes: `ObservationBuilder.observe(step_index: int) -> Observation`
- Protects: `ObservationBuilder.clear_cache() -> None`

- [x] **Step 1: Add the characterization test**

Append a test that observes an identical frame twice, verifies one OCR call, clears the cache, observes again, and verifies a second OCR call:

```python
def test_clear_cache_forces_ocr_for_an_unchanged_frame() -> None:
    screenshot = screenshot_fixture()
    capture = FakeCapture(screenshot)
    ocr = FakeOCR([])
    builder = ObservationBuilder(capture, ocr)

    builder.observe(0)
    builder.observe(1)
    assert len(ocr.calls) == 1

    builder.clear_cache()
    builder.observe(2)

    assert len(ocr.calls) == 2
```

- [x] **Step 2: Run the focused characterization test**

Run:

```powershell
uv run --no-sync pytest tests/test_agent_observation.py -q
```

Expected: all observation tests pass on the baseline implementation. This is a characterization test, so it is expected to be green before refactoring.

- [x] **Step 3: Record the public-surface baseline**

Use the immutable `origin/master` tree as the baseline and count its public definitions:

```powershell
$public = git grep -n -E "^(class|def) [A-Za-z][A-Za-z0-9_]*|^    def [A-Za-z][A-Za-z0-9_]*" `
  origin/master -- src/gui_agent
"PUBLIC_SYMBOL_BASELINE_LINES=$($public.Count)"
uv run --no-sync ruff check .
```

Observed: `PUBLIC_SYMBOL_BASELINE_LINES=332`; Ruff exits cleanly. The final audit compares directly against the same immutable Git tree instead of a generated copy.

- [x] **Step 4: Commit the compatibility guard**

```powershell
git add tests/test_agent_observation.py docs/superpowers/plans/2026-09-10-codebase-cleanup-docstrings.md
git commit -m "test: protect the public observation cache contract"
```

---

### Task 2: Consolidate Strict Frozen Pydantic Models

**Files:**
- Create: `src/gui_agent/_models.py`
- Create: `tests/test_shared_models.py`
- Modify: `src/gui_agent/agent/types.py`
- Modify: `src/gui_agent/datasets/schema.py`
- Modify: `src/gui_agent/training/schema.py`
- Modify: `src/gui_agent/training/evaluation.py`
- Modify: `src/gui_agent/evaluation/cli.py`
- Modify: `src/gui_agent/evaluation/metrics.py`
- Modify: `src/gui_agent/evaluation/report.py`
- Modify: `src/gui_agent/evaluation/suite.py`

**Interfaces:**
- Produces: `gui_agent._models.StrictFrozenModel`
- Preserves: strict input validation, forbidden extras, frozen instances, and every concrete schema class

- [x] **Step 1: Write a failing shared-base test**

Create `tests/test_shared_models.py`:

```python
import pytest
from pydantic import ValidationError

from gui_agent._models import StrictFrozenModel


class ExampleModel(StrictFrozenModel):
    value: int


def test_strict_frozen_model_preserves_the_repository_schema_policy() -> None:
    model = ExampleModel(value=1)

    with pytest.raises(ValidationError):
        ExampleModel(value="1")
    with pytest.raises(ValidationError):
        ExampleModel(value=1, extra=True)
    with pytest.raises(ValidationError):
        model.value = 2
```

- [x] **Step 2: Verify the test fails because the module does not exist**

```powershell
uv run --no-sync pytest tests/test_shared_models.py -q
```

Expected: collection fails with `ModuleNotFoundError: gui_agent._models`.

- [x] **Step 3: Add the canonical model base and migrate consumers**

Create the shared implementation:

```python
"""Shared Pydantic model policies used by persisted and runtime schemas."""

from pydantic import BaseModel, ConfigDict


class StrictFrozenModel(BaseModel):
    """Reject coercion and unknown fields while keeping validated values immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
```

Replace each private `_StrictFrozenModel` definition with an import of `StrictFrozenModel`, update concrete base classes, and remove now-unused `BaseModel`/`ConfigDict` imports. Do not change any concrete field or validator.

- [x] **Step 4: Run schema-focused tests and static checks**

```powershell
uv run --no-sync pytest tests/test_shared_models.py tests/test_agent_types.py `
  tests/test_dataset_schema.py tests/test_training_schema.py `
  tests/test_training_evaluation.py tests/test_evaluation_metrics.py `
  tests/test_evaluation_report.py tests/test_evaluation_suite.py `
  tests/test_evaluation_cli.py -q
uv run --no-sync ruff check src/gui_agent/_models.py src/gui_agent/agent/types.py `
  src/gui_agent/datasets/schema.py src/gui_agent/training/schema.py `
  src/gui_agent/training/evaluation.py src/gui_agent/evaluation tests/test_shared_models.py
uv run --no-sync mypy src tests/test_shared_models.py
```

Expected: all selected tests and both static checks pass.

- [x] **Step 5: Commit the model consolidation**

```powershell
git add src/gui_agent/_models.py src/gui_agent/agent/types.py `
  src/gui_agent/datasets/schema.py src/gui_agent/training/schema.py `
  src/gui_agent/training/evaluation.py src/gui_agent/evaluation `
  tests/test_shared_models.py
git commit -m "refactor: share the strict frozen model policy"
```

---

### Task 3: Consolidate CLI Integer Parsing Without Changing Errors

**Files:**
- Create: `src/gui_agent/cli_args.py`
- Create: `tests/test_cli_args.py`
- Modify: `src/gui_agent/cli.py`
- Modify: `src/gui_agent/datasets/cli.py`
- Modify: `src/gui_agent/training/cli.py`
- Modify: `scripts/benchmark_ocr.py`

**Interfaces:**
- Produces: `parse_integer_at_least(value: str, *, minimum: int, message: str, wrap_conversion_error: bool = False) -> int`
- Preserves: each existing private argparse callback and its exact boundary/error behavior

- [x] **Step 1: Write failing tests for the shared parser**

Create `tests/test_cli_args.py` with cases for accepted integers, below-minimum values, wrapped conversion errors, and unwrapped conversion errors:

```python
import argparse

import pytest

from gui_agent.cli_args import parse_integer_at_least


def test_parse_integer_at_least_accepts_its_boundary() -> None:
    assert parse_integer_at_least("0", minimum=0, message="bad") == 0
    assert parse_integer_at_least("1", minimum=1, message="bad") == 1


def test_parse_integer_at_least_uses_the_requested_argparse_error() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="exact message"):
        parse_integer_at_least("0", minimum=1, message="exact message")
    with pytest.raises(argparse.ArgumentTypeError, match="exact message"):
        parse_integer_at_least(
            "not-an-int",
            minimum=1,
            message="exact message",
            wrap_conversion_error=True,
        )


def test_parse_integer_at_least_can_preserve_raw_conversion_errors() -> None:
    with pytest.raises(ValueError):
        parse_integer_at_least("not-an-int", minimum=0, message="bad")
```

- [x] **Step 2: Verify the new tests fail on the missing module**

```powershell
uv run --no-sync pytest tests/test_cli_args.py -q
```

Expected: collection fails with `ModuleNotFoundError: gui_agent.cli_args`.

- [x] **Step 3: Implement the shared primitive and delegate existing callbacks**

Implement the exact conversion policy:

```python
def parse_integer_at_least(
    value: str,
    *,
    minimum: int,
    message: str,
    wrap_conversion_error: bool = False,
) -> int:
    try:
        converted = int(value)
    except ValueError as error:
        if wrap_conversion_error:
            raise argparse.ArgumentTypeError(message) from error
        raise
    if converted < minimum:
        raise argparse.ArgumentTypeError(message)
    return converted
```

Keep `_positive_integer` and `_non_negative_integer` as thin private wrappers so existing callback names and module seams remain intact. Pass the original message from each caller: root CLI uses `must be a positive integer` and wraps conversion errors; the dataset/training callbacks preserve their current raw conversion behavior; the benchmark preserves `must be positive` / `must be non-negative`.

- [x] **Step 4: Run CLI tests and compare help output**

```powershell
uv run --no-sync pytest tests/test_cli_args.py tests/test_agent_cli.py `
  tests/test_dataset_adapters.py tests/test_training_cli.py `
  tests/test_perception_benchmark.py -q
uv run --no-sync gui-agent --help *> artifacts/week8-cleanup/gui-agent-help-after.txt
uv run --no-sync ruff check src/gui_agent/cli_args.py src/gui_agent/cli.py `
  src/gui_agent/datasets/cli.py src/gui_agent/training/cli.py `
  scripts/benchmark_ocr.py tests/test_cli_args.py
uv run --no-sync mypy src tests/test_cli_args.py scripts/benchmark_ocr.py
```

Expected: all tests and static checks pass; no option names/defaults disappear from CLI help.

- [x] **Step 5: Commit the CLI consolidation**

```powershell
git add src/gui_agent/cli_args.py src/gui_agent/cli.py `
  src/gui_agent/datasets/cli.py src/gui_agent/training/cli.py `
  scripts/benchmark_ocr.py tests/test_cli_args.py
git commit -m "refactor: share CLI integer validation"
```

---

### Task 4: Consolidate Lazy Model Loading

**Files:**
- Create: `src/gui_agent/model_loading.py`
- Create: `tests/test_model_loading.py`
- Modify: `src/gui_agent/agent/qwen.py`
- Modify: `src/gui_agent/training/lora.py`

**Interfaces:**
- Produces: `load_processor(model_name: str, **kwargs: object) -> object`
- Produces: `load_multimodal_model(model_name: str, **kwargs: object) -> object`
- Preserves: lazy optional imports and both existing dependency-injection defaults

- [x] **Step 1: Write failing loader tests**

Use a fake `transformers` module to verify forwarded model names and keyword arguments without loading a real model:

```python
def test_shared_model_loaders_forward_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = LoaderProbe("processor")
    model = LoaderProbe("model")
    fake = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=processor),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=model),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)

    assert load_processor("model-id", revision="abc") == "processor"
    assert load_multimodal_model("model-id", dtype="bf16") == "model"
    assert processor.calls == [("model-id", {"revision": "abc"})]
    assert model.calls == [("model-id", {"dtype": "bf16"})]
```

- [x] **Step 2: Verify the loader test fails because the module is missing**

```powershell
uv run --no-sync pytest tests/test_model_loading.py -q
```

Expected: collection fails with `ModuleNotFoundError: gui_agent.model_loading`.

- [x] **Step 3: Implement the shared lazy loaders and retain wrappers**

Move only the `AutoProcessor.from_pretrained` and `AutoModelForMultimodalLM.from_pretrained` logic into the new module. Keep `_default_processor_loader` and `_default_model_loader` in `agent/qwen.py` and `training/lora.py` as one-line delegates so function defaults, monkeypatch seams, and private module structure remain stable.

- [x] **Step 4: Run loader, planner, and LoRA tests**

```powershell
uv run --no-sync pytest tests/test_model_loading.py tests/test_agent_planner.py `
  tests/test_training_lora.py -q
uv run --no-sync ruff check src/gui_agent/model_loading.py `
  src/gui_agent/agent/qwen.py src/gui_agent/training/lora.py `
  tests/test_model_loading.py
uv run --no-sync mypy src tests/test_model_loading.py
```

Expected: all selected tests and static checks pass without importing real model weights.

- [x] **Step 5: Commit the model-loader consolidation**

```powershell
git add src/gui_agent/model_loading.py src/gui_agent/agent/qwen.py `
  src/gui_agent/training/lora.py tests/test_model_loading.py
git commit -m "refactor: share lazy multimodal model loaders"
```

---

### Task 5: Consolidate Text Normalization and Frame Fingerprints

**Files:**
- Create: `src/gui_agent/perception/text.py`
- Create: `src/gui_agent/perception/fingerprint.py`
- Create: `tests/test_perception_text.py`
- Create: `tests/test_perception_fingerprint.py`
- Modify: `src/gui_agent/perception/benchmark.py`
- Modify: `src/gui_agent/agent/events.py`
- Modify: `src/gui_agent/agent/verification.py`
- Modify: `src/gui_agent/agent/observation.py`

**Interfaces:**
- Produces: `normalize_text(value: str) -> str`
- Produces: `image_fingerprint(image: ImageArray) -> str`
- Preserves: `gui_agent.perception.benchmark.normalize_text` as an importable public symbol
- Preserves: OCR summary hashes, frame-change detection, and observation cache keys

- [x] **Step 1: Write failing shared-helper tests**

Test whitespace/case normalization and fingerprint sensitivity to value, shape, and dtype:

```python
def test_normalize_text_collapses_whitespace_and_casefolds() -> None:
    assert normalize_text("  Save\n  FILE  ") == "save file"


def test_image_fingerprint_tracks_shape_dtype_and_bytes() -> None:
    base = np.zeros((2, 3, 3), dtype=np.uint8)
    changed = base.copy()
    changed[0, 0, 0] = 1

    assert image_fingerprint(base) == image_fingerprint(base.copy())
    assert image_fingerprint(base) != image_fingerprint(changed)
    assert image_fingerprint(base) != image_fingerprint(base.astype(np.uint16))
    assert image_fingerprint(base) != image_fingerprint(base.reshape(3, 2, 3))
```

- [x] **Step 2: Verify both tests fail on missing modules**

```powershell
uv run --no-sync pytest tests/test_perception_text.py `
  tests/test_perception_fingerprint.py -q
```

Expected: collection fails because the new perception modules do not exist.

- [x] **Step 3: Implement and migrate normalization**

Implement `normalize_text` as exactly `" ".join(value.split()).casefold()`. Import it into `benchmark.py` without aliasing so the existing public import path still resolves. Use it from verification and event metadata without changing detection order, blank filtering, separators, or hash encoding.

- [x] **Step 4: Implement and migrate fingerprints**

Implement `image_fingerprint` using a contiguous NumPy view and the existing SHA-256 inputs in the same order: ASCII shape, ASCII dtype, then image bytes. Observation cache keys must continue to add screenshot origin, confidence, and OCR cache token outside the shared image digest.

- [x] **Step 5: Run perception and agent regression tests**

```powershell
uv run --no-sync pytest tests/test_perception_text.py `
  tests/test_perception_fingerprint.py tests/test_perception_benchmark.py `
  tests/test_agent_events.py tests/test_agent_verification.py `
  tests/test_agent_observation.py -q
uv run --no-sync ruff check src/gui_agent/perception src/gui_agent/agent `
  tests/test_perception_text.py tests/test_perception_fingerprint.py
uv run --no-sync mypy src tests/test_perception_text.py `
  tests/test_perception_fingerprint.py
```

Expected: all selected tests and static checks pass, including existing benchmark imports.

- [x] **Step 6: Commit the perception-helper consolidation**

```powershell
git add src/gui_agent/perception/text.py src/gui_agent/perception/fingerprint.py `
  src/gui_agent/perception/benchmark.py src/gui_agent/agent/events.py `
  src/gui_agent/agent/verification.py src/gui_agent/agent/observation.py `
  tests/test_perception_text.py tests/test_perception_fingerprint.py
git commit -m "refactor: share text and frame normalization helpers"
```

---

### Task 6: Centralize Dataset Source Metadata

**Files:**
- Create: `src/gui_agent/datasets/sources.py`
- Create: `tests/test_dataset_sources.py`
- Modify: `src/gui_agent/datasets/pipeline.py`
- Modify: `src/gui_agent/training/dataset.py`

**Interfaces:**
- Produces: `DatasetSourceMetadata(url: str, license: str)`
- Produces: immutable `DATASET_SOURCES: Mapping[DatasetSource, DatasetSourceMetadata]`
- Preserves: every manifest URL and license string exactly

- [ ] **Step 1: Write a failing canonical-metadata test**

```python
def test_dataset_source_metadata_preserves_urls_and_licenses() -> None:
    assert DATASET_SOURCES["screenagent"] == DatasetSourceMetadata(
        url="https://github.com/niuzaisheng/ScreenAgent",
        license="Apache-2.0 (dataset); MIT (code)",
    )
    assert DATASET_SOURCES["mind2web"].license == (
        "Creative Commons Attribution 4.0 International"
    )
    assert DATASET_SOURCES["webarena"].url == (
        "https://github.com/web-arena-x/webarena"
    )
```

- [ ] **Step 2: Verify the test fails on the missing module**

```powershell
uv run --no-sync pytest tests/test_dataset_sources.py -q
```

Expected: collection fails with `ModuleNotFoundError: gui_agent.datasets.sources`.

- [ ] **Step 3: Implement immutable metadata and migrate consumers**

Define a frozen, slotted dataclass and expose the mapping through `MappingProxyType`. Replace `_SOURCE_METADATA` and `_SOURCE_LICENSES` lookups without changing string literals or manifest field construction.

- [ ] **Step 4: Run dataset and training regressions**

```powershell
uv run --no-sync pytest tests/test_dataset_sources.py `
  tests/test_dataset_adapters.py tests/test_training_dataset.py -q
uv run --no-sync ruff check src/gui_agent/datasets src/gui_agent/training/dataset.py `
  tests/test_dataset_sources.py
uv run --no-sync mypy src tests/test_dataset_sources.py
```

Expected: all selected tests and static checks pass; existing manifest assertions remain unchanged.

- [ ] **Step 5: Commit source metadata consolidation**

```powershell
git add src/gui_agent/datasets/sources.py src/gui_agent/datasets/pipeline.py `
  src/gui_agent/training/dataset.py tests/test_dataset_sources.py
git commit -m "refactor: centralize dataset source metadata"
```

---

### Task 7: Audit Private Dead Code and Provenance Boundaries

**Files:**
- Modify: `docs/test-reports/week8-codebase-cleanup-report.md`
- Modify only confirmed private source files if the evidence below identifies a candidate

**Interfaces:**
- Consumes: all private top-level definitions in `src/gui_agent`
- Preserves: callbacks, protocol methods, serializer hooks, dependency-injection seams, and Qwen-specific adapter validation

- [ ] **Step 1: Run the private-definition reference audit**

Run the AST/text audit from the repository root. It lists only private top-level definitions whose name occurs no more than once across source, tests, examples, and scripts:

```powershell
uv run --no-sync python -c "import ast,pathlib,re; root=pathlib.Path('src/gui_agent'); files=[*root.rglob('*.py'),*pathlib.Path('tests').rglob('*.py'),*pathlib.Path('examples').rglob('*.py'),*pathlib.Path('scripts').rglob('*.py')]; texts='\n'.join(p.read_text(encoding='utf-8') for p in files); out=[]; [(lambda tree,p:[out.append((str(p),n.lineno,n.name,len(re.findall(r'(?<![A-Za-z0-9_])'+re.escape(n.name)+r'(?![A-Za-z0-9_])',texts)))) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name.startswith('_') and not n.name.startswith('__')])(ast.parse(p.read_text(encoding='utf-8')),p) for p in root.rglob('*.py')]; print('\n'.join(f'{p}:{line}:{name}:refs={count}' for p,line,name,count in out if count<=1))" `
  | Set-Content -Encoding utf8 artifacts/week8-cleanup/private-reference-audit.txt
```

Expected on the approved baseline: no candidates. If later consolidation leaves a private wrapper with no consumer, delete that wrapper only after `rg -w <name>` confirms no dynamic/test seam.

- [ ] **Step 2: Run compiler/linter dead-code checks**

```powershell
uv run --no-sync ruff check . --select F,B,SIM
uv run --no-sync python -m compileall -q src tests examples scripts
```

Expected: no unused imports, unused local variables, undefined names, or simplification errors.

- [ ] **Step 3: Audit provenance rather than moving domain validation**

Confirm `resolve_adapter_layout`, `read_json_object`, `file_sha256`, and `adapter_provenance` remain centralized in `provenance.py`. Keep model/grid/prompt/hash validation in `agent/qwen.py` because it is Qwen runtime policy, not duplicate generic provenance behavior.

- [ ] **Step 4: Write the cleanup evidence report**

Create `docs/test-reports/week8-codebase-cleanup-report.md` with the baseline counts, the zero-candidate private audit result, the consolidated helper families, the preserved public `clear_cache` method, and the final gates to be filled in Task 11. Do not claim a function was removed unless its exact evidence is recorded.

- [ ] **Step 5: Commit the audit record and any proven deletions**

```powershell
git add docs/test-reports/week8-codebase-cleanup-report.md src/gui_agent
git commit -m "refactor: remove only proven private dead code"
```

Expected: the commit may contain only the report when the audit has no safe deletion candidates.

---

### Task 8: Document Core Types, Perception, and Control

**Files:**
- Modify: `src/gui_agent/types.py`
- Modify: `src/gui_agent/perception/*.py`
- Modify: `src/gui_agent/control/*.py`

**Interfaces:**
- Preserves all behavior; produces only docstrings/comments in these files

- [ ] **Step 1: Capture the expected Ruff documentation failures**

```powershell
uv run --no-sync ruff check src/gui_agent/types.py src/gui_agent/perception `
  src/gui_agent/control --select D --config "lint.pydocstyle.convention='google'"
```

Expected: non-zero with missing-module/class/method/function docstring diagnostics.

- [ ] **Step 2: Document contracts and invariants**

Add module and public API docstrings. Use concise contracts such as:

```python
class DesktopController:
    """Execute validated desktop input, or record it harmlessly in dry-run mode."""


def capture_region(...):
    """Capture one absolute virtual-desktop region without persisting it by default."""
```

Explain image color order, absolute versus local coordinates, OCR backend boundaries, preprocessing immutability, bounding-box invariants, and the fact that `DesktopController` can produce real input only when constructed outside dry-run mode. Do not add comments to obvious field declarations.

- [ ] **Step 3: Run documentation and behavior gates for these packages**

```powershell
uv run --no-sync ruff check src/gui_agent/types.py src/gui_agent/perception `
  src/gui_agent/control --select D --config "lint.pydocstyle.convention='google'"
uv run --no-sync pytest tests/test_types.py tests/test_capture.py tests/test_ocr.py `
  tests/test_localization.py tests/test_perception_benchmark.py `
  tests/test_perception_preprocessing.py tests/test_control.py -q
```

Expected: zero documentation diagnostics and all selected tests pass.

- [ ] **Step 4: Commit core/perception/control documentation**

```powershell
git add src/gui_agent/types.py src/gui_agent/perception src/gui_agent/control
git commit -m "docs: explain core perception and control contracts"
```

---

### Task 9: Document the Agent Package

**Files:**
- Modify: `src/gui_agent/agent/*.py`

**Interfaces:**
- Preserves all agent loop, planner, policy, retry, verification, event, and Qwen behavior

- [ ] **Step 1: Capture agent documentation failures**

```powershell
uv run --no-sync ruff check src/gui_agent/agent --select D `
  --config "lint.pydocstyle.convention='google'"
```

Expected: non-zero before docstrings are added.

- [ ] **Step 2: Document agent contracts and safety boundaries**

Document all public APIs and add targeted invariant comments. The first sentences for safety-sensitive classes must be explicit:

```python
class SafetyPolicy:
    """Authorize real desktop input only after validation and exact user confirmation."""


class PlannedActionExecutor:
    """Execute approved plan actions and report failures without bypassing policy."""
```

Explain strict action schemas, plan-step identity, normalized-to-pixel coordinates, fail-closed confirmation, redacted events, bounded retry/replan behavior, verification evidence, lazy model loading, and prompt-injection-resistant observation summaries. Do not expose input text in comments or examples.

- [ ] **Step 3: Run agent documentation and behavior gates**

```powershell
uv run --no-sync ruff check src/gui_agent/agent --select D `
  --config "lint.pydocstyle.convention='google'"
uv run --no-sync pytest tests/test_agent_cli.py tests/test_agent_coordinates.py `
  tests/test_agent_events.py tests/test_agent_executor.py tests/test_agent_loop.py `
  tests/test_agent_observation.py tests/test_agent_planner.py `
  tests/test_agent_policy.py tests/test_agent_progress.py `
  tests/test_agent_retry.py tests/test_agent_types.py `
  tests/test_agent_verification.py -q
```

Expected: zero documentation diagnostics and all selected tests pass.

- [ ] **Step 4: Commit agent documentation**

```powershell
git add src/gui_agent/agent
git commit -m "docs: explain agent planning and safety contracts"
```

---

### Task 10: Document Datasets, Training, Simulation, Evaluation, and CLIs

**Files:**
- Modify: `src/gui_agent/datasets/*.py`
- Modify: `src/gui_agent/training/*.py`
- Modify: `src/gui_agent/simulation/*.py`
- Modify: `src/gui_agent/evaluation/*.py`
- Modify: `src/gui_agent/cli.py`
- Modify: `src/gui_agent/provenance.py`
- Modify: `src/gui_agent/_models.py`
- Modify: `src/gui_agent/cli_args.py`
- Modify: `src/gui_agent/model_loading.py`

**Interfaces:**
- Preserves every dataset, training, simulation, evaluation, provenance, and CLI behavior

- [ ] **Step 1: Capture remaining production documentation failures**

```powershell
uv run --no-sync ruff check src/gui_agent/datasets src/gui_agent/training `
  src/gui_agent/simulation src/gui_agent/evaluation src/gui_agent/cli.py `
  src/gui_agent/provenance.py src/gui_agent/_models.py `
  src/gui_agent/cli_args.py src/gui_agent/model_loading.py `
  --select D --config "lint.pydocstyle.convention='google'"
```

Expected: non-zero before the remaining docstrings are added.

- [ ] **Step 2: Document remaining production contracts**

Document source-license provenance, deterministic split/no-leakage behavior, LoRA optional dependency boundaries, evaluation schema provenance, synthetic-versus-real input boundaries, deterministic rendering, timing wrappers, report ownership, and CLI orchestration. `SimulationPolicy` and simulated executors must state that they never authorize or emit real desktop input.

- [ ] **Step 3: Run remaining package tests and documentation checks**

```powershell
uv run --no-sync ruff check src/gui_agent/datasets src/gui_agent/training `
  src/gui_agent/simulation src/gui_agent/evaluation src/gui_agent/cli.py `
  src/gui_agent/provenance.py src/gui_agent/_models.py `
  src/gui_agent/cli_args.py src/gui_agent/model_loading.py `
  --select D --config "lint.pydocstyle.convention='google'"
uv run --no-sync pytest tests/test_dataset_adapters.py tests/test_dataset_schema.py `
  tests/test_training_cli.py tests/test_training_collator.py `
  tests/test_training_config.py tests/test_training_dataset.py `
  tests/test_training_evaluation.py tests/test_training_formatting.py `
  tests/test_training_lora.py tests/test_training_schema.py `
  tests/test_simulation_harness.py tests/test_simulation_render.py `
  tests/test_simulation_state.py tests/test_evaluation_cli.py `
  tests/test_evaluation_metrics.py tests/test_evaluation_plots.py `
  tests/test_evaluation_report.py tests/test_evaluation_runner.py `
  tests/test_evaluation_suite.py -q
```

Expected: zero documentation diagnostics and all selected tests pass.

- [ ] **Step 4: Commit remaining documentation**

```powershell
git add src/gui_agent/datasets src/gui_agent/training src/gui_agent/simulation `
  src/gui_agent/evaluation src/gui_agent/cli.py src/gui_agent/provenance.py `
  src/gui_agent/_models.py src/gui_agent/cli_args.py `
  src/gui_agent/model_loading.py
git commit -m "docs: document data training simulation and evaluation APIs"
```

---

### Task 11: Enable the Documentation Gate and Verify Compatibility

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/test-reports/week8-codebase-cleanup-report.md`
- Modify: `docs/superpowers/plans/2026-09-10-codebase-cleanup-docstrings.md`

**Interfaces:**
- Produces: repository-wide Ruff enforcement for production docstrings
- Preserves: test/example ergonomics through narrow per-file ignores

- [ ] **Step 1: Enable Ruff documentation rules**

Use the Google convention and retain the existing rules:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "D"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D"]
"examples/*" = ["D103"]
```

If Ruff reports duplicate constructor or magic-method prose under `D105`/`D107`, document those call boundaries rather than applying a production-wide ignore. Use a narrow ignore only when Ruff requires two docstrings for the same public contract and record the exact rule in the cleanup report.

- [ ] **Step 2: Compare public definitions against the merged baseline**

```powershell
git diff --unified=0 origin/master -- src/gui_agent `
  | Select-String '^[+-](class|def|    def) [A-Za-z][A-Za-z0-9_]*' `
  | Set-Content -Encoding utf8 artifacts/week8-cleanup/public-definition-diff.txt
Get-Content artifacts/week8-cleanup/public-definition-diff.txt
```

Expected: no removed public definition and no changed public callable signature. New shared helper definitions are additions only. Investigate every output line before continuing.

- [ ] **Step 3: Run the full final gate**

```powershell
$env:VIRTUAL_ENV = $null
uv lock --check
uv run --no-sync ruff check .
uv run --no-sync mypy src tests examples scripts
uv run --no-sync pytest -m "not integration" `
  --basetemp artifacts/pytest-week8-cleanup-final `
  --cov=gui_agent --cov-report=term-missing
uv run --no-sync pytest -m integration `
  --basetemp artifacts/pytest-week8-cleanup-final-int
```

Expected: lock, Ruff, and mypy pass; at least 486 tests are collected; non-integration coverage is at least 88%; integration results have only the two baseline opt-in GPU/model skips unless their environment flags are intentionally enabled.

- [ ] **Step 4: Finish the cleanup report and inspect repository hygiene**

Record exact test counts, coverage, Ruff/mypy results, public API comparison, removed private symbols, retained compatibility wrappers, and known limitations. Then run:

```powershell
git status --short --ignored
git diff --check
git diff --stat origin/master...HEAD
git ls-files artifacts .venv models | Should -BeNullOrEmpty
```

In ordinary PowerShell, replace the last Pester-style assertion with:

```powershell
$unexpected = git ls-files artifacts .venv models
if ($unexpected) { throw "generated/private artifacts are tracked: $unexpected" }
```

- [ ] **Step 5: Commit the final gate configuration and report**

```powershell
git add pyproject.toml docs/test-reports/week8-codebase-cleanup-report.md `
  docs/superpowers/plans/2026-09-10-codebase-cleanup-docstrings.md
git commit -m "chore: enforce documented production APIs"
```

- [ ] **Step 6: Re-run final verification on the committed tree**

Repeat Step 3 after the commit. Do not rely on the pre-commit run.

- [ ] **Step 7: Push only the completed branch**

```powershell
git push --set-upstream origin codex/week8-codebase-cleanup
git ls-remote --heads origin refs/heads/codex/week8-codebase-cleanup
git rev-parse HEAD
```

Expected: the remote and local commit hashes match exactly.
