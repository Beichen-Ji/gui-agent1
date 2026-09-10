# Codebase Cleanup and Documentation Design

## Context

The repository has grown through eight weekly milestones. Its behavior is covered by a broad test suite, but several private helpers are duplicated across packages and many public modules, classes, and functions lack useful documentation. The cleanup must improve maintainability without changing the public API, command-line behavior, persisted schemas, safety policy, model behavior, evaluation semantics, or task outcomes.

This work starts from merged Week 7 commit `7f41422` on the isolated branch `codex/week8-codebase-cleanup`.

## Goals

1. Remove imports, variables, private functions, and private classes that are provably unused.
2. Consolidate equivalent private implementations behind focused internal modules.
3. Add concise English module docstrings and public API docstrings throughout `src/gui_agent`.
4. Add inline comments only where they explain a non-obvious invariant, safety boundary, compatibility constraint, or algorithmic decision.
5. Preserve the observable behavior and public interfaces of the merged Week 7 repository.
6. Push the verified result to a dedicated GitHub branch after all cleanup and documentation work is complete.

## Non-goals

- Removing or renaming public classes, methods, functions, constants, CLI options, or entry points.
- Changing action authorization, confirmation, retry, verification, OCR, planning, training, simulation, or evaluation behavior.
- Fixing the real-desktop terminal focus limitation discovered during Week 7 evaluation.
- Renaming week-numbered files, changing package version metadata, or restructuring the user documentation hierarchy.
- Changing persisted `kind` literals, JSON field names, event payloads, manifests, or report schemas.
- Adding new runtime dependencies.

## Compatibility Contract

The refactor is accepted only when all of the following remain true:

- Existing imports from `gui_agent` and its public subpackages keep resolving.
- Every existing public symbol remains present with the same callable signature.
- CLI commands, options, defaults, help-visible names, exit codes, and JSON output shapes remain unchanged.
- Persisted schema literals and field names remain byte-for-byte compatible at the contract level.
- Safety remains fail-closed; no confirmation bypass or broader action allowance is introduced.
- The existing task, model, OCR, training, and evaluation tests keep their behavior.

`ObservationBuilder.clear_cache()` is retained because its non-underscored name makes it part of the public class surface, even though the repository currently has no caller. It will receive a contract test and documentation instead of being deleted.

## Cleanup Strategy

### Evidence required before deletion

A private definition may be deleted only when all of these checks agree:

1. Ruff or the Python compiler identifies the associated import, variable, or branch as unused or unreachable, when applicable.
2. A repository-wide `rg` search finds no runtime, test, example, script, or documentation consumer.
3. An AST-based inventory finds no indirect same-name reference that simple text search missed.
4. The focused tests and complete verification suite pass after deletion.

Callbacks, protocol implementations, magic methods, framework entry points, serializer hooks, and symbols imported dynamically are not treated as dead solely because direct calls are absent.

### Duplicate implementations

Equivalent private helpers will move to small internal modules while their consumers retain the same public behavior:

| Shared responsibility | Canonical internal location | Existing consumers |
|---|---|---|
| Strict frozen Pydantic base model | `src/gui_agent/_models.py` | agent, datasets, training, and evaluation schemas |
| Positive/non-negative CLI integer parsing | `src/gui_agent/cli_args.py` | root CLI, dataset CLI, training CLI, OCR benchmark script |
| Default processor/model loaders | `src/gui_agent/model_loading.py` | agent Qwen provider and LoRA training |
| Adapter manifest discovery and validation | `src/gui_agent/provenance.py` | agent adapter loading and training evaluation provenance |
| Screenshot/frame fingerprints | `src/gui_agent/perception/fingerprint.py` | observation cache and outcome verification |
| OCR text normalization | `src/gui_agent/perception/text.py` | verification, benchmark scoring, and event metadata |
| Dataset URLs and licenses | `src/gui_agent/datasets/sources.py` | dataset normalization and training manifests |

The consolidation will preserve existing exception types, messages tested by the suite, accepted input types, return values, hashing inputs, and normalization semantics. Internal helpers may remain as thin private wrappers when tests monkeypatch them or when a wrapper preserves an established injection seam.

## Documentation Strategy

Documentation will be added in this order: core types, perception, control, agent, datasets, training, simulation, evaluation, and CLI modules.

- Every production module in `src/gui_agent` receives a concise module docstring.
- Every public class, exception, protocol, method, property, and function receives a useful docstring.
- Docstrings explain purpose, behavioral boundaries, side effects, failure behavior, or invariants rather than restating names and annotations.
- `Args`, `Returns`, and `Raises` sections are included only when they convey information not obvious from the signature.
- Private helpers receive docstrings only when their contract or reasoning is non-obvious.
- Inline comments explain why a constraint exists; they do not narrate individual Python statements.
- Safety-related classes state in their first docstring sentence whether they can produce real desktop input.
- Source comments and docstrings use English to match identifiers and type annotations. Longer explanatory documents may remain Chinese.

Ruff's `D` rules with the Google convention will enforce the production documentation boundary. Tests are exempt from docstrings, and examples may omit function docstrings where the code is intentionally tutorial-style. Narrow per-file or per-rule ignores are allowed only for a documented reason; blanket production-package ignores are not.

## Implementation Sequence

1. Capture a machine-readable baseline inventory of public symbols and signatures for later comparison.
2. Add compatibility tests for public surfaces that are at risk during consolidation, including `ObservationBuilder.clear_cache()`.
3. Consolidate one helper family at a time, with a focused red/green test cycle and a separate reviewable commit.
4. Run the dead-code evidence checks and remove only confirmed private dead code.
5. Add documentation package by package, keeping runtime changes separate from docstring-only commits.
6. Enable the Ruff docstring gate only after the production source tree is clean under that rule set.
7. Compare the final public API inventory with the baseline and investigate every difference.
8. Run all verification gates and inspect the final Git diff for generated files, model data, screenshots, OCR text, and other ignored artifacts.
9. Push `codex/week8-codebase-cleanup` only after every required gate passes.

## Verification

The merged Week 7 baseline on 2026-09-10 is:

- `uv lock --check`: pass
- `uv run --no-sync ruff check .`: pass
- `uv run --no-sync mypy src tests examples scripts`: 113 source files, no issues
- non-integration tests: 473 passed, 13 deselected, 88% coverage
- integration tests: 11 passed, 2 opt-in GPU/model tests skipped, 473 deselected
- total collected tests: 486

The final gate will run:

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

Acceptance requires zero Ruff or mypy errors, no unexpected test skips, no reduction in the 486-test collection, and non-integration coverage of at least 88%. The public API inventory must show no removed or signature-changed public symbols.

## Risk Controls

- **Semantic drift while deduplicating:** characterize each existing implementation before moving it and preserve separately tested edge cases.
- **Monkeypatch seams breaking:** search tests for private loader/parser patch points and retain wrappers where the seam is intentional.
- **Over-documentation obscuring code:** prefer short contract-focused docstrings and avoid comments on self-explanatory statements.
- **False-positive dead code removal:** require static, textual, AST, and test evidence; preserve uncertain symbols.
- **Accidental user-facing change:** review CLI help, exported names, schema constants, and serialized snapshots separately from the general test suite.
- **Generated or private artifacts entering Git:** stage explicit paths, inspect `git status --ignored`, and reject screenshots, model weights, complete OCR output, and `artifacts/` content.

## Deliverables

- Consolidated internal helper modules and updated consumers.
- Confirmed private dead-code removals with test coverage.
- Documented production modules and public APIs with a Ruff documentation gate.
- A compatibility verification record summarizing public API comparison, tests, typing, lint, and coverage.
- A clean GitHub branch `codex/week8-codebase-cleanup` containing only source, tests, configuration, and documentation changes.
