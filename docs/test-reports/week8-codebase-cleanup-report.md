# Week 8 Codebase Cleanup Report

## Scope

This cleanup preserves the existing public API and runtime behavior while reducing
internal duplication and documenting production contracts. It does not change the
Week 7 terminal-focus limitation, safety decisions, schemas, CLI options, model
defaults, or persisted output formats.

## Baseline

- Baseline revision: `origin/master` at cleanup start (`7f41422`).
- Public definition lines under `src/gui_agent`: 332.
- Test collection: 486 tests.
- Non-integration result: 473 passed, 13 deselected, 88% coverage.
- Integration result: 11 passed, 2 skipped, 473 deselected.
- Ruff and mypy: passed before cleanup.
- Google-style docstring audit: 378 missing-docstring diagnostics before cleanup.

## Consolidation Results

- Strict, frozen Pydantic configuration now comes from
  `gui_agent._models.StrictFrozenModel`.
- CLI integer lower-bound validation now comes from
  `gui_agent.cli_args.parse_integer_at_least`; existing private callbacks retain
  their original error messages and conversion-error behavior.
- Optional multimodal processor and model loading now comes from
  `gui_agent.model_loading`; imports remain lazy and existing injection seams remain.
- Text normalization and image fingerprints now have one implementation each under
  `gui_agent.perception`.
- Dataset URLs and license strings now come from the immutable
  `gui_agent.datasets.sources.DATASET_SOURCES` mapping.
- `ObservationBuilder.clear_cache()` remains public and has a characterization test
  proving that it forces OCR on the next unchanged frame.

## Dead-Code Evidence

The private top-level definition audit found zero names with at most one textual
reference across source, tests, examples, and scripts. Ruff's `F`, `B`, and `SIM`
checks and Python bytecode compilation also passed. Therefore no private function or
class has been removed without evidence. Existing callbacks, protocol methods,
serialization hooks, and dependency-injection seams remain intact.

Eight duplicated private `_StrictFrozenModel` class definitions were removed from
the agent, dataset, training, and evaluation schema modules after their strict,
frozen, and extra-field behavior was protected by a shared-base test. The private
CLI callbacks and Qwen/LoRA default loader wrappers remain because they preserve
module seams and caller-specific behavior.

Generic adapter provenance remains centralized in `provenance.py`. Qwen-specific
model, grid, prompt-profile, and hash validation remains in `agent/qwen.py` because
it is runtime adapter policy rather than duplicate generic provenance behavior.

## Final Verification

- Public API comparison: 314 baseline AST-visible definitions and 320 current
  definitions; zero changed signatures. The only relocated definition is
  `perception.benchmark.normalize_text`, whose original import path and signature
  now resolve directly to `perception.text.normalize_text`.
- Ruff: passed repository-wide with Google-style production docstrings enabled;
  production missing-docstring diagnostics decreased from 378 to 0.
- mypy: passed for 125 source files under `src`, `tests`, `examples`, and `scripts`.
- Non-integration tests: 482 passed, 13 deselected; total coverage remained 88%.
- Integration tests: 11 passed, 2 expected opt-in GPU/model tests skipped, 482
  deselected.
- Collection: 495 tests, nine more than the 486-test baseline.
- Repository hygiene: `git diff --check` passed. No environment, model, screenshot,
  or generated artifact is tracked; the pre-existing `artifacts/.gitkeep` directory
  placeholder remains intentionally tracked.

## Known Limitation

The local Qwen planner can still select the wrong desktop target when another
window overlaps the Week 4 testbed. Fixing focus or window targeting would change
runtime behavior and is intentionally outside this compatibility-preserving cleanup.
