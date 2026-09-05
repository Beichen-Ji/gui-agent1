from pathlib import Path

import pytest

from gui_agent.agent.types import (
    AgentDecision,
    ClickAction,
    TaskPlan,
    TaskStep,
    TypeTextAction,
)
from gui_agent.training.evaluation import EvaluationCase, EvaluationCondition


def _plan(*words: str) -> TaskPlan:
    return TaskPlan(
        goal="evaluation",
        steps=tuple(
            TaskStep(id=f"step-{index}", description=word)
            for index, word in enumerate(words, start=1)
        ),
    )


def _condition(name: str = "baseline") -> EvaluationCondition:
    return EvaluationCondition(
        model=name,
        prompt_profile="week4-baseline",
    )


def _click_case(
    case_id: str = "click",
    *,
    x: int = 500,
    y: int = 500,
) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        canvas=(100, 100),
        instruction="Open Browser",
        plan_keywords=("open", "browser"),
        elements=(),
        expected_action=ClickAction(x=x, y=y),
        target_box=(400, 400, 600, 600),
    )


def test_metrics_count_invalid_outputs_in_every_denominator() -> None:
    from gui_agent.training.evaluation import EvaluationPrediction, evaluate_predictions

    condition = _condition()
    cases = (_click_case("first"), _click_case("second"))
    predictions = (
        EvaluationPrediction(
            condition_id=condition.id,
            case_id="first",
            plan=_plan("Open Browser"),
            action=ClickAction(x=500, y=500),
            latency_ms=10,
            peak_vram_mib=100,
        ),
        EvaluationPrediction(
            condition_id=condition.id,
            case_id="second",
            failure_type="PlannerError",
            latency_ms=20,
            peak_vram_mib=110,
        ),
    )

    report = evaluate_predictions(cases, condition, predictions, cases_sha256="a" * 64)

    assert report.metrics.schema_valid_rate == 0.5
    assert report.metrics.plan_requirement_recall == 0.5
    assert report.metrics.action_kind_accuracy == 0.5
    assert report.metrics.action_parameter_accuracy == 0.5
    assert report.metrics.click_hit_rate == 0.5
    assert report.metrics.median_latency_ms == 15
    assert report.metrics.peak_vram_mib == 110
    assert report.failures == {"PlannerError": 1}


def test_pointer_parameter_tolerance_is_inclusive_at_fifty_grid_points() -> None:
    from gui_agent.training.evaluation import EvaluationPrediction, evaluate_predictions

    condition = _condition()
    case = _click_case()

    accepted = evaluate_predictions(
        (case,),
        condition,
        (
            EvaluationPrediction(
                condition_id=condition.id,
                case_id="click",
                plan=_plan("Open Browser"),
                action=ClickAction(x=550, y=450),
                latency_ms=1,
                peak_vram_mib=1,
            ),
        ),
        cases_sha256="a" * 64,
    )
    rejected = evaluate_predictions(
        (case,),
        condition,
        (
            EvaluationPrediction(
                condition_id=condition.id,
                case_id="click",
                plan=_plan("Open Browser"),
                action=ClickAction(x=551, y=500),
                latency_ms=1,
                peak_vram_mib=1,
            ),
        ),
        cases_sha256="a" * 64,
    )

    assert accepted.metrics.action_parameter_accuracy == 1.0
    assert accepted.metrics.click_hit_rate == 1.0
    assert rejected.metrics.action_parameter_accuracy == 0.0


def test_action_accuracy_is_macro_averaged_over_expected_action_kinds() -> None:
    from gui_agent.training.evaluation import (
        EvaluationCase,
        EvaluationPrediction,
        evaluate_predictions,
    )

    condition = _condition()
    cases = (
        _click_case("click-correct"),
        _click_case("click-wrong"),
        EvaluationCase(
            id="type-correct",
            canvas=(100, 100),
            instruction="Type hello",
            plan_keywords=("type",),
            elements=(),
            expected_action=TypeTextAction(text="hello"),
        ),
    )
    predictions = (
        EvaluationPrediction(
            condition_id=condition.id,
            case_id="click-correct",
            plan=_plan("Open Browser"),
            action=ClickAction(x=500, y=500),
            latency_ms=1,
            peak_vram_mib=1,
        ),
        EvaluationPrediction(
            condition_id=condition.id,
            case_id="click-wrong",
            plan=_plan("Open Browser"),
            action=TypeTextAction(text="wrong"),
            latency_ms=1,
            peak_vram_mib=1,
        ),
        EvaluationPrediction(
            condition_id=condition.id,
            case_id="type-correct",
            plan=_plan("Type"),
            action=TypeTextAction(text="hello"),
            latency_ms=1,
            peak_vram_mib=1,
        ),
    )

    report = evaluate_predictions(cases, condition, predictions, cases_sha256="a" * 64)

    assert report.metrics.action_kind_accuracy == 0.75
    assert report.metrics.action_parameter_accuracy == 0.75


def test_predictions_from_different_conditions_cannot_be_mixed() -> None:
    from gui_agent.training.evaluation import EvaluationPrediction, evaluate_predictions

    with pytest.raises(ValueError, match="condition"):
        evaluate_predictions(
            (_click_case(),),
            _condition("baseline"),
            (
                EvaluationPrediction(
                    condition_id=_condition("adapter").id,
                    case_id="click",
                    failure_type="wrong condition",
                    latency_ms=1,
                    peak_vram_mib=1,
                ),
            ),
            cases_sha256="a" * 64,
        )


def test_report_json_is_deterministic_for_prediction_order(tmp_path: Path) -> None:
    from gui_agent.training.evaluation import (
        EvaluationPrediction,
        evaluate_predictions,
        write_evaluation_report,
    )

    condition = _condition()
    cases = (_click_case("b"), _click_case("a"))
    predictions = (
        EvaluationPrediction(
            condition_id=condition.id,
            case_id="b",
            failure_type="PlannerError",
            latency_ms=2,
            peak_vram_mib=2,
        ),
        EvaluationPrediction(
            condition_id=condition.id,
            case_id="a",
            plan=_plan("Open Browser"),
            action=ClickAction(x=500, y=500),
            latency_ms=1,
            peak_vram_mib=1,
        ),
    )
    first = evaluate_predictions(cases, condition, predictions, cases_sha256="a" * 64)
    second = evaluate_predictions(
        reversed(cases),
        condition,
        reversed(predictions),
        cases_sha256="a" * 64,
    )

    write_evaluation_report(first, tmp_path / "first.json")
    write_evaluation_report(second, tmp_path / "second.json")

    assert first == second
    assert (tmp_path / "first.json").read_bytes() == (
        tmp_path / "second.json"
    ).read_bytes()
    assert [outcome.case_id for outcome in first.outcomes] == ["a", "b"]


def test_condition_identity_includes_model_and_adapter_hashes(tmp_path: Path) -> None:
    from gui_agent.training.evaluation import build_evaluation_condition

    adapters: list[Path] = []
    for name, weight in (("first", b"first-weight"), ("second", b"second-weight")):
        root = tmp_path / name
        adapter = root / "adapter"
        adapter.mkdir(parents=True)
        (root / "run-manifest.json").write_text(
            f'{{"adapter": "{name}"}}',
            encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(weight)
        adapters.append(root)

    first = build_evaluation_condition(
        model_name="model-a",
        prompt_profile="week5-grounded",
        adapter_path=adapters[0],
    )
    other_model = build_evaluation_condition(
        model_name="model-b",
        prompt_profile="week5-grounded",
        adapter_path=adapters[0],
    )
    other_adapter = build_evaluation_condition(
        model_name="model-a",
        prompt_profile="week5-grounded",
        adapter_path=adapters[1],
    )

    assert len({first.id, other_model.id, other_adapter.id}) == 3
    assert first.adapter_label is not None
    assert first.adapter_label.endswith("first")
    assert str(tmp_path) not in first.model_dump_json()


def test_writer_never_overwrites_an_unowned_or_non_json_file(tmp_path: Path) -> None:
    from gui_agent.training.evaluation import (
        EvaluationPrediction,
        evaluate_predictions,
        write_evaluation_report,
    )

    condition = _condition()
    report = evaluate_predictions(
        (_click_case(),),
        condition,
        (
            EvaluationPrediction(
                condition_id=condition.id,
                case_id="click",
                plan=_plan("Open Browser"),
                action=ClickAction(x=500, y=500),
                latency_ms=1,
                peak_vram_mib=1,
            ),
        ),
        cases_sha256="a" * 64,
    )
    unowned = tmp_path / "adapter_model.safetensors"
    unowned.write_bytes(b"keep me")

    with pytest.raises(ValueError, match="JSON"):
        write_evaluation_report(report, unowned, overwrite=True)
    assert unowned.read_bytes() == b"keep me"

    unowned_json = tmp_path / "user.json"
    unowned_json.write_text('{"kind": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError, match="owned"):
        write_evaluation_report(report, unowned_json, overwrite=True)
    assert unowned_json.read_text(encoding="utf-8") == '{"kind": "something-else"}'


def test_run_evaluation_renders_converts_failures_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gui_agent.agent.planner import PlannerError
    from gui_agent.training import evaluation

    class FakePlanner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def create_plan(self, goal: str, _observation: object) -> TaskPlan:
            if goal == "Fail safely":
                raise PlannerError("synthetic failure")
            return _plan("Open Browser")

        def next_action(self, _state: object) -> AgentDecision:
            return AgentDecision(
                current_step_id="step-1",
                rationale_summary="Visible button",
                action=ClickAction(x=50, y=50),
                expected_outcome="Browser opens",
            )

    cases = tmp_path / "cases.json"
    cases.write_text(
        """{
  "schema_version": 1,
  "cases": [
    {
      "id": "success",
      "canvas": [100, 100],
      "instruction": "Open Browser",
      "plan_keywords": ["open", "browser"],
      "elements": [{"label": "Browser", "box": [20, 20, 80, 80], "fill": "white"}],
      "expected_action": {"kind": "click", "x": 505, "y": 505},
      "target_box": [400, 400, 600, 600]
    },
    {
      "id": "failure",
      "canvas": [100, 100],
      "instruction": "Fail safely",
      "plan_keywords": ["fail"],
      "elements": [],
      "expected_action": {"kind": "type_text", "text": "never"},
      "target_box": null
    }
  ]
}\n""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluation, "QwenTransformersPlanner", FakePlanner)
    monkeypatch.setattr(evaluation, "_reset_cuda_peak", lambda: None)
    monkeypatch.setattr(evaluation, "_peak_vram_mib", lambda: 123.0)
    ticks = iter((1.0, 1.01, 2.0, 2.02))

    output = Path("artifacts/evaluation.json")
    report = evaluation.run_evaluation(
        cases_path=cases,
        model_name="model-a",
        adapter_path=None,
        prompt_profile="week4-baseline",
        output_path=output,
        clock=lambda: next(ticks),
    )

    assert report.metrics.schema_valid_rate == 0.5
    assert report.metrics.click_hit_rate == 1.0
    assert report.failures == {"PlannerError": 1}
    assert report.outcomes[1].predicted_action == ClickAction(x=505, y=505)
    assert output.is_file()
