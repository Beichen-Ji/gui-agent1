from pathlib import Path

from gui_agent.agent.planner import FakePlanner
from gui_agent.agent.types import AgentDecision, FinishAction, TaskPlan, TaskStep
from gui_agent.evaluation.report import EvaluationContext
from gui_agent.evaluation.runner import run_evaluation
from gui_agent.evaluation.suite import EvaluationTask, load_task_suite


def _context() -> EvaluationContext:
    return EvaluationContext(
        suite_sha256="a" * 64,
        conditions_sha256="b" * 64,
        git_revision="deadbeef",
        model="fake",
        prompt_profile="week4-baseline",
        ocr_profile="balanced",
        observation_mode="oracle",
        seed=7,
        environment={"python": "3.11"},
    )


def _planner(task: EvaluationTask, _resolution: tuple[int, int]) -> FakePlanner:
    steps = (
        TaskStep(id="open", description="Open Browser"),
        TaskStep(id="done", description="Finish"),
    )
    plan = TaskPlan(goal=task.instruction, steps=steps)
    return FakePlanner(
        plan=plan,
        decisions=(
            AgentDecision(
                current_step_id="open",
                rationale_summary="The Browser tab is visible.",
                action=task.reference_actions[0],
                expected_outcome="Browser controls become visible.",
            ),
            AgentDecision(
                current_step_id="done",
                rationale_summary="The Browser search box is visible.",
                action=FinishAction(success=True, summary="Browser opened"),
                expected_outcome=task.success_criteria,
            ),
        ),
    )


def test_runner_uses_real_agent_loop_and_writes_per_task_events(tmp_path: Path) -> None:
    task = load_task_suite(Path("configs/week7_task_suite.json"))[0]
    output = tmp_path / "evaluation"

    report = run_evaluation(
        (task,),
        context=_context(),
        resolution=(1280, 720),
        planner_factory=_planner,
        output_dir=output,
    )

    assert report.outcomes[0].status == "succeeded"
    assert report.outcomes[0].step_count == 2
    assert (output / task.id / "events.jsonl").is_file()
    assert (output / "evaluation.json").is_file()


def test_resume_skips_completed_tasks_without_changing_report_bytes(tmp_path: Path) -> None:
    task = load_task_suite(Path("configs/week7_task_suite.json"))[0]
    output = tmp_path / "evaluation"
    first = run_evaluation(
        (task,),
        context=_context(),
        resolution=(1280, 720),
        planner_factory=_planner,
        output_dir=output,
    )
    before = (output / "evaluation.json").read_bytes()
    calls = 0

    def unexpected_factory(
        _task: EvaluationTask,
        _resolution: tuple[int, int],
    ) -> FakePlanner:
        nonlocal calls
        calls += 1
        raise AssertionError("a completed task must be skipped")

    resumed = run_evaluation(
        (task,),
        context=_context(),
        resolution=(1280, 720),
        planner_factory=unexpected_factory,
        output_dir=output,
        resume=True,
    )

    assert resumed == first
    assert calls == 0
    assert (output / "evaluation.json").read_bytes() == before
