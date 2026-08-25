import argparse
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

import numpy as np
from pydantic import TypeAdapter, ValidationError

from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.loop import AgentRunResult, GUIAgent, ObservationSource
from gui_agent.agent.observation import ObservationBuilder
from gui_agent.agent.planner import FakePlanner, LangChainPlanner, MultimodalPlanner
from gui_agent.agent.policy import SafetyPolicy
from gui_agent.agent.qwen import DEFAULT_QWEN_MODEL, QwenTransformersPlanner
from gui_agent.agent.types import (
    AgentAction,
    AgentDecision,
    FinishAction,
    Observation,
    TaskPlan,
    TaskStep,
    WaitAction,
)
from gui_agent.control.controller import DesktopController
from gui_agent.perception.capture import ScreenCapture
from gui_agent.perception.ocr import EasyOCRBackend
from gui_agent.types import BoundingBox, OCRDetection, Point, ScreenRegion, ScreenshotResult

Provider: TypeAlias = Literal["fake", "qwen", "openai-compatible"]
InputFunction: TypeAlias = Callable[[str], str]
DEFAULT_TASKS_PATH = Path(__file__).resolve().parents[2] / "configs" / "week4_tasks.json"
_ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    id: str
    instruction: str
    success_criteria: str
    actions: tuple[AgentAction, ...]


@dataclass(frozen=True, slots=True)
class RunConfig:
    goal: str
    task_id: str | None
    provider: Provider
    model: str
    monitor: int
    max_steps: int
    execute: bool
    allow_remote_image: bool
    trace_dir: Path | None
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)
    fake_actions: tuple[AgentAction, ...] = ()


class AgentRunner(Protocol):
    def run(self, goal: str, *, max_steps: int = 10) -> AgentRunResult: ...


RuntimeFactory: TypeAlias = Callable[[RunConfig, InputFunction], AgentRunner]


def _positive_integer(value: str) -> int:
    try:
        converted = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if converted < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return converted


def load_task_definitions(path: Path) -> dict[str, TaskDefinition]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read task definitions: {path}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("task definitions require schema_version 1")
    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("task definitions must contain a non-empty tasks list")

    tasks: dict[str, TaskDefinition] = {}
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise ValueError("each task definition must be an object")
        task_id = raw_task.get("id")
        instruction = raw_task.get("instruction")
        success_criteria = raw_task.get("success_criteria")
        raw_actions = raw_task.get("actions")
        if (
            not isinstance(task_id, str)
            or not task_id.strip()
            or not isinstance(instruction, str)
            or not instruction.strip()
            or not isinstance(success_criteria, str)
            or not success_criteria.strip()
            or not isinstance(raw_actions, list)
            or not raw_actions
        ):
            raise ValueError("task definition fields are missing or invalid")
        if task_id in tasks:
            raise ValueError(f"duplicate task id: {task_id}")
        try:
            actions = tuple(
                _ACTION_ADAPTER.validate_json(json.dumps(raw_action))
                for raw_action in raw_actions
            )
        except ValidationError as error:
            raise ValueError(f"task {task_id} contains an invalid action") from error
        tasks[task_id] = TaskDefinition(
            id=task_id,
            instruction=instruction,
            success_criteria=success_criteria,
            actions=actions,
        )
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gui-agent",
        description="Safe Week 4 desktop GUI agent prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "dataset",
        add_help=False,
        help="Normalize a public GUI task dataset",
    )
    subparsers.add_parser(
        "model-smoke",
        add_help=False,
        help="Run a synthetic one-shot model check",
    )

    run = subparsers.add_parser("run", help="Run the bounded GUI agent loop")
    task_group = run.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Natural-language desktop task")
    task_group.add_argument("--task-id", help="ID from configs/week4_tasks.json")
    run.add_argument(
        "--provider",
        choices=("fake", "qwen", "openai-compatible"),
        default="fake",
    )
    run.add_argument(
        "--model",
        default=os.environ.get("GUI_AGENT_MODEL", DEFAULT_QWEN_MODEL),
    )
    run.add_argument("--api-base", default=os.environ.get("GUI_AGENT_API_BASE"))
    run.add_argument("--api-key", default=os.environ.get("GUI_AGENT_API_KEY"))
    run.add_argument("--monitor", type=_positive_integer, default=1)
    run.add_argument("--max-steps", type=_positive_integer, default=10)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--allow-remote-image", action="store_true")
    run.add_argument("--trace-dir", type=Path)
    return parser


def _dataset_main(argv: Sequence[str]) -> int:
    from gui_agent.datasets.cli import main as dataset_main

    return dataset_main(argv)


def _model_smoke_main(argv: Sequence[str]) -> int:
    from gui_agent.agent.smoke import main as model_smoke_main

    return model_smoke_main(argv)


class _SyntheticObserver:
    def observe(self, step_index: int) -> Observation:
        screenshot = ScreenshotResult(
            image=np.full((360, 640, 3), 245, dtype=np.uint8),
            monitor_index=None,
            captured_at=datetime.now(UTC),
            origin=Point(0, 0),
        )
        detections = (
            OCRDetection("Browser", 0.99, BoundingBox(40, 60, 180, 130)),
            OCRDetection("Search", 0.99, BoundingBox(190, 60, 320, 130)),
            OCRDetection("Files", 0.99, BoundingBox(330, 60, 450, 130)),
            OCRDetection("Messages", 0.99, BoundingBox(460, 60, 600, 130)),
        )
        return Observation(
            screenshot=screenshot,
            detections=detections,
            step_index=step_index,
        )


def _fake_planner(goal: str, configured: tuple[AgentAction, ...]) -> FakePlanner:
    actions = configured or (
        WaitAction(seconds=0.01),
        FinishAction(success=True, summary="Synthetic dry-run completed"),
    )
    steps = tuple(
        TaskStep(id=f"step-{index + 1}", description=f"Perform {action.kind}")
        for index, action in enumerate(actions)
    )
    decisions = tuple(
        AgentDecision(
            current_step_id=steps[index].id,
            rationale_summary="Use the configured deterministic fake action",
            action=action,
            expected_outcome=f"The synthetic testbed accepts {action.kind}",
        )
        for index, action in enumerate(actions)
    )
    return FakePlanner(plan=TaskPlan(goal=goal, steps=steps), decisions=decisions)


def build_runtime(config: RunConfig, input_fn: InputFunction) -> GUIAgent:
    planner: MultimodalPlanner
    observer: ObservationSource
    if config.provider == "fake":
        observer = _SyntheticObserver()
        planner = _fake_planner(config.goal, config.fake_actions)
        synthetic_bounds = ScreenRegion(0, 0, 640, 360)
        controller = DesktopController(
            dry_run=True,
            bounds_provider=lambda: synthetic_bounds,
        )
    else:
        capture = ScreenCapture()
        observer = ObservationBuilder(
            capture,
            EasyOCRBackend(),
            monitor_index=config.monitor,
            min_confidence=0.5,
        )
        if config.provider == "qwen":
            planner = QwenTransformersPlanner(model_name=config.model)
        else:
            planner = LangChainPlanner(
                model_name=config.model,
                base_url=config.api_base,
                api_key=config.api_key,
                allow_remote_image=config.allow_remote_image,
            )
        controller = DesktopController(
            dry_run=not config.execute,
            bounds_provider=capture.virtual_bounds,
        )
    return GUIAgent(
        observer,
        planner,
        SafetyPolicy(execute=config.execute, input_fn=input_fn),
        ActionExecutor(controller),
    )


def _write_trace(result: AgentRunResult, trace_dir: Path) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / "run-summary.json"
    payload = {
        "schema_version": 1,
        "status": result.status,
        "failure_stage": result.failure_stage,
        "decision_count": len(result.decisions),
        "result_statuses": [item.status for item in result.results],
        "action_kinds": [item.action.kind for item in result.decisions],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: InputFunction = input,
    runtime_factory: RuntimeFactory | None = None,
) -> int:
    parser = build_parser()
    args, remainder = parser.parse_known_args(argv)
    if args.command == "dataset":
        return _dataset_main(remainder)
    if args.command == "model-smoke":
        return _model_smoke_main(remainder)
    if remainder:
        parser.error(f"unrecognized arguments: {' '.join(remainder)}")
    if args.provider == "openai-compatible" and not args.allow_remote_image:
        parser.error("openai-compatible provider requires --allow-remote-image")

    task_id = cast(str | None, args.task_id)
    configured_actions: tuple[AgentAction, ...] = ()
    if task_id is None:
        goal = cast(str, args.task)
    else:
        try:
            task = load_task_definitions(DEFAULT_TASKS_PATH)[task_id]
        except KeyError:
            parser.error(f"unknown task id: {task_id}")
        except ValueError as error:
            parser.error(str(error))
        goal = task.instruction
        configured_actions = task.actions

    config = RunConfig(
        goal=goal,
        task_id=task_id,
        provider=cast(Provider, args.provider),
        model=cast(str, args.model),
        monitor=cast(int, args.monitor),
        max_steps=cast(int, args.max_steps),
        execute=cast(bool, args.execute),
        allow_remote_image=cast(bool, args.allow_remote_image),
        trace_dir=cast(Path | None, args.trace_dir),
        api_base=cast(str | None, args.api_base),
        api_key=cast(str | None, args.api_key),
        fake_actions=configured_actions,
    )
    runner = (runtime_factory or build_runtime)(config, input_fn)
    result = runner.run(config.goal, max_steps=config.max_steps)
    summary = {
        "status": result.status,
        "message": result.message,
        "failure_stage": result.failure_stage,
        "decision_count": len(result.decisions),
        "result_count": len(result.results),
        "dry_run": not config.execute or config.provider == "fake",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if config.trace_dir is not None:
        _write_trace(result, config.trace_dir)
    return 0 if result.status == "succeeded" else 1


__all__ = [
    "DEFAULT_TASKS_PATH",
    "AgentRunner",
    "RunConfig",
    "TaskDefinition",
    "build_parser",
    "build_runtime",
    "load_task_definitions",
    "main",
]
