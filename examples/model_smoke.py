import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import cv2
import numpy as np

from gui_agent.agent.planner import (
    FakePlanner,
    LangChainPlanner,
    MultimodalPlanner,
    PlannerError,
)
from gui_agent.agent.qwen import DEFAULT_QWEN_MODEL, QwenTransformersPlanner
from gui_agent.agent.types import (
    AgentDecision,
    AgentState,
    ClickAction,
    Observation,
    TaskPlan,
    TaskStep,
)
from gui_agent.types import BoundingBox, OCRDetection, Point, ScreenshotResult


def synthetic_observation() -> Observation:
    image = np.full((360, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (40, 60), (180, 130), (230, 230, 230), thickness=-1)
    cv2.rectangle(image, (40, 60), (180, 130), (40, 90, 180), thickness=2)
    cv2.putText(
        image,
        "Browser",
        (61, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    screenshot = ScreenshotResult(
        image=image,
        monitor_index=None,
        captured_at=datetime.now(UTC),
        origin=Point(0, 0),
    )
    detection = OCRDetection("Browser", 0.99, BoundingBox(40, 60, 180, 130))
    return Observation(screenshot=screenshot, detections=(detection,), step_index=0)


def _fake_planner() -> FakePlanner:
    plan = TaskPlan(
        goal="Open the synthetic browser",
        steps=(TaskStep(id="step-1", description="Click the Browser button"),),
    )
    decision = AgentDecision(
        current_step_id="step-1",
        rationale_summary="The synthetic Browser button is visible",
        action=ClickAction(x=110, y=95),
        expected_outcome="The synthetic browser opens",
    )
    return FakePlanner(plan=plan, decisions=(decision,))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one plan and action inference against a synthetic GUI"
    )
    parser.add_argument(
        "--provider",
        choices=("fake", "qwen", "openai-compatible"),
        default="fake",
    )
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--api-base", default=os.environ.get("GUI_AGENT_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("GUI_AGENT_API_KEY"))
    parser.add_argument("--allow-remote-image", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.synthetic:
        parser.error("--synthetic is required; this smoke test never captures the desktop")

    planner: MultimodalPlanner
    if args.provider == "fake":
        planner = _fake_planner()
    elif args.provider == "qwen":
        planner = QwenTransformersPlanner(model_name=args.model)
    else:
        planner = LangChainPlanner(
            model_name=args.model,
            base_url=args.api_base,
            api_key=args.api_key,
            allow_remote_image=args.allow_remote_image,
        )

    observation = synthetic_observation()
    goal = "Open the synthetic browser"
    try:
        plan = planner.create_plan(goal, observation)
        state = AgentState(
            goal=goal,
            plan=plan,
            observation=observation,
            decisions=(),
            results=(),
        )
        decision = planner.next_action(state)
    except PlannerError as exc:
        print(f"Model smoke failed: {exc}", file=sys.stderr)
        return 1

    print("TaskPlan:")
    print(plan.model_dump_json(indent=2))
    print("AgentDecision:")
    print(decision.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
