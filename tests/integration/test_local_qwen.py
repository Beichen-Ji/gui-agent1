import os
import time

import pytest

from examples.model_smoke import synthetic_observation
from gui_agent.agent.qwen import QwenTransformersPlanner
from gui_agent.agent.types import AgentState


@pytest.mark.integration
def test_local_qwen_produces_a_plan_and_action_from_synthetic_ui() -> None:
    if os.environ.get("GUI_AGENT_RUN_LOCAL_QWEN") != "1":
        pytest.skip("set GUI_AGENT_RUN_LOCAL_QWEN=1 to load the real local model")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the local Qwen integration test")

    planner = QwenTransformersPlanner()
    observation = synthetic_observation()

    load_started = time.perf_counter()
    planner._ensure_loaded()
    torch.cuda.synchronize()
    load_elapsed = time.perf_counter() - load_started

    torch.cuda.reset_peak_memory_stats()
    plan_started = time.perf_counter()
    plan = planner.create_plan("Open the synthetic browser", observation)
    torch.cuda.synchronize()
    plan_elapsed = time.perf_counter() - plan_started
    state = AgentState(
        goal="Open the synthetic browser",
        plan=plan,
        observation=observation,
        decisions=(),
        results=(),
    )
    action_started = time.perf_counter()
    decision = planner.next_action(state)
    torch.cuda.synchronize()
    action_elapsed = time.perf_counter() - action_started
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"model={planner.model_name} load_seconds={load_elapsed:.3f} "
        f"plan_seconds={plan_elapsed:.3f} action_seconds={action_elapsed:.3f} "
        f"peak_allocated_gib={peak_memory_gib:.3f}"
    )

    assert plan.steps
    assert decision.current_step_id
    assert load_elapsed > 0
    assert plan_elapsed > 0
    assert action_elapsed > 0
    assert peak_memory_gib > 0
