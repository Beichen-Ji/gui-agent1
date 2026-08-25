import re

from gui_agent.agent.types import AgentState, Observation

_ALLOWED_ACTIONS = ("click", "type_text", "hotkey", "scroll", "drag", "wait", "finish")
_WINDOWS_PATH = re.compile(r"(?i)\b[a-z]:\\[^\s\"']+")
_POSIX_PATH = re.compile(r"(?<!\w)/(?:Users|home|mnt|opt|private|tmp|var)/[^\s\"']+")
_LIKELY_SECRET = re.compile(r"(?i)\b(?:sk|key|token)-[a-z0-9_-]{8,}\b")


def _safe_text(value: str, *, max_length: int) -> str:
    sanitized = _WINDOWS_PATH.sub("[local-path]", value)
    sanitized = _POSIX_PATH.sub("[local-path]", sanitized)
    sanitized = _LIKELY_SECRET.sub("[secret]", sanitized)
    return " ".join(sanitized.split())[:max_length]


def _observation_summary(observation: Observation) -> str:
    screenshot = observation.screenshot
    lines = [
        (
            f"Screen: {screenshot.width}x{screenshot.height}; "
            f"origin=({screenshot.origin.x},{screenshot.origin.y}); "
            f"step_index={observation.step_index}"
        ),
        "OCR detections:",
    ]
    for detection in observation.detections[:50]:
        box = detection.box
        text = _safe_text(detection.text, max_length=120)
        lines.append(
            f'- text="{text}" confidence={detection.confidence:.3f} '
            f"box=({box.left},{box.top},{box.right},{box.bottom})"
        )
    if not observation.detections:
        lines.append("- none")
    return "\n".join(lines)


def build_plan_prompt(goal: str, observation: Observation) -> str:
    safe_goal = _safe_text(goal, max_length=1000)
    allowed = ", ".join(_ALLOWED_ACTIONS)
    return (
        "Create a short desktop task plan from the current visual observation.\n"
        f"Goal: {safe_goal}\n"
        f"Allowed action kinds: {allowed}.\n"
        "Use stable step IDs and concise descriptions. Return only the structured plan.\n"
        f"{_observation_summary(observation)}"
    )


def build_action_prompt(state: AgentState) -> str:
    active_step = (
        state.decisions[-1].current_step_id if state.decisions else state.plan.steps[0].id
    )
    plan_lines = "\n".join(
        f"- {step.id}: {_safe_text(step.description, max_length=500)}"
        for step in state.plan.steps
    )
    if state.results:
        result_lines = "\n".join(
            (
                f"- step_index={result.step_index} status={result.status}: "
                f"{_safe_text(result.message, max_length=500)}"
            )
            for result in state.results[-3:]
        )
    else:
        result_lines = "- none"
    allowed = ", ".join(_ALLOWED_ACTIONS)
    return (
        "Choose exactly one next desktop action from the current visual observation.\n"
        f"Goal: {_safe_text(state.goal, max_length=1000)}\n"
        f"Active step: {active_step}\n"
        f"Plan:\n{plan_lines}\n"
        f"Recent results:\n{result_lines}\n"
        f"Allowed action kinds: {allowed}.\n"
        "Return a concise rationale summary, expected outcome, and one structured action.\n"
        f"{_observation_summary(state.observation)}"
    )


__all__ = ["build_action_prompt", "build_plan_prompt"]
