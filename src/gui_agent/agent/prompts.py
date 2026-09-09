import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from gui_agent.agent.types import AgentState, Observation, ReplanContext

_ALLOWED_ACTIONS = ("click", "type_text", "hotkey", "scroll", "drag", "wait", "finish")
_WINDOWS_PATH = re.compile(r"(?i)\b[a-z]:\\[^\s\"']+")
_POSIX_PATH = re.compile(r"(?<!\w)/(?:Users|home|mnt|opt|private|tmp|var)/[^\s\"']+")
_LIKELY_SECRET = re.compile(r"(?i)\b(?:sk|key|token)-[a-z0-9_-]{8,}\b")


@dataclass(frozen=True, slots=True)
class PromptProfile:
    id: str
    system_prompt: str
    plan_instruction: str
    action_instruction: str
    coordinate_instruction: str


_COORDINATE_INSTRUCTION = (
    "For click, scroll, and drag actions, return image-relative integer coordinates on a "
    "1000x1000 grid from 0 through 999. (0,0) is the image's top-left pixel and "
    "(999,999) is its bottom-right pixel. Do not add the desktop origin; the application "
    "converts these coordinates to absolute desktop pixels."
)

PROMPT_PROFILES: Mapping[str, PromptProfile] = MappingProxyType(
    {
        "week4-baseline": PromptProfile(
            id="week4-baseline",
            system_prompt="You are a careful desktop GUI planning assistant.",
            plan_instruction=(
                "Create a short desktop task plan from the current visual observation."
            ),
            action_instruction=(
                "Choose exactly one next desktop action from the current visual observation."
            ),
            coordinate_instruction=_COORDINATE_INSTRUCTION,
        ),
        "week5-grounded": PromptProfile(
            id="week5-grounded",
            system_prompt=(
                "You are a cautious GUI agent that grounds decisions in the supplied image."
            ),
            plan_instruction=(
                "Create a short plan using only controls supported by visible image evidence."
            ),
            action_instruction=(
                "Choose exactly one allowed action and ground every pointer action in visible "
                "image evidence; never invent an unseen UI element."
            ),
            coordinate_instruction=_COORDINATE_INSTRUCTION,
        ),
    }
)


def get_prompt_profile(profile: str | PromptProfile) -> PromptProfile:
    if isinstance(profile, PromptProfile):
        return profile
    try:
        return PROMPT_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"unknown prompt profile: {profile}") from error


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


def build_plan_prompt(
    goal: str,
    observation: Observation,
    *,
    profile: str | PromptProfile = "week4-baseline",
) -> str:
    selected = get_prompt_profile(profile)
    safe_goal = _safe_text(goal, max_length=1000)
    allowed = ", ".join(_ALLOWED_ACTIONS)
    return (
        f"{selected.plan_instruction}\n"
        f"Prompt profile: {selected.id}.\n"
        f"Goal: {safe_goal}\n"
        f"Allowed action kinds: {allowed}.\n"
        "Use stable step IDs and concise descriptions. Return only the structured plan.\n"
        f"{_observation_summary(observation)}"
    )


def build_action_prompt(
    state: AgentState,
    *,
    profile: str | PromptProfile = "week4-baseline",
) -> str:
    selected = get_prompt_profile(profile)
    descriptions = {step.id: step.description for step in state.plan.steps}
    completed_lines = "\n".join(
        f"- {step.step_id}: {_safe_text(descriptions[step.step_id], max_length=500)}"
        for step in state.progress.steps
        if step.status == "completed"
    )
    if not completed_lines:
        completed_lines = "- none"
    active_progress = next(
        step
        for step in state.progress.steps
        if step.step_id == state.progress.active_step_id
    )
    active_step = (
        f"{active_progress.step_id}: "
        f"{_safe_text(descriptions[active_progress.step_id], max_length=500)} "
        f"(status={active_progress.status}, attempts={active_progress.attempts})"
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
    failure_line = (
        f"{state.replan_context.reason_code}: "
        f"{_safe_text(state.replan_context.summary, max_length=500)}"
        if state.replan_context is not None
        else "none"
    )
    allowed = ", ".join(_ALLOWED_ACTIONS)
    return (
        f"{selected.action_instruction}\n"
        f"Prompt profile: {selected.id}.\n"
        f"Goal: {_safe_text(state.goal, max_length=1000)}\n"
        f"Completed steps:\n{completed_lines}\n"
        f"Active step: {active_step}\n"
        f"Failure reason: {failure_line}\n"
        f"Recent results:\n{result_lines}\n"
        f"Allowed action kinds: {allowed}.\n"
        "Return a concise rationale summary, expected outcome, and one structured action.\n"
        f"{_observation_summary(state.observation)}"
    )


def build_replan_prompt(
    state: AgentState,
    failure: ReplanContext,
    *,
    profile: str | PromptProfile = "week4-baseline",
) -> str:
    selected = get_prompt_profile(profile)
    descriptions = {step.id: step.description for step in state.plan.steps}
    completed = "\n".join(
        f"- {step.step_id}: {_safe_text(descriptions[step.step_id], max_length=500)}"
        for step in state.progress.steps
        if step.status == "completed"
    ) or "- none"
    active = next(
        step
        for step in state.progress.steps
        if step.step_id == state.progress.active_step_id
    )
    recent = "\n".join(
        f"- {result.status}: {_safe_text(result.message, max_length=500)}"
        for result in state.results[-3:]
    ) or "- none"
    return (
        f"{selected.plan_instruction}\n"
        "Revise only the unfinished portion of the plan. Never remove or rewrite completed "
        "steps, and keep stable IDs for unchanged unfinished steps.\n"
        f"Prompt profile: {selected.id}.\n"
        f"Goal: {_safe_text(state.goal, max_length=1000)}\n"
        f"Completed steps:\n{completed}\n"
        f"Active step: {active.step_id}: "
        f"{_safe_text(descriptions[active.step_id], max_length=500)}\n"
        f"Failure reason: {failure.reason_code}: "
        f"{_safe_text(failure.summary, max_length=500)}\n"
        f"Recent results:\n{recent}\n"
        f"{_observation_summary(state.observation)}"
    )


__all__ = [
    "PROMPT_PROFILES",
    "PromptProfile",
    "build_action_prompt",
    "build_plan_prompt",
    "build_replan_prompt",
    "get_prompt_profile",
]
