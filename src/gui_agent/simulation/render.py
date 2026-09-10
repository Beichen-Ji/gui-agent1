"""Lay out and render deterministic simulated desktops as uint8 BGR images."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gui_agent.simulation.apps import ControlDefinition, controls_for
from gui_agent.simulation.state import TestbedState
from gui_agent.types import BoundingBox, ImageArray

_BACKGROUND_RGB = (240, 243, 245)
_SURFACE_RGB = (255, 255, 255)
_BORDER_RGB = (77, 89, 102)
_ACTIVE_RGB = (214, 232, 255)
_TEXT_RGB = (24, 32, 40)
_MUTED_RGB = (228, 232, 236)


@dataclass(frozen=True, slots=True)
class LayoutControl:
    """Store one control's pixel geometry and fitted label layout."""

    id: str
    role: str
    label: str
    box: BoundingBox
    label_box: BoundingBox
    font_size: int
    interactive: bool


def _absolute_box(control: ControlDefinition, canvas: tuple[int, int]) -> BoundingBox:
    width, height = canvas
    return BoundingBox(
        left=round(control.box.left * width),
        top=round(control.box.top * height),
        right=round(control.box.right * width),
        bottom=round(control.box.bottom * height),
    )


def _display_label(state: TestbedState, control: ControlDefinition) -> str:
    dynamic = {
        "browser.address": state.text_value("browser.address") or control.label,
        "browser.search": state.text_value("browser.search") or control.label,
        "browser.result": state.search_result or control.label,
        "browser.web_tab": "Web (selected)" if state.browser_tab == "web" else "Web",
        "browser.history_tab": (
            "History (selected)" if state.browser_tab == "history" else "History"
        ),
        "files.filename": state.text_value("files.filename") or control.label,
        "files.list": f"Files (scroll={state.file_scroll})\n{DEMO_LIST_ENTRY}",
        "files.content": state.file_content or control.label,
        "messages.recipient": state.text_value("messages.recipient") or control.label,
        "messages.body": state.text_value("messages.body") or control.label,
        "messages.inbox": (
            "Inbox" if not state.messages else "Inbox\n" + "\n".join(state.messages[-3:])
        ),
        "settings.checkbox": f"[{'x' if state.settings_enabled else ' '}] Enable feature",
        "settings.theme": f"Theme: {state.settings_theme}",
        "settings.volume": f"Volume: {round(state.settings_volume * 100)}%",
        "settings.save_button": "Saved" if state.settings_saved else "Save",
        "editor.text": state.editor_text or control.label,
        "editor.find": state.editor_find or control.label,
        "editor.status": (
            f"{'Saved' if state.editor_saved else control.label} | scroll={state.editor_scroll}"
        ),
    }
    label = dynamic.get(control.id, control.label)
    if state.selection_all and control.id == state.focused_control:
        return f"{label} [selected]"
    return label


DEMO_LIST_ENTRY = "week4-demo.txt"


def _fit_label(
    label: str,
    box: BoundingBox,
    base_size: int,
) -> tuple[BoundingBox, int]:
    padding = max(4, base_size // 2)
    available_width = box.right - box.left - padding * 2
    available_height = box.bottom - box.top - padding * 2
    for font_size in range(base_size, 7, -1):
        font = ImageFont.load_default(size=font_size)
        left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
            (0, 0),
            label,
            font=font,
            spacing=max(2, font_size // 4),
        )
        text_width = round(right - left)
        text_height = round(bottom - top)
        if text_width <= available_width and text_height <= available_height:
            label_left = box.left + padding
            label_top = box.top + max(padding, (box.bottom - box.top - text_height) // 2)
            return (
                BoundingBox(
                    label_left,
                    label_top,
                    label_left + max(1, text_width),
                    label_top + max(1, text_height),
                ),
                font_size,
            )
    raise ValueError(f"control label does not fit inside its box: {label!r}")


def layout_desktop(
    state: TestbedState,
    canvas: tuple[int, int],
) -> tuple[LayoutControl, ...]:
    """Convert normalized active-app controls into deterministic pixel layout."""
    width, height = canvas
    if width < 320 or height < 180:
        raise ValueError("simulation canvas must be at least 320x180")
    base_size = max(14, round(18 * min(width / 1280, height / 720)))
    result: list[LayoutControl] = []
    for control in controls_for(state.active_app):
        box = _absolute_box(control, canvas)
        label = _display_label(state, control)
        label_box, font_size = _fit_label(label, box, base_size)
        result.append(
            LayoutControl(
                id=control.id,
                role=control.role,
                label=label,
                box=box,
                label_box=label_box,
                font_size=font_size,
                interactive=control.interactive,
            )
        )
    return tuple(result)


def render_desktop(
    state: TestbedState,
    canvas: tuple[int, int],
) -> tuple[ImageArray, dict[str, BoundingBox]]:
    """Render state to BGR pixels and return matching interactive hitboxes."""
    width, height = canvas
    image = Image.new("RGB", (width, height), _BACKGROUND_RGB)
    draw = ImageDraw.Draw(image)
    layout = layout_desktop(state, canvas)
    for control in layout:
        active = (
            control.id == state.focused_control
            or control.id == f"app.{state.active_app}"
            or control.id == f"browser.{state.browser_tab}_tab"
        )
        fill = _ACTIVE_RGB if active else (_SURFACE_RGB if control.interactive else _MUTED_RGB)
        draw.rectangle(
            (control.box.left, control.box.top, control.box.right - 1, control.box.bottom - 1),
            fill=fill,
            outline=_BORDER_RGB,
            width=max(1, round(min(width / 1280, height / 720))),
        )
        draw.multiline_text(
            (control.label_box.left, control.label_box.top),
            control.label,
            fill=_TEXT_RGB,
            font=ImageFont.load_default(size=control.font_size),
            spacing=max(2, control.font_size // 4),
        )
    rgb = np.asarray(image, dtype=np.uint8)
    bgr: ImageArray = rgb[:, :, ::-1].copy()
    return bgr, {control.id: control.box for control in layout}


__all__ = ["LayoutControl", "layout_desktop", "render_desktop"]
