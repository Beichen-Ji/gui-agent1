"""Define deterministic simulated applications and normalized controls."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

AppId: TypeAlias = Literal["browser", "files", "messages", "settings", "editor"]
ControlRole: TypeAlias = Literal[
    "button",
    "checkbox",
    "combobox",
    "list",
    "panel",
    "slider",
    "status",
    "tab",
    "textbox",
]

APP_IDS: tuple[AppId, ...] = (
    "browser",
    "files",
    "messages",
    "settings",
    "editor",
)


@dataclass(frozen=True, slots=True)
class RelativeBox:
    """Represent a positive control box in normalized canvas coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        """Validate ordered numeric coordinates within the unit square."""
        values = (self.left, self.top, self.right, self.bottom)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("relative coordinates must be numbers")
        if not (0.0 <= self.left < self.right <= 1.0):
            raise ValueError("relative horizontal coordinates must be inside 0.0-1.0")
        if not (0.0 <= self.top < self.bottom <= 1.0):
            raise ValueError("relative vertical coordinates must be inside 0.0-1.0")


@dataclass(frozen=True, slots=True)
class ControlDefinition:
    """Describe one stable simulated UI control."""

    id: str
    role: ControlRole
    box: RelativeBox
    label: str
    interactive: bool = True

    def __post_init__(self) -> None:
        """Require non-blank control identity and label."""
        if not self.id.strip() or not self.label.strip():
            raise ValueError("control ID and label must not be blank")


@dataclass(frozen=True, slots=True)
class ApplicationDefinition:
    """Group uniquely identified controls under one simulated application."""

    id: AppId
    label: str
    controls: tuple[ControlDefinition, ...]

    def __post_init__(self) -> None:
        """Validate unique, application-prefixed control identifiers."""
        ids = [control.id for control in self.controls]
        if len(ids) != len(set(ids)):
            raise ValueError(f"control IDs must be unique for application {self.id}")
        if any(not control.id.startswith(f"{self.id}.") for control in self.controls):
            raise ValueError("application control IDs must use the application prefix")


def _control(
    control_id: str,
    role: ControlRole,
    box: tuple[float, float, float, float],
    label: str,
    *,
    interactive: bool = True,
) -> ControlDefinition:
    return ControlDefinition(
        id=control_id,
        role=role,
        box=RelativeBox(*box),
        label=label,
        interactive=interactive,
    )


APP_TABS: tuple[ControlDefinition, ...] = tuple(
    _control(
        f"app.{app_id}",
        "tab",
        (0.03 + index * 0.19, 0.03, 0.19 + index * 0.19, 0.10),
        app_id.title(),
    )
    for index, app_id in enumerate(APP_IDS)
)


APPLICATIONS: tuple[ApplicationDefinition, ...] = (
    ApplicationDefinition(
        id="browser",
        label="Browser",
        controls=(
            _control("browser.address", "textbox", (0.05, 0.15, 0.95, 0.22), "Address"),
            _control("browser.web_tab", "tab", (0.05, 0.25, 0.18, 0.31), "Web"),
            _control("browser.history_tab", "tab", (0.20, 0.25, 0.35, 0.31), "History"),
            _control("browser.search", "textbox", (0.05, 0.35, 0.78, 0.43), "Search query"),
            _control(
                "browser.search_button",
                "button",
                (0.81, 0.35, 0.95, 0.43),
                "Search",
            ),
            _control(
                "browser.result",
                "panel",
                (0.05, 0.48, 0.95, 0.67),
                "Search result: ready",
                interactive=False,
            ),
        ),
    ),
    ApplicationDefinition(
        id="files",
        label="Files",
        controls=(
            _control("files.filename", "textbox", (0.05, 0.15, 0.78, 0.23), "File name"),
            _control("files.open_button", "button", (0.81, 0.15, 0.95, 0.23), "Open"),
            _control("files.list", "list", (0.05, 0.28, 0.42, 0.80), "Files"),
            _control(
                "files.content",
                "panel",
                (0.45, 0.28, 0.95, 0.80),
                "File content",
                interactive=False,
            ),
        ),
    ),
    ApplicationDefinition(
        id="messages",
        label="Messages",
        controls=(
            _control(
                "messages.recipient",
                "textbox",
                (0.05, 0.15, 0.95, 0.23),
                "Recipient",
            ),
            _control("messages.body", "textbox", (0.05, 0.28, 0.78, 0.48), "Message"),
            _control("messages.send_button", "button", (0.81, 0.28, 0.95, 0.38), "Send"),
            _control("messages.inbox", "list", (0.05, 0.54, 0.95, 0.82), "Inbox"),
        ),
    ),
    ApplicationDefinition(
        id="settings",
        label="Settings",
        controls=(
            _control(
                "settings.checkbox",
                "checkbox",
                (0.05, 0.18, 0.32, 0.25),
                "Enable feature",
            ),
            _control("settings.theme", "combobox", (0.05, 0.32, 0.40, 0.40), "Theme"),
            _control("settings.volume", "slider", (0.05, 0.47, 0.70, 0.54), "Volume"),
            _control("settings.save_button", "button", (0.72, 0.67, 0.82, 0.75), "Save"),
            _control("settings.cancel_button", "button", (0.84, 0.67, 0.95, 0.75), "Cancel"),
        ),
    ),
    ApplicationDefinition(
        id="editor",
        label="Editor",
        controls=(
            _control("editor.text", "textbox", (0.05, 0.15, 0.95, 0.64), "Editor text"),
            _control("editor.find", "textbox", (0.05, 0.70, 0.55, 0.78), "Find"),
            _control(
                "editor.status",
                "status",
                (0.58, 0.70, 0.95, 0.78),
                "Ctrl+A select all | Ctrl+S save",
                interactive=False,
            ),
        ),
    ),
)

APPLICATION_BY_ID = {application.id: application for application in APPLICATIONS}
CONTROL_BY_ID = {
    control.id: control
    for application in APPLICATIONS
    for control in application.controls
}
CONTROL_BY_ID.update({control.id: control for control in APP_TABS})


def controls_for(app_id: AppId) -> tuple[ControlDefinition, ...]:
    """Return application tabs followed by controls for the active app."""
    return APP_TABS + APPLICATION_BY_ID[app_id].controls


__all__ = [
    "APPLICATIONS",
    "APPLICATION_BY_ID",
    "APP_IDS",
    "APP_TABS",
    "CONTROL_BY_ID",
    "AppId",
    "ApplicationDefinition",
    "ControlDefinition",
    "ControlRole",
    "RelativeBox",
    "controls_for",
]
