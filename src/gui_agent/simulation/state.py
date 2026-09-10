"""Maintain deterministic local state for simulated desktop applications."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias

from gui_agent.simulation.apps import APP_IDS, CONTROL_BY_ID, AppId

DEFAULT_TESTBED_ROOT = Path.cwd() / "artifacts" / "testbed"
DEMO_FILENAME = "week4-demo.txt"
DEMO_CONTENT = "WEEK4_DEMO_READY\nThis file belongs to the local Week 4 GUI testbed.\n"
FaultProfile: TypeAlias = Literal["none", "transient", "delayed"]
FAULT_PROFILES: tuple[FaultProfile, ...] = ("none", "transient", "delayed")


class TestbedState:
    """Pure local state shared by the Tk testbed and headless simulation."""

    __test__ = False

    def __init__(self, root: Path, *, fault_profile: FaultProfile = "none") -> None:
        """Initialize isolated testbed files and optional deterministic faults."""
        if fault_profile not in FAULT_PROFILES:
            raise ValueError(f"unknown fault profile: {fault_profile}")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        demo = self.root / DEMO_FILENAME
        if not demo.exists():
            demo.write_text(DEMO_CONTENT, encoding="utf-8")

        self.active_app: AppId = "browser"
        self.focused_control: str | None = None
        self.browser_open = False
        self.browser_tab: Literal["web", "history"] = "web"
        self.search_query: str | None = None
        self.search_result: str | None = None
        self.opened_file: str | None = None
        self.file_content: str | None = None
        self.file_scroll = 0
        self.messages: list[str] = []
        self.settings_enabled = False
        self.settings_theme: Literal["light", "dark"] = "light"
        self.settings_volume = 0.5
        self.settings_saved = False
        self.editor_text = ""
        self.editor_find = ""
        self.editor_saved = False
        self.editor_scroll = 0
        self.closed = False
        self.fault_profile = fault_profile
        self.faults_triggered = 0
        self._transient_search_ignored = False
        self._selection_all = False
        self._text_values: dict[str, str] = {
            "browser.address": "local://testbed",
            "browser.search": "",
            "files.filename": DEMO_FILENAME,
            "messages.recipient": "",
            "messages.body": "",
            "editor.text": "",
            "editor.find": "",
        }

    def activate_app(self, app_id: AppId) -> None:
        """Activate an application and clear its previous text focus."""
        if app_id not in APP_IDS:
            raise ValueError(f"unknown application: {app_id}")
        self.active_app = app_id
        self.focused_control = None
        self._selection_all = False
        if app_id == "browser":
            self.open_browser()

    def open_browser(self) -> None:
        """Mark the simulated browser as open."""
        self.browser_open = True

    def focus(self, control_id: str) -> None:
        """Focus a textbox belonging to the active simulated application."""
        control = CONTROL_BY_ID.get(control_id)
        if control is None:
            raise ValueError(f"unknown control: {control_id}")
        if not control_id.startswith(f"{self.active_app}."):
            raise ValueError("control does not belong to the active application")
        if control.role != "textbox":
            raise ValueError("only text boxes can receive text focus")
        self.focused_control = control_id
        self._selection_all = False

    def type_text(self, text: str) -> None:
        """Append or replace bounded text in the focused control."""
        if self.focused_control is None:
            raise ValueError("text input requires a focused control")
        if not text or len(text) > 500:
            raise ValueError("text must contain 1 to 500 characters")
        current = self._text_values[self.focused_control]
        updated = text if self._selection_all else current + text
        if len(updated) > 500:
            raise ValueError("focused text value may contain at most 500 characters")
        self._text_values[self.focused_control] = updated
        self._selection_all = False
        self._sync_text_fields()

    def click(self, control_id: str) -> None:
        """Apply the deterministic behavior of one active-app control."""
        if control_id.startswith("app."):
            self.activate_app(control_id.removeprefix("app."))  # type: ignore[arg-type]
            return
        control = CONTROL_BY_ID.get(control_id)
        if control is None or not control_id.startswith(f"{self.active_app}."):
            raise ValueError("control does not belong to the active application")
        if not control.interactive:
            raise ValueError("control is not interactive")
        if control.role == "textbox":
            self.focus(control_id)
        elif control_id == "browser.search_button":
            self.search(self._text_values["browser.search"])
        elif control_id == "browser.web_tab":
            self.browser_tab = "web"
        elif control_id == "browser.history_tab":
            self.browser_tab = "history"
        elif control_id == "files.open_button":
            self.open_file(self._text_values["files.filename"])
        elif control_id == "messages.send_button":
            self._send_composed_message()
        elif control_id == "settings.checkbox":
            self.settings_enabled = not self.settings_enabled
            self.settings_saved = False
        elif control_id == "settings.theme":
            self.settings_theme = "dark" if self.settings_theme == "light" else "light"
            self.settings_saved = False
        elif control_id == "settings.save_button":
            self.settings_saved = True
        elif control_id == "settings.cancel_button":
            self.settings_enabled = False
            self.settings_theme = "light"
            self.settings_volume = 0.5
            self.settings_saved = False
        elif control.role in {"list", "slider"}:
            self.focused_control = control_id

    def search(self, query: str) -> str:
        """Run a local deterministic search with optional injected faults."""
        normalized = query.strip()
        if not normalized or len(normalized) > 200:
            raise ValueError("search query must contain 1 to 200 characters")
        self.open_browser()
        self.search_query = normalized
        self._text_values["browser.search"] = normalized
        if self.fault_profile == "transient" and not self._transient_search_ignored:
            self._transient_search_ignored = True
            self.faults_triggered += 1
            self.search_result = "Search result: transient action ignored"
            return self.search_result
        if self.fault_profile == "delayed":
            self.faults_triggered += 1
            self.search_result = "Search result: pending"
            return self.search_result
        self.search_result = f"Search result: {normalized}"
        return self.search_result

    def complete_delayed_search(self) -> str:
        """Resolve one pending delayed search result."""
        if self.fault_profile != "delayed" or self.search_query is None:
            raise ValueError("no delayed search is pending")
        self.search_result = f"Search result: {self.search_query}"
        return self.search_result

    def open_file(self, filename: str) -> str:
        """Read a UTF-8 file only from inside the isolated testbed root."""
        candidate = (self.root / filename).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("files must stay inside the testbed directory")
        if not candidate.is_file():
            raise ValueError(f"testbed file does not exist: {filename}")
        content = candidate.read_text(encoding="utf-8")
        self.opened_file = candidate.name
        self.file_content = content
        self._text_values["files.filename"] = candidate.name
        return content

    def send_message(self, text: str) -> None:
        """Append one bounded message to the local simulated inbox."""
        if not text.strip() or len(text) > 500:
            raise ValueError("message must contain 1 to 500 characters")
        self.messages.append(text)

    def _send_composed_message(self) -> None:
        recipient = self._text_values["messages.recipient"].strip()
        body = self._text_values["messages.body"].strip()
        if not recipient or not body:
            raise ValueError("recipient and message body must not be blank")
        self.send_message(f"{recipient}: {body}")
        self._text_values["messages.body"] = ""

    def hotkey(self, keys: Sequence[str]) -> None:
        """Apply a supported active-app hotkey to simulated state."""
        normalized = tuple(key.strip().lower() for key in keys)
        if not normalized or any(not key for key in normalized):
            raise ValueError("hotkey must contain non-blank keys")
        if normalized == ("ctrl", "a"):
            if self.focused_control not in self._text_values:
                raise ValueError("Ctrl+A requires a focused text control")
            self._selection_all = True
            return
        if normalized == ("ctrl", "s") and self.active_app == "editor":
            self.editor_saved = True
            return
        if normalized in {("left",), ("right",)} and self.active_app == "settings":
            delta = -0.1 if normalized == ("left",) else 0.1
            self.settings_volume = min(1.0, max(0.0, self.settings_volume + delta))
            self.settings_saved = False
            return
        raise ValueError(f"unsupported hotkey for {self.active_app}: {'+'.join(normalized)}")

    def drag(self, control_id: str, value: float) -> None:
        """Set the simulated settings volume using a normalized drag value."""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("drag value must be between 0.0 and 1.0")
        if self.active_app != "settings" or control_id != "settings.volume":
            raise ValueError("drag is only supported by the active settings volume")
        self.settings_volume = float(value)
        self.settings_saved = False

    def scroll(self, clicks: int) -> None:
        """Scroll the active files or editor view within bounded state."""
        if isinstance(clicks, bool) or not isinstance(clicks, int) or not -20 <= clicks <= 20:
            raise ValueError("scroll clicks must be an integer between -20 and 20")
        if self.active_app == "files":
            self.file_scroll = min(100, max(-100, self.file_scroll + clicks))
        elif self.active_app == "editor":
            self.editor_scroll = min(100, max(-100, self.editor_scroll + clicks))
        else:
            raise ValueError("scroll is only supported by files and editor")

    def wait(self, seconds: float) -> None:
        """Advance delayed behavior without sleeping or real desktop input."""
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not 0.0 <= seconds <= 5.0
        ):
            raise ValueError("wait seconds must be between 0.0 and 5.0")
        if self.fault_profile == "delayed" and self.search_result == "Search result: pending":
            self.complete_delayed_search()

    def close(self) -> None:
        """Mark the local simulated testbed as closed."""
        self.closed = True

    def text_value(self, control_id: str) -> str:
        """Return text stored by a simulated text control."""
        try:
            return self._text_values[control_id]
        except KeyError as error:
            raise ValueError(f"control does not contain text: {control_id}") from error

    @property
    def selection_all(self) -> bool:
        """Return whether the focused simulated text is selected."""
        return self._selection_all

    def snapshot(self) -> dict[str, object]:
        """Return a serializable snapshot of externally visible testbed state."""
        return {
            "active_app": self.active_app,
            "focused_control": self.focused_control,
            "browser_open": self.browser_open,
            "browser_tab": self.browser_tab,
            "search_query": self.search_query,
            "search_result": self.search_result,
            "opened_file": self.opened_file,
            "file_content": self.file_content,
            "file_scroll": self.file_scroll,
            "messages": tuple(self.messages),
            "settings_enabled": self.settings_enabled,
            "settings_theme": self.settings_theme,
            "settings_volume": self.settings_volume,
            "settings_saved": self.settings_saved,
            "editor_text": self.editor_text,
            "editor_find": self.editor_find,
            "editor_saved": self.editor_saved,
            "editor_scroll": self.editor_scroll,
            "closed": self.closed,
            "fault_profile": self.fault_profile,
            "faults_triggered": self.faults_triggered,
        }

    def _sync_text_fields(self) -> None:
        self.editor_text = self._text_values["editor.text"]
        self.editor_find = self._text_values["editor.find"]


__all__ = [
    "DEFAULT_TESTBED_ROOT",
    "DEMO_CONTENT",
    "DEMO_FILENAME",
    "FAULT_PROFILES",
    "FaultProfile",
    "TestbedState",
]
