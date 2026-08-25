import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, cast

DEFAULT_TESTBED_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "testbed"
DEMO_FILENAME = "week4-demo.txt"
DEMO_CONTENT = "WEEK4_DEMO_READY\nThis file belongs to the local Week 4 GUI testbed.\n"


class TestbedState:
    """Pure local state for the Browser, Files, and Messages test areas."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        demo = self.root / DEMO_FILENAME
        if not demo.exists():
            demo.write_text(DEMO_CONTENT, encoding="utf-8")
        self.browser_open = False
        self.search_query: str | None = None
        self.search_result: str | None = None
        self.opened_file: str | None = None
        self.file_content: str | None = None
        self.messages: list[str] = []
        self.closed = False

    def open_browser(self) -> None:
        self.browser_open = True

    def search(self, query: str) -> str:
        normalized = query.strip()
        if not normalized or len(normalized) > 200:
            raise ValueError("search query must contain 1 to 200 characters")
        self.open_browser()
        self.search_query = normalized
        self.search_result = f"Search result: {normalized}"
        return self.search_result

    def open_file(self, filename: str) -> str:
        candidate = (self.root / filename).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("files must stay inside the testbed directory")
        if not candidate.is_file():
            raise ValueError(f"testbed file does not exist: {filename}")
        content = candidate.read_text(encoding="utf-8")
        self.opened_file = candidate.name
        self.file_content = content
        return content

    def send_message(self, text: str) -> None:
        if not text.strip() or len(text) > 500:
            raise ValueError("message must contain 1 to 500 characters")
        self.messages.append(text)

    def close(self) -> None:
        self.closed = True

    def snapshot(self) -> dict[str, object]:
        return {
            "browser_open": self.browser_open,
            "search_query": self.search_query,
            "search_result": self.search_result,
            "opened_file": self.opened_file,
            "file_content": self.file_content,
            "messages": tuple(self.messages),
            "closed": self.closed,
        }


class TestbedApp:
    def __init__(self, root: tk.Tk, state: TestbedState) -> None:
        self._root = root
        self._state = state
        root.title("GUI Agent Week 4 Testbed")
        root.geometry("760x520")
        root.minsize(680, 460)

        heading = ttk.Label(
            root,
            text="Local GUI Agent Testbed — no external accounts or network actions",
            font=("Segoe UI", 13, "bold"),
        )
        heading.pack(fill="x", padx=16, pady=(14, 8))

        self._tabs = ttk.Notebook(root)
        self._browser_tab = ttk.Frame(self._tabs, padding=16)
        self._files_tab = ttk.Frame(self._tabs, padding=16)
        self._messages_tab = ttk.Frame(self._tabs, padding=16)
        self._tabs.add(self._browser_tab, text="Browser")
        self._tabs.add(self._files_tab, text="Files")
        self._tabs.add(self._messages_tab, text="Messages")
        self._tabs.pack(fill="both", expand=True, padx=16, pady=8)
        self._tabs.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_browser_tab()
        self._build_files_tab()
        self._build_messages_tab()

        footer = ttk.Frame(root)
        footer.pack(fill="x", padx=16, pady=(4, 14))
        self._status = ttk.Label(footer, text="", anchor="w")
        self._status.pack(side="left", fill="x", expand=True)
        ttk.Button(footer, text="Close Testbed", command=self._close).pack(side="right")
        root.protocol("WM_DELETE_WINDOW", self._close)

        self._state.open_browser()
        self._refresh_status()

    def _build_browser_tab(self) -> None:
        ttk.Label(
            self._browser_tab,
            text="Browser — local deterministic search",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        row = ttk.Frame(self._browser_tab)
        row.pack(fill="x", pady=14)
        self._search_entry = ttk.Entry(row)
        self._search_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Search", command=self._search).pack(side="left", padx=(8, 0))
        self._search_output = ttk.Label(
            self._browser_tab,
            text="Search result: ready",
            anchor="w",
            relief="solid",
            padding=12,
        )
        self._search_output.pack(fill="x")

    def _build_files_tab(self) -> None:
        ttk.Label(
            self._files_tab,
            text="Files — artifacts/testbed only",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        row = ttk.Frame(self._files_tab)
        row.pack(fill="x", pady=14)
        self._file_entry = ttk.Entry(row)
        self._file_entry.insert(0, DEMO_FILENAME)
        self._file_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Open File", command=self._open_file).pack(
            side="left", padx=(8, 0)
        )
        self._file_output = tk.Text(self._files_tab, height=12, wrap="word")
        self._file_output.pack(fill="both", expand=True)

    def _build_messages_tab(self) -> None:
        ttk.Label(
            self._messages_tab,
            text="Messages — in-memory test inbox",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        row = ttk.Frame(self._messages_tab)
        row.pack(fill="x", pady=14)
        self._message_entry = ttk.Entry(row)
        self._message_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Send Message", command=self._send_message).pack(
            side="left", padx=(8, 0)
        )
        self._inbox = tk.Listbox(self._messages_tab, height=12)
        self._inbox.pack(fill="both", expand=True)

    def _on_tab_changed(self, _event: object) -> None:
        tabs = cast(Any, self._tabs)
        if tabs.index(tabs.select()) == 0:
            self._state.open_browser()
        self._refresh_status()

    def _search(self) -> None:
        try:
            result = self._state.search(self._search_entry.get())
        except ValueError as error:
            result = f"Error: {error}"
        self._search_output.configure(text=result)
        self._refresh_status()

    def _open_file(self) -> None:
        try:
            content = self._state.open_file(self._file_entry.get())
        except (OSError, UnicodeError, ValueError) as error:
            content = f"Error: {error}"
        self._file_output.delete("1.0", tk.END)
        self._file_output.insert("1.0", content)
        self._refresh_status()

    def _send_message(self) -> None:
        text = self._message_entry.get()
        try:
            self._state.send_message(text)
        except ValueError as error:
            self._inbox.insert(tk.END, f"Error: {error}")
        else:
            self._inbox.insert(tk.END, text)
            self._message_entry.delete(0, tk.END)
        self._refresh_status()

    def _refresh_status(self) -> None:
        snapshot = self._state.snapshot()
        self._status.configure(
            text=(
                f"STATUS browser_open={snapshot['browser_open']} "
                f"search={snapshot['search_query']!r} "
                f"file={snapshot['opened_file']!r} "
                f"messages={len(self._state.messages)}"
            )
        )

    def _close(self) -> None:
        self._state.close()
        self._root.destroy()


def main() -> int:
    root = tk.Tk()
    TestbedApp(root, TestbedState(DEFAULT_TESTBED_ROOT))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
