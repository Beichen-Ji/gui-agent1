from pathlib import Path

import pytest

from gui_agent.simulation.state import TestbedState


def test_state_supports_all_five_application_workflows(tmp_path: Path) -> None:
    state = TestbedState(tmp_path / "desktop")

    state.focus("browser.search")
    state.type_text("week7 evaluation")
    state.click("browser.search_button")

    state.activate_app("files")
    state.focus("files.filename")
    state.hotkey(("ctrl", "a"))
    state.type_text("week4-demo.txt")
    state.click("files.open_button")
    state.scroll(-3)

    state.activate_app("messages")
    state.focus("messages.recipient")
    state.type_text("local-test-user")
    state.focus("messages.body")
    state.type_text("simulation ready")
    state.click("messages.send_button")

    state.activate_app("settings")
    state.click("settings.checkbox")
    state.click("settings.theme")
    state.drag("settings.volume", 0.75)
    state.click("settings.save_button")

    state.activate_app("editor")
    state.focus("editor.text")
    state.type_text("draft")
    state.hotkey(("ctrl", "a"))
    state.type_text("final text")
    state.hotkey(("ctrl", "s"))
    state.scroll(-2)

    snapshot = state.snapshot()
    assert snapshot["search_result"] == "Search result: week7 evaluation"
    assert snapshot["opened_file"] == "week4-demo.txt"
    assert snapshot["file_scroll"] == -3
    assert snapshot["messages"] == ("local-test-user: simulation ready",)
    assert snapshot["settings_enabled"] is True
    assert snapshot["settings_theme"] == "dark"
    assert snapshot["settings_volume"] == pytest.approx(0.75)
    assert snapshot["settings_saved"] is True
    assert snapshot["editor_text"] == "final text"
    assert snapshot["editor_saved"] is True
    assert snapshot["editor_scroll"] == -2


def test_state_rejects_invalid_controls_inputs_and_file_escape(tmp_path: Path) -> None:
    state = TestbedState(tmp_path / "desktop")

    with pytest.raises(ValueError, match="application"):
        state.activate_app("calculator")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="active application"):
        state.focus("files.filename")
    with pytest.raises(ValueError, match="focused"):
        state.type_text("orphan text")
    with pytest.raises(ValueError, match="inside the testbed directory"):
        state.open_file("../outside.txt")
    with pytest.raises(ValueError, match="1 to 500"):
        state.send_message(" ")
    with pytest.raises(ValueError, match="drag value"):
        state.drag("settings.volume", 1.1)


@pytest.mark.parametrize("profile", ["none", "transient", "delayed"])
def test_search_fault_profiles_keep_their_week6_semantics(
    tmp_path: Path,
    profile: str,
) -> None:
    state = TestbedState(
        tmp_path / profile,
        fault_profile=profile,  # type: ignore[arg-type]
    )

    first = state.search("fault check")
    if profile == "transient":
        assert first == "Search result: transient action ignored"
        assert state.search("fault check") == "Search result: fault check"
    elif profile == "delayed":
        assert first == "Search result: pending"
        state.wait(0.75)
        assert state.snapshot()["search_result"] == "Search result: fault check"
    else:
        assert first == "Search result: fault check"
