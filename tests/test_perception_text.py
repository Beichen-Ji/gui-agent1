from gui_agent.perception.text import normalize_text


def test_normalize_text_collapses_whitespace_and_casefolds() -> None:
    assert normalize_text("  Save\n  FILE  ") == "save file"
