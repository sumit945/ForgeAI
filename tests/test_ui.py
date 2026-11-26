"""Headless smoke test of the Streamlit UI (no Ollama needed: only renders pages)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parents[1] / "forgeai" / "ui.py"


def test_ui_renders_without_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEAI_ROOT", str(tmp_path))
    at = AppTest.from_file(str(UI), default_timeout=30).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 3
    assert any("Build agent" in b.label for b in at.button)
