"""Smoke test: every dashboard page renders without an exception (headless)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.mark.parametrize("page", [None, "forecast", "analysis", "planner"])
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if page:
        at.switch_page(f"pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
