"""Keep the suite off the operator's real state/, Chroma store, and outputs/.

``master_agent.config`` reads these at import time, so the redirect happens
when pytest loads this file, before any test module imports the package.
Tracked fixtures under ``state/`` (object_info cache, model inventory) are
copied in so tests that read them still see the shipped snapshot.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_REPO_STATE = Path(__file__).resolve().parent.parent / "state"
_TRACKED_STATE_FILES = ("object_info.json", "model_inventory.json")

_root = Path(tempfile.mkdtemp(prefix="vb-pytest-"))
_state = _root / "state"
_state.mkdir()
for name in _TRACKED_STATE_FILES:
    src = _REPO_STATE / name
    if src.is_file():
        shutil.copy2(src, _state / name)

os.environ.setdefault("STATE_DIR", str(_state))
os.environ.setdefault("CHROMA_DIR", str(_state / "chroma"))
os.environ.setdefault("OUTPUTS_DIR", str(_root / "outputs"))


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    shutil.rmtree(_root, ignore_errors=True)
