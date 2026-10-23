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
# Keep LLM discovery off the operator's real ~/.ollama blobs. Tests that want
# a blob set OLLAMA_MODELS themselves. Read at call time, so this sticks.
os.environ["OLLAMA_MODELS"] = str(_root / "ollama-models")
# Existing submits are not the first-launch sequence. Seat tests opt back in.
os.environ.setdefault("FIRST_GATE", "passed")
os.environ.setdefault("SEAT_MIN_FREE_GB", "0")


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    shutil.rmtree(_root, ignore_errors=True)


import pytest  # noqa: E402

from master_agent.comfy import layout as _comfy_layout  # noqa: E402

_REAL_PORTABLE_SEARCH_ROOTS = _comfy_layout.portable_search_roots


@pytest.fixture(autouse=True)
def _reset_gpu_seat():
    from master_agent.gpu_seat import reset_state

    reset_state()
    yield
    reset_state()


@pytest.fixture(autouse=True)
def _no_real_portable_discovery(monkeypatch):
    """Never auto-detect the operator's real ComfyUI_windows_portable next to the repo."""
    monkeypatch.setattr(_comfy_layout, "portable_search_roots", lambda: [])


@pytest.fixture
def real_portable_search(monkeypatch):
    """Opt back in to the real search (config.PORTABLE_ROOT + PROJECT_ROOT glob)."""
    monkeypatch.setattr(_comfy_layout, "portable_search_roots", _REAL_PORTABLE_SEARCH_ROOTS)
