"""Saved "LTX 2.3 weights deferred" marker (``state/weights_deferred.json``).

``automatic-install --skip-weights`` writes it so the install, and every later
doctor run, reports the missing LTX 2.3 pack as INFO instead of a required
FAIL. Downloading the pack or ``models register`` completing it clears it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

BUNDLE = "ltx23_core"
FILENAME = "weights_deferred.json"
REGISTER_CMD = "python -m master_agent models register <file-or-folder>"
DOWNLOAD_CMD = "python -m master_agent download-models --ltx23 --yes"
REASON = "you deferred the LTX 2.3 weights (--skip-weights); only generate needs them"


def marker_path() -> Path:
    from master_agent.comfy.tower import state_path

    return state_path().parent / FILENAME


def read_marker() -> dict[str, Any] | None:
    path = marker_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def weights_deferred() -> bool:
    return read_marker() is not None


def mark_deferred(missing: list[str], *, reason: str = "--skip-weights") -> Path:
    path = marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle": BUNDLE,
        "reason": reason,
        "deferred_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "missing": list(missing),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def clear_deferred() -> bool:
    try:
        marker_path().unlink()
    except FileNotFoundError:
        return False
    return True
