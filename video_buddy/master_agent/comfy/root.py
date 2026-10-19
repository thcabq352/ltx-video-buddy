"""Where Buddy's own (managed) ComfyUI lives. Every caller resolves it here.

Precedence, first match wins:

1. an explicit path (``comfy start --workspace <dir>``)
2. ``MANAGED_COMFY_ROOT`` (process env, or ``.env`` loaded by ``config``)
3. the workspace saved in ``state/managed_comfy.json`` by an earlier
   ``comfy start --workspace`` (only an explicit choice is saved, so a moved
   project folder never keeps pointing at the old default). A saved folder with
   no ``main.py`` is stale: it is skipped with a note, not used.
4. ``PROJECT_ROOT/ComfyUI`` (``video_buddy/ComfyUI``) when it has ``main.py``
5. a Windows portable build found under ``video_buddy``
   (``ComfyUI_windows_portable*`` or ``COMFYUI_PORTABLE_ROOT``)
6. ``PROJECT_ROOT/ComfyUI`` anyway, so the "not installed" error names it

A path may name the portable top folder or its inner ``ComfyUI`` folder; both
resolve to the folder holding ``main.py``. The env var beats the saved state so
that setting ``MANAGED_COMFY_ROOT`` after a first ``comfy start`` still moves
every command. This is not ``COMFYUI_ROOT`` / ``EXTERNAL_COMFY_ROOT``: those
name a user-owned install Buddy only reads from.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from master_agent.comfy.layout import code_dir_for, discover_portable

ENV_VAR = "MANAGED_COMFY_ROOT"

SOURCE_FLAG = "--workspace"
SOURCE_ENV = ENV_VAR
SOURCE_STATE = "state/managed_comfy.json"
SOURCE_DEFAULT = "default"
SOURCE_PORTABLE = "auto-detected portable"
# Sources worth remembering across runs; env, default and auto-detected are re-read every time.
SAVED_SOURCES = frozenset({SOURCE_FLAG, SOURCE_STATE})

__all__ = [
    "ComfyRoot",
    "ENV_VAR",
    "default_managed_comfy_root",
    "managed_comfy_root",
    "resolve_managed_comfy_root",
]


@dataclass(frozen=True)
class ComfyRoot:
    path: Path
    source: str
    notes: tuple[str, ...] = ()
    stale_saved: Path | None = None

    def describe(self) -> str:
        if self.source == SOURCE_PORTABLE:
            return f"{self.path} ({self.source})"
        return f"{self.path} (from {self.source})"


def _resolve(raw: Path | str) -> Path:
    return code_dir_for(Path(str(raw).strip().strip('"')).expanduser().resolve())


def _has_main(path: Path) -> bool:
    return (path / "main.py").is_file()


def default_managed_comfy_root() -> Path:
    from master_agent import config

    return _resolve(Path(config.PROJECT_ROOT) / "ComfyUI")


def _env_root() -> Path | None:
    raw = (os.getenv(ENV_VAR) or "").strip().strip('"')
    return _resolve(raw) if raw else None


def _saved_root() -> Path | None:
    from master_agent.comfy.tower import state_path

    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    raw = data.get("workspace") if isinstance(data, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    return _resolve(raw)


def resolve_managed_comfy_root(
    explicit: Path | str | None = None, *, use_saved: bool = True
) -> ComfyRoot:
    if explicit is not None and str(explicit).strip():
        return ComfyRoot(_resolve(explicit), SOURCE_FLAG)
    env = _env_root()
    if env is not None:
        return ComfyRoot(env, SOURCE_ENV)
    notes: list[str] = []
    stale: Path | None = None
    if use_saved:
        saved = _saved_root()
        if saved is not None:
            if _has_main(saved):
                return ComfyRoot(saved, SOURCE_STATE)
            stale = saved
            notes.append(f"ignored stale {SOURCE_STATE} workspace {saved} (no main.py there)")
    default = default_managed_comfy_root()
    if _has_main(default):
        return ComfyRoot(default, SOURCE_DEFAULT, tuple(notes), stale)
    portable = discover_portable()
    if portable is not None:
        return ComfyRoot(portable.code_dir, SOURCE_PORTABLE, tuple(notes), stale)
    return ComfyRoot(default, SOURCE_DEFAULT, tuple(notes), stale)


def managed_comfy_root(explicit: Path | str | None = None) -> Path:
    return resolve_managed_comfy_root(explicit).path
