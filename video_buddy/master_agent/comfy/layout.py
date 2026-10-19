"""What kind of ComfyUI tree a folder is, and how to launch it.

Three layouts:

- ``portable``: the official Windows portable build. A top folder
  (``ComfyUI_windows_portable``) holds ``python_embeded/python.exe`` and the
  code in ``ComfyUI/``. It launches like ``run_nvidia_gpu.bat`` does: cwd is the
  top folder, ``python_embeded\\python.exe -s ComfyUI\\main.py --windows-standalone-build``.
- ``venv``: ``main.py`` with ``.venv`` / ``venv`` next to it (comfy-cli install).
- ``plain``: ``main.py`` and no interpreter of its own; Buddy's python runs it.

Either the portable top folder or its inner ``ComfyUI`` folder may be given;
:func:`code_dir_for` maps both to the folder that holds ``main.py``, which is
what every other module (custom_nodes, models, comfy-cli ``--workspace``) uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EMBEDDED_DIR = "python_embeded"
EMBEDDED_PYTHON = "python.exe"
PORTABLE_CODE_DIR = "ComfyUI"
PORTABLE_GLOB = "ComfyUI_windows_portable*"
PORTABLE_PYTHON_FLAGS = ("-s",)
PORTABLE_COMFY_FLAGS = ("--windows-standalone-build",)

PORTABLE = "portable"
VENV = "venv"
PLAIN = "plain"


@dataclass(frozen=True)
class ComfyLayout:
    kind: str
    code_dir: Path
    top: Path
    python: Path | None

    @property
    def cwd(self) -> Path:
        return self.top

    @property
    def main_script(self) -> str:
        """``main.py`` relative to :attr:`cwd`, in the platform's own separator."""
        if self.top == self.code_dir:
            return "main.py"
        return f"{self.code_dir.name}{os.sep}main.py"

    @property
    def python_flags(self) -> tuple[str, ...]:
        return PORTABLE_PYTHON_FLAGS if self.kind == PORTABLE else ()

    @property
    def comfy_flags(self) -> tuple[str, ...]:
        return PORTABLE_COMFY_FLAGS if self.kind == PORTABLE else ()

    def describe(self) -> str:
        if self.kind == PORTABLE:
            return f"portable build at {self.top} (python {self.python})"
        if self.kind == VENV:
            return f"ComfyUI with its own venv (python {self.python})"
        return "ComfyUI without its own python (runs in Buddy's)"


def embedded_python(top: Path) -> Path:
    return Path(top) / EMBEDDED_DIR / EMBEDDED_PYTHON


def code_dir_for(path: Path | str) -> Path:
    """The folder with ``main.py``: ``path`` itself, or ``path/ComfyUI`` for a portable top folder."""
    p = Path(path)
    if (p / "main.py").is_file():
        return p
    inner = p / PORTABLE_CODE_DIR
    if (inner / "main.py").is_file():
        return inner
    return p


def portable_top(code_dir: Path) -> Path | None:
    """Portable top folder for ``code_dir``: where ``python_embeded/python.exe`` sits."""
    for top in (code_dir.parent, code_dir):
        if embedded_python(top).is_file():
            return top
    return None


def detect_layout(path: Path | str | None) -> ComfyLayout | None:
    """Layout of the ComfyUI tree at ``path``, or None when there is no ``main.py``."""
    if path is None:
        return None
    code = code_dir_for(path)
    if not (code / "main.py").is_file():
        return None
    top = portable_top(code)
    if top is not None:
        return ComfyLayout(PORTABLE, code, top, embedded_python(top))
    from master_agent.comfy.comfy_venv import workspace_venv_python

    venv_py = workspace_venv_python(code)
    if venv_py is not None:
        return ComfyLayout(VENV, code, code, Path(venv_py))
    return ComfyLayout(PLAIN, code, code, None)


def comfy_interpreter(path: Path | str | None) -> Path | None:
    """ComfyUI's own python: the portable ``python_embeded`` or the workspace venv."""
    layout = detect_layout(path)
    if layout is not None:
        return layout.python
    if path is None:
        return None
    from master_agent.comfy.comfy_venv import workspace_venv_python

    return workspace_venv_python(path)


def portable_search_roots() -> list[Path]:
    from master_agent import config

    roots = [Path(config.PORTABLE_ROOT)]
    base = Path(config.PROJECT_ROOT)
    try:
        roots.extend(sorted(p for p in base.glob(PORTABLE_GLOB) if p.is_dir()))
    except OSError:
        pass
    return roots


def discover_portable(roots: Iterable[Path] | None = None) -> ComfyLayout | None:
    """First portable build under ``COMFYUI_PORTABLE_ROOT`` or ``PROJECT_ROOT/ComfyUI_windows_portable*``."""
    seen: set[str] = set()
    for root in roots if roots is not None else portable_search_roots():
        key = os.path.normcase(str(root))
        if key in seen:
            continue
        seen.add(key)
        layout = detect_layout(root)
        if layout is not None and layout.kind == PORTABLE:
            return layout
    return None
