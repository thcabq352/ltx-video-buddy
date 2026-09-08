#!/usr/bin/env python3
"""VIDEO BUDDY installer — run with system Python 3.10+ on Windows, macOS, or Linux.

Creates .venv, installs requirements, then checks/installs ffmpeg, Playwright,
.env, and Ollama models.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MIN_PY = (3, 10)
ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQ = ROOT / "requirements.txt"


def _die(msg: str, code: int = 1) -> None:
    print(f"FAIL  {msg}")
    raise SystemExit(code)


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def main() -> int:
    print("VIDEO BUDDY installer")
    if sys.version_info[:2] < MIN_PY:
        _die(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ required (found {sys.version.split()[0]})")
    os.chdir(ROOT)
    py = _venv_python()
    if not (VENV / "pyvenv.cfg").is_file():
        print(f"creating {VENV}")
        code = subprocess.call([sys.executable, "-m", "venv", str(VENV)])
        if code != 0:
            _die("could not create .venv")
    if not py.is_file():
        _die(f"venv python missing: {py}")
    print("upgrading pip…")
    subprocess.call([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    print("installing requirements…")
    code = subprocess.call([str(py), "-m", "pip", "install", "-r", str(REQ)])
    if code != 0:
        _die("pip install failed")
    extra = [a for a in sys.argv[1:] if a in ("--fix", "--check")]
    if "--check" not in extra:
        extra = ["--fix"]
    return subprocess.call([str(py), "-m", "master_agent", "setup", *extra])


if __name__ == "__main__":
    raise SystemExit(main())
