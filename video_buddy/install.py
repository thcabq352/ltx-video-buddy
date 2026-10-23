#!/usr/bin/env python3
"""VIDEO BUDDY installer — run with system Python 3.10+ on Windows, macOS, or Linux.

Plain ``python install.py`` is the LIGHT install: .venv, requirements,
Playwright, .env, ffmpeg and the empty model folders (``setup --fix``). It does
not install ComfyUI or download weights, and says so in a banner at the start
and the end. Local LLM GGUFs with no public source are skipped; Ollama is never
installed or pulled.

    python install.py --check

only reports (``setup``): no venv, no pip, no installs. It needs an existing .venv.

    python install.py --automatic-install --yes

runs automatic install instead (pre-flight, then ComfyUI + LTX 2.3 + llama.cpp).
Without --yes it prints the pre-flight and the plan and changes nothing.
Also forwards --preflight-only, --dry-run, --skip-weights, --skip-sage, --skip-llm, --gpu X.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PY = (3, 10)
ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQ = ROOT / "requirements.txt"
AUTOMATIC_INSTALL_FLAGS = (
    "--yes",
    "--preflight-only",
    "--dry-run",
    "--skip-weights",
    "--skip-sage",
    "--skip-llm",
    "--klein-4b",
)


def _say(msg: str = "") -> None:
    # Children write straight to the same stdout; flush so lines stay in order.
    print(msg, flush=True)


def _die(msg: str, code: int = 1) -> None:
    _say(f"FAIL  {msg}")
    raise SystemExit(code)


def _banner(where: str) -> str:
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from master_agent.install_report import light_install_banner

    return light_install_banner(where=where)


def _venv_has_pip() -> bool:
    return any(VENV.glob("lib*/python*/site-packages/pip")) or (VENV / "Lib" / "site-packages" / "pip").is_dir()


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _check_only(argv: list[str]) -> int:
    py = _venv_python()
    if not (VENV / "pyvenv.cfg").is_file() or not py.is_file():
        print(f"NEED  {VENV} missing — run `python install.py` (without --check) to create it. Nothing changed.")
        return 1
    if "--fix" in argv:
        print("--check: --fix ignored. Nothing installed.")
    sys.stdout.flush()
    return subprocess.call([str(py), "-m", "master_agent", "setup"])


def main() -> int:
    _say("VIDEO BUDDY installer")
    if sys.version_info[:2] < MIN_PY:
        _die(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ required (found {sys.version.split()[0]})")
    os.chdir(ROOT)
    argv = sys.argv[1:]
    if "--check" in argv and "--automatic-install" not in argv:
        return _check_only(argv)
    light = "--automatic-install" not in argv
    if light:
        _say(_banner("start"))
    py = _venv_python()
    if not (VENV / "pyvenv.cfg").is_file():
        _say(f"creating {VENV}")
        existed = VENV.exists()
        code = subprocess.call([sys.executable, "-m", "venv", str(VENV)])
        if code != 0:
            # A failed `venv` leaves pyvenv.cfg behind; the next run would reuse a venv with no pip.
            if not existed:
                shutil.rmtree(VENV, ignore_errors=True)
            _die(
                "could not create .venv. "
                "Debian/Ubuntu: sudo apt install python3-venv, then run install.py again."
            )
    if not py.is_file():
        _die(f"venv python missing: {py}. Delete {VENV} and run install.py again.")
    _say("upgrading pip...")
    if subprocess.call([str(py), "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        if not _venv_has_pip():
            _die(
                f"{VENV} has no pip (an earlier venv create did not finish). "
                f"Delete {VENV} and run install.py again. Debian/Ubuntu: sudo apt install python3-venv first."
            )
        _say("FAILED-NONFATAL  pip self-upgrade")
        _say("        safe to ignore because the pip already in .venv still installs requirements.txt")
    _say("installing requirements...")
    code = subprocess.call([str(py), "-m", "pip", "install", "-r", str(REQ)])
    if code != 0:
        _die("pip install failed")
    if not light:
        forward = [a for a in argv if a in AUTOMATIC_INSTALL_FLAGS]
        if "--gpu" in argv:
            i = argv.index("--gpu")
            if i + 1 < len(argv):
                forward += ["--gpu", argv[i + 1]]
        return subprocess.call([str(py), "-m", "master_agent", "automatic-install", *forward])
    code = subprocess.call([str(py), "-m", "master_agent", "setup", "--fix"])
    _say()
    _say(_banner("end"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
