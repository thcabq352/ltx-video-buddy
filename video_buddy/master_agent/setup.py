"""VIDEO BUDDY setup — check and install local dependencies.

Pip packages, Playwright Chromium, .env, ffmpeg, and Ollama models.
ComfyUI itself is a separate render engine (not downloaded here).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

MIN_PY = (3, 10)
OLLAMA_MODELS = ("qwen3-vl-heretic", "nomic-embed-text")

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements.txt"
ENV_EXAMPLE = ROOT / ".env.example"
ENV_FILE = ROOT / ".env"
VENV = ROOT / ".venv"


def _py() -> str:
    if sys.prefix == str(VENV) or Path(sys.prefix).resolve() == VENV.resolve():
        return sys.executable
    win = VENV / "Scripts" / "python.exe"
    unix = VENV / "bin" / "python"
    if win.is_file():
        return str(win)
    if unix.is_file():
        return str(unix)
    return sys.executable


def _run(cmd: list[str], *, timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _row(name: str, ok: bool, detail: str, *, fix: str = "") -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail, "fix": fix}


def check_python() -> dict[str, Any]:
    ver = sys.version_info[:3]
    ok = ver >= MIN_PY
    return _row(
        "python",
        ok,
        f"{ver[0]}.{ver[1]}.{ver[2]} ({sys.executable})",
        fix=f"Install Python {MIN_PY[0]}.{MIN_PY[1]}+ from https://www.python.org/downloads/",
    )


def check_venv() -> dict[str, Any]:
    exists = (VENV / "pyvenv.cfg").is_file()
    in_venv = Path(sys.prefix).resolve() == VENV.resolve()
    return _row(
        "venv",
        exists,
        "project .venv ready" if exists else "missing .venv",
        fix=f"{sys.executable} -m venv .venv",
    ) | {"in_venv": in_venv}


def check_pip() -> dict[str, Any]:
    if not REQ.is_file():
        return _row("pip", False, "requirements.txt missing")
    needed = []
    for line in REQ.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(";", 1)[0].split("[", 1)[0]
        for sep in (">=", "==", "~=", "<=", ">", "<"):
            if sep in name:
                name = name.split(sep, 1)[0]
                break
        needed.append(name.strip().lower().replace("_", "-"))
    code, out = _run([_py(), "-m", "pip", "freeze"])
    if code != 0:
        return _row("pip", False, "pip freeze failed", fix=f"{_py()} -m pip install -r requirements.txt")
    have = {line.split("==", 1)[0].strip().lower().replace("_", "-") for line in out.splitlines() if "==" in line}
    missing = [n for n in needed if n not in have]
    # huggingface_hub vs huggingface-hub
    aliases = {"huggingface-hub": "huggingface_hub", "huggingface_hub": "huggingface-hub"}
    missing = [n for n in missing if aliases.get(n, n) not in have]
    ok = not missing
    return _row(
        "pip",
        ok,
        "all requirements installed" if ok else "missing " + ", ".join(missing[:8]),
        fix=f"{_py()} -m pip install -r requirements.txt",
    )


def check_playwright() -> dict[str, Any]:
    code, _ = _run([_py(), "-c", "from playwright.sync_api import sync_playwright"])
    if code != 0:
        return _row("playwright", False, "python package missing", fix="pip install playwright && playwright install chromium")
    code, out = _run([_py(), "-m", "playwright", "install", "--dry-run", "chromium"], timeout=60)
    # dry-run may not exist on older playwright; treat import success as soft-ok
    detail = "python package present"
    if code == 0 and out:
        detail = out.splitlines()[-1][:160]
    return _row("playwright", True, detail, fix=f"{_py()} -m playwright install chromium")


def check_env() -> dict[str, Any]:
    if ENV_FILE.is_file():
        return _row("env", True, str(ENV_FILE))
    return _row("env", False, ".env missing", fix=f"copy {ENV_EXAMPLE.name} to .env")


def check_ffmpeg() -> dict[str, Any]:
    path = _which("ffmpeg")
    if path:
        return _row("ffmpeg", True, path)
    return _row(
        "ffmpeg",
        False,
        "not on PATH",
        fix="Windows: winget install Gyan.FFmpeg  |  macOS: brew install ffmpeg  |  Linux: sudo apt install ffmpeg",
    )


def check_ollama() -> dict[str, Any]:
    path = _which("ollama")
    if not path:
        return _row(
            "ollama",
            False,
            "not on PATH",
            fix="Install from https://ollama.com/download then: ollama pull qwen3-vl-heretic",
        )
    code, out = _run(["ollama", "list"], timeout=20)
    have = out.lower() if code == 0 else ""
    missing = [m for m in OLLAMA_MODELS if m not in have]
    ok = not missing
    return _row(
        "ollama",
        ok,
        path if ok else f"{path} — pull {', '.join(missing)}",
        fix=" && ".join(f"ollama pull {m}" for m in missing) if missing else "",
    )


def check_comfy() -> dict[str, Any]:
    url = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
    try:
        import httpx

        r = httpx.get(f"{url.rstrip('/')}/system_stats", timeout=2.0)
        if r.status_code < 400:
            return _row("comfyui", True, url)
    except Exception:
        pass
    portable = ROOT / "ComfyUI_windows_portable" / "run_api_8188.bat"
    hint = str(portable) if portable.is_file() else "start ComfyUI with --port 8188 (or set COMFYUI_URL)"
    return _row("comfyui", False, f"not reachable at {url}", fix=hint)


def check_ltx25_weights() -> dict[str, Any]:
    """Scan-only. Never downloads. Missing files become NEED + ask-to-download."""
    try:
        from master_agent.models.weights import scan_bundle

        status = scan_bundle("ltx25_core")
    except Exception as exc:
        return _row("ltx25-weights", False, f"scan failed: {exc}", fix="python -m master_agent download-models --ltx25")
    from master_agent.models.weights import describe_transformer_pick

    pick = describe_transformer_pick(
        Path(status.found_paths["transformer"]) if status.found_paths.get("transformer") else None
    )
    if status.ok:
        found = ", ".join(Path(p).name for p in status.found_paths.values()) or "accepted local names"
        return _row("ltx25-weights", True, f"{pick}; present ({found})")
    names = ", ".join(w.filename for w in status.missing_mandatory[:4])
    more = f" (+{len(status.missing_mandatory) - 4} more)" if len(status.missing_mandatory) > 4 else ""
    hint = "python -m master_agent download-models --ltx25   # review confirmed-missing only, then --yes"
    if len(status.missing_mandatory) == 1 and status.missing_mandatory[0].key == "duration_head":
        hint = "duration-head is missing or zero-byte — " + hint
    detail = f"missing {names}{more}"
    if status.found_paths.get("transformer"):
        detail = f"{pick}; {detail}"
    return _row(
        "ltx25-weights",
        False,
        detail,
        fix=hint,
    )


def snapshot() -> list[dict[str, Any]]:
    return [
        check_python(),
        check_venv(),
        check_pip(),
        check_playwright(),
        check_env(),
        check_ffmpeg(),
        check_ollama(),
        check_comfy(),
        check_ltx25_weights(),
    ]


def print_report(rows: list[dict[str, Any]]) -> int:
    print("VIDEO BUDDY setup")
    failed = 0
    for row in rows:
        mark = "OK  " if row["ok"] else "NEED"
        if not row["ok"]:
            failed += 1
        print(f"  {mark}  {row['name']:<12} {row['detail']}")
        if not row["ok"] and row.get("fix"):
            print(f"        → {row['fix']}")
    print()
    if failed:
        print(f"{failed} check(s) need work. Re-run: python -m master_agent setup --fix")
    else:
        print("All checked dependencies are ready.")
    return 0 if failed == 0 else 1


def ensure_venv() -> None:
    if (VENV / "pyvenv.cfg").is_file():
        return
    print(f"creating venv {VENV}")
    code, out = _run([sys.executable, "-m", "venv", str(VENV)], timeout=180)
    if code != 0:
        raise RuntimeError(f"venv failed: {out}")


def install_pip() -> None:
    py = _py()
    print("installing Python packages…")
    code, out = _run([py, "-m", "pip", "install", "--upgrade", "pip"], timeout=180)
    if code != 0:
        print(out[-400:])
    code, out = _run([py, "-m", "pip", "install", "-r", str(REQ)], timeout=900)
    if code != 0:
        raise RuntimeError(f"pip install failed:\n{out[-800:]}")


def install_playwright() -> None:
    py = _py()
    print("installing Playwright Chromium…")
    code, out = _run([py, "-m", "playwright", "install", "chromium"], timeout=600)
    if code != 0:
        print(f"WARN  playwright chromium: {out[-400:]}")


def install_env() -> None:
    if ENV_FILE.is_file() or not ENV_EXAMPLE.is_file():
        return
    ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {ENV_FILE} from .env.example")


def install_ffmpeg() -> None:
    if _which("ffmpeg"):
        return
    plat = sys.platform
    print("installing ffmpeg…")
    if plat == "win32" and _which("winget"):
        code, out = _run(
            ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--accept-package-agreements", "--accept-source-agreements"],
            timeout=300,
        )
        if code == 0:
            return
        print(out[-300:])
    elif plat == "darwin" and _which("brew"):
        code, out = _run(["brew", "install", "ffmpeg"], timeout=600)
        if code == 0:
            return
        print(out[-300:])
    elif plat.startswith("linux") and _which("apt-get"):
        code, out = _run(["sudo", "apt-get", "update"], timeout=180)
        if code == 0:
            code, out = _run(["sudo", "apt-get", "install", "-y", "ffmpeg"], timeout=300)
            if code == 0:
                return
        print(out[-300:])
    print("WARN  could not auto-install ffmpeg — " + check_ffmpeg()["fix"])


def install_ollama_models() -> None:
    if not _which("ollama"):
        print("WARN  ollama not on PATH — install from https://ollama.com/download")
        return
    for model in OLLAMA_MODELS:
        print(f"ollama pull {model}…")
        code, out = _run(["ollama", "pull", model], timeout=1800)
        if code != 0:
            print(f"WARN  ollama pull {model}: {out[-300:]}")


def fix() -> int:
    print("VIDEO BUDDY setup --fix")
    if sys.version_info[:2] < MIN_PY:
        print(f"FAIL  Python {MIN_PY[0]}.{MIN_PY[1]}+ required")
        return 1
    try:
        ensure_venv()
        install_pip()
        install_playwright()
        install_env()
        install_ffmpeg()
        install_ollama_models()
    except RuntimeError as exc:
        print(f"FAIL  {exc}")
        return 1
    print()
    return print_report(snapshot())


def cmd_setup(*, do_fix: bool, fix_models: bool = False) -> int:
    rc = 0
    if do_fix:
        rc = fix()
        if not fix_models:
            return rc
    if fix_models:
        from master_agent.models.weights import MissingWeightsError, download_missing_bundle, format_ask, scan_bundle

        status = scan_bundle("ltx25_all")
        if status.ok:
            print("OK    LTX 2.5 weights already present")
            return rc
        print(format_ask(status))
        print()
        print("doctor --fix-models is explicit consent to fetch the missing mandatory set.")
        try:
            download_missing_bundle("ltx25_all", yes=True)
        except MissingWeightsError as exc:
            print(f"FAIL  {exc}")
            return 1
        except Exception as exc:
            print(f"FAIL  {exc}")
            return 1
        model_rc = print_report(snapshot())
        return rc or model_rc
    return print_report(snapshot())
