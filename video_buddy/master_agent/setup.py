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


def ollama_list_text() -> str:
    """Raw ``ollama list`` output, or empty when the CLI is missing."""
    if not _which("ollama"):
        return ""
    code, out = _run(["ollama", "list"], timeout=20)
    return out if code == 0 else ""


def ollama_has_model(listed: str, model: str) -> bool:
    """True when ``model`` (or ``model:tag``) is already in ``ollama list``."""
    want = (model or "").strip().lower()
    if not want:
        return False
    for line in (listed or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        token = parts[0].lower()
        if token in {"name", "name:"}:
            continue
        base = token.split(":", 1)[0]
        if token == want or base == want or token.startswith(want + ":"):
            return True
    return False


def confirm_prompt(prompt: str) -> bool:
    """y/n. Non-interactive stdin defaults to no so nothing is pulled by surprise."""
    if sys.stdin is None or not sys.stdin.isatty():
        print(prompt + "n  (non-interactive; pass --yes)")
        return False
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def check_ollama() -> dict[str, Any]:
    path = _which("ollama")
    if not path:
        return _row(
            "ollama",
            False,
            "not on PATH",
            fix="Install from https://ollama.com/download then: ollama pull qwen3-vl-heretic",
        )
    out = ollama_list_text()
    missing = [m for m in OLLAMA_MODELS if not ollama_has_model(out, m)]
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
    from master_agent.models.vram_policy import heavy_not_suggested
    from master_agent.models.weights import describe_transformer_pick

    transformer_path = status.found_paths.get("transformer")
    pick = describe_transformer_pick(Path(transformer_path) if transformer_path else None)
    if transformer_path and heavy_not_suggested(Path(transformer_path).name):
        pick += " — on disk, not suggested for this VRAM; prefer GGUF (FORCE_LOADER=gguf, VRAM_GB=12)"
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


_H3_US_NOTE = (
    "US is an excluded territory under the MiniMax H3 license; "
    "leave these workflows unused there."
)


def _with_h3_us(detail: str) -> str:
    if _H3_US_NOTE in detail:
        return detail
    return f"{detail} {_H3_US_NOTE}"


def check_ltx23_weights() -> dict[str, Any]:
    """Scan for an LTX 2.3 GGUF (including Sulphur Q3_K_S). Never downloads."""
    try:
        from master_agent.models.weights import find_ltx23_compatible

        found = find_ltx23_compatible()
    except Exception as exc:
        return _row(
            "ltx23-weights",
            False,
            f"scan failed: {exc}",
            fix="python -m master_agent inventory",
        )
    if found is not None:
        return _row(
            "ltx23-weights",
            True,
            f"GGUF-first {found.name} at {found} — fp8/EROS only if no GGUF matches",
        )
    return _row(
        "ltx23-weights",
        False,
        "no compatible LTX 2.3 GGUF after inventory "
        "(sulphur_dev-Q3_K_S.gguf and other LTX 2.3 Q3/Q4/Q5 GGUF count)",
        fix="python -m master_agent inventory   # doctor does not download this",
    )


def check_h3_weights() -> dict[str, Any]:
    """Scan-only MiniMax H3 inventory. Never downloads."""
    try:
        from master_agent.models.weights import scan_bundle

        status = scan_bundle("h3_fl2va")
    except Exception as exc:
        return _row(
            "h3-weights",
            False,
            _with_h3_us(f"scan failed: {exc}"),
            fix="python -m master_agent download-models --h3",
        )
    from master_agent.models.weights import describe_h3_transformer_pick

    pick = describe_h3_transformer_pick(
        Path(status.found_paths["h3_fl2va"]) if status.found_paths.get("h3_fl2va") else None
    )
    if status.ok:
        found = ", ".join(Path(p).name for p in status.found_paths.values()) or "accepted local names"
        return _row("h3-weights", True, _with_h3_us(f"{pick}; present ({found})"))
    names = ", ".join(w.filename for w in status.missing_mandatory[:4])
    more = f" (+{len(status.missing_mandatory) - 4} more)" if len(status.missing_mandatory) > 4 else ""
    hint = "python -m master_agent download-models --h3   # review confirmed-missing only, then --yes"
    detail = f"missing {names}{more}"
    if status.found_paths.get("h3_fl2va"):
        detail = f"{pick}; {detail}"
    return _row("h3-weights", False, _with_h3_us(detail), fix=hint)


def check_ltx25_ic_lora() -> dict[str, Any]:
    """Exact Ingredients LoRA for ltx25_msr / ltx25_v2v. Scan only."""
    try:
        from master_agent.models.weights import ic_ingredients_placement

        place = ic_ingredients_placement()
    except Exception as exc:
        return _row(
            "ltx25-ic-lora",
            False,
            f"scan failed: {exc}",
            fix="python -m master_agent download-models --ltx25",
        )
    return _row(
        "ltx25-ic-lora",
        bool(place.get("ok")),
        str(place.get("detail") or ""),
        fix=str(place.get("fix") or ""),
    )


def check_ltx23_inoutpaint_lora() -> dict[str, Any]:
    """Official LTX 2.3 In-Outpainting IC-LoRA. Scan only — never downloads."""
    try:
        from master_agent.models.weights import ltx23_inoutpaint_lora_placement

        place = ltx23_inoutpaint_lora_placement()
    except Exception as exc:
        return _row("ltx23-inoutpaint-lora", False, f"scan failed: {exc}")
    return _row(
        "ltx23-inoutpaint-lora",
        bool(place.get("ok")),
        str(place.get("detail") or ""),
        fix=str(place.get("fix") or ""),
    )


def check_ltx23_latent_upscaler() -> dict[str, Any]:
    """LatentUpscaleModelLoader only sees models/latent_upscale_models/."""
    try:
        from master_agent.models.weights import ltx23_latent_upscaler_placement

        place = ltx23_latent_upscaler_placement()
    except Exception as exc:
        return _row("ltx23-upscaler", False, f"scan failed: {exc}")
    return _row(
        "ltx23-upscaler",
        bool(place.get("ok")),
        str(place.get("detail") or ""),
        fix=str(place.get("fix") or ""),
    )


def check_dev_fp8_trailer() -> dict[str, Any]:
    """Warn when the dev fp8 checkpoint is bigger than its safetensors header.

    The detail reports actual size, header-declared size, and extra bytes.
    The fix is to back up the file and truncate it to the declared size.
    The check does not rewrite the file.
    """
    try:
        from master_agent.models.weights import dev_fp8_trailing_bytes

        place = dev_fp8_trailing_bytes()
    except Exception as exc:
        return _row("dev-fp8-trailer", True, f"scan skipped: {exc}")
    return _row(
        "dev-fp8-trailer",
        bool(place.get("ok")),
        str(place.get("detail") or ""),
        fix=str(place.get("fix") or ""),
    )


def check_ltx_guide() -> dict[str, Any]:
    """Report whether pause-reset can pin a last-frame keyframe. Never downloads."""
    try:
        from master_agent.config import OBJECT_INFO_CACHE
        from master_agent.orchestrator.lipdub_guide import (
            GUIDE_INFEASIBLE_MESSAGE,
            GUIDE_MISSING_MESSAGE,
            GUIDE_READY_MESSAGE,
            probe_pause_reset_guide,
        )

        if not OBJECT_INFO_CACHE.is_file():
            return _row(
                "ltx-guide",
                False,
                GUIDE_MISSING_MESSAGE + " No cached object_info.",
                fix="python -m master_agent fetch-object-info",
            )
        import json

        info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
        available, message = probe_pause_reset_guide(info if isinstance(info, dict) else {})
        if available:
            return _row("ltx-guide", True, message or GUIDE_READY_MESSAGE)
        fix = "python -m master_agent fetch-object-info"
        if message.startswith("LTXVAddGuide is registered"):
            fix = ""
            message = message or GUIDE_INFEASIBLE_MESSAGE
        return _row("ltx-guide", False, message or GUIDE_MISSING_MESSAGE, fix=fix)
    except Exception as exc:
        return _row(
            "ltx-guide",
            False,
            f"pause-reset guide probe failed: {exc}. "
            "LTXVAddGuide is a core LTX node, not a new download. "
            "Without it, pauses re-anchor on the source still with a crossfade inside the silence.",
        )


def check_vram_policy() -> dict[str, Any]:
    """Shared 16GB-class pack policy (does not fetch)."""
    try:
        from master_agent.models.vram_policy import format_doctor_line

        return _row("vram-policy", True, format_doctor_line())
    except Exception as exc:
        return _row("vram-policy", False, f"policy failed: {exc}")


def check_hardware() -> dict[str, Any]:
    """NVIDIA / AMD / ROCm routing sentence. Never a failed install check."""
    try:
        from master_agent.comfy.hardware import scan_hardware

        scan = scan_hardware()
    except Exception as exc:
        scan = {
            "sentence": f"Hardware scan failed ({exc}). This scan does not block install.",
            "band": "unknown",
        }
    row = _row("hardware", True, str(scan.get("sentence") or ""))
    row["blocks_install"] = False
    row["band"] = scan.get("band") or "unknown"
    return row


def check_pack_pins() -> dict[str, Any]:
    """Report stale Comfy pins. Does not run ``comfy update``."""
    try:
        from master_agent.comfy.updates import scan_packs

        report = scan_packs()
    except Exception as exc:
        return _row("pack-pins", True, f"Pack scan failed ({exc}). No update ran.")
    stale = report.get("stale") or []
    if not stale:
        return _row("pack-pins", True, "nothing stale vs known pins. No update ran.")
    names = ", ".join(str(item.get("name")) for item in stale)
    return _row(
        "pack-pins",
        False,
        f"stale: {names}. No update ran.",
        fix="python -m master_agent comfy update --yes",
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
        check_vram_policy(),
        check_hardware(),
        check_pack_pins(),
        check_ltx_guide(),
        check_ltx25_weights(),
        check_ltx23_weights(),
        check_ltx25_ic_lora(),
        check_ltx23_inoutpaint_lora(),
        check_ltx23_latent_upscaler(),
        check_dev_fp8_trailer(),
        check_h3_weights(),
        *heartmula_rows(),
    ]


# Weight rows stay informational under --use-existing (do not fail the process).
_WEIGHT_ROWS = frozenset(
    {
        "ltx25-weights",
        "ltx23-weights",
        "ltx25-ic-lora",
        "ltx23-inoutpaint-lora",
        "ltx23-upscaler",
        "h3-weights",
        "dev-fp8-trailer",
    }
)

# Optional or report-only rows: show them, and do not fail doctor.
# Hardware never blocks install. Stale packs wait for an explicit comfy update.
_OPTIONAL_INFO_ROWS = frozenset(
    {
        "heartlib",
        "heartmula-weights",
        "heartmula-comfy",
        "hardware",
        "pack-pins",
    }
)


def heartmula_rows() -> list[dict[str, Any]]:
    from master_agent.heartmula.doctor import heartmula_doctor_rows

    return heartmula_doctor_rows()


def print_report(rows: list[dict[str, Any]], *, weight_optional: bool = False) -> int:
    print("VIDEO BUDDY setup")
    failed = 0
    for row in rows:
        mark = "OK  " if row["ok"] else "NEED"
        info_only = row["name"] in _OPTIONAL_INFO_ROWS
        optional = weight_optional and not row["ok"] and row["name"] in _WEIGHT_ROWS
        if not row["ok"] and not optional and not info_only:
            failed += 1
        detail = row["detail"]
        if optional:
            detail = f"{detail} (--use-existing: not downloading)"
        print(f"  {mark}  {row['name']:<18} {detail}")
        if not row["ok"] and row.get("fix") and (info_only or not optional):
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


def install_ollama_models(*, consent: bool = False) -> None:
    """Pull Ollama models only when they are absent and the operator opted in.

    Models already listed by ``ollama list`` are skipped. A missing model is
    not pulled unless ``consent`` is true (``--yes``) or the operator answers
    ``y`` at the prompt. Non-interactive runs without ``--yes`` do not pull.
    """
    if not _which("ollama"):
        print("WARN  ollama not on PATH — install from https://ollama.com/download")
        return
    listed = ollama_list_text()
    for model in OLLAMA_MODELS:
        if ollama_has_model(listed, model):
            print(f"SKIP  ollama {model} — already in ollama list")
            continue
        allowed = consent or confirm_prompt(f"Pull ollama model {model}? [y/N] ")
        if not allowed:
            print(f"SKIP  ollama pull {model} — no confirmation (pass --yes to pull)")
            continue
        print(f"ollama pull {model}…")
        code, out = _run(["ollama", "pull", model], timeout=1800)
        if code != 0:
            print(f"WARN  ollama pull {model}: {out[-300:]}")


def fix(*, pull_ollama: bool = False) -> int:
    """Install local tooling. Does not run ``comfy install`` or ``comfy update``."""
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
        install_ollama_models(consent=pull_ollama)
    except RuntimeError as exc:
        print(f"FAIL  {exc}")
        return 1
    print()
    return print_report(snapshot())


def print_inventory_preamble() -> None:
    """Count discovered weights before doctor or download-models suggests anything."""
    try:
        from master_agent.config import COMFYUI_ROOT, MODELS_DIR
        from master_agent.models.inventory import scan_inventory

        inv = scan_inventory(MODELS_DIR, COMFYUI_ROOT, write=False)
    except Exception as exc:
        print(f"Inventory scan failed ({exc}). Nothing downloaded.")
        return
    usable = [e for e in inv.entries if not e.partial]
    print(
        f"Inventory first: {len(usable)} weight file(s) on disk. "
        "`python -m master_agent inventory` lists paths and roles. "
        "Nothing downloaded yet."
    )


def cmd_setup(
    *,
    do_fix: bool,
    fix_models: bool = False,
    mode: str = "report",
    yes: bool = False,
) -> int:
    """``mode`` is report | scan | existing | download.

    ``scan`` and ``existing`` never fetch weights. ``download`` fetches only
    after ``--yes`` or a y/n prompt. ``--fix-models`` remains the consent
    shortcut (it still inventories first and skips files already on disk).
    """
    print_inventory_preamble()
    weight_optional = mode == "existing"
    rc = 0
    if do_fix and mode != "scan":
        rc = fix(pull_ollama=yes)
        if mode not in {"download"} and not fix_models:
            return rc
    elif do_fix and mode == "scan":
        rc = fix(pull_ollama=False)
        return print_report(snapshot(), weight_optional=False) if rc == 0 else rc
    if mode in {"scan", "existing"} or (not fix_models and mode != "download"):
        return print_report(snapshot(), weight_optional=weight_optional) or rc
    from master_agent.models.weights import MissingWeightsError, download_missing_bundle, format_ask, scan_bundle

    status = scan_bundle("ltx25_all")
    if status.ok:
        print("OK    LTX 2.5 weights already present (compatible names included)")
        return print_report(snapshot(), weight_optional=weight_optional) or rc
    print(format_ask(status))
    print()
    consent = bool(yes or fix_models)
    if not consent:
        consent = confirm_prompt("Download the confirmed-missing set? [y/N] ")
    if not consent:
        print("Nothing downloaded.")
        return 2
    if fix_models:
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
