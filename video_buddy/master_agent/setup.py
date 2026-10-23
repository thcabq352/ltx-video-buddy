"""VIDEO BUDDY setup — check and install local dependencies.

Pip packages, Playwright Chromium, .env, ffmpeg, and the empty Comfy-style
model folders under ``MODELS_DIR`` (``--fix`` only). The local LLM GGUFs
(``LLAMACPP_MODEL``, ``KB_EMBED_MODEL``) have public sources, but this light
install does not download them. It prints a loud warning with the direct link
and writes ``state/LAUNCH_NOTES.txt``. ``automatic-install --yes`` downloads
them. Setup never installs Ollama or pulls Ollama models.
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
    locations = [line.split(":", 1)[1].strip() for line in out.splitlines() if "Install location:" in line]
    if code == 0 and locations:
        detail = f"python package present; chromium at {locations[0]}"[:160]
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


def confirm_prompt(prompt: str) -> bool:
    """y/n. Non-interactive stdin defaults to no so nothing is downloaded by surprise."""
    if sys.stdin is None or not sys.stdin.isatty():
        print(prompt + "n  (non-interactive; pass --yes)")
        return False
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def check_llm_models() -> dict[str, Any]:
    """Local LLM GGUFs in ``MODELS_DIR``. Scan only. Missing files are a WARN, not a quiet skip."""
    try:
        from master_agent.models.llm_gguf import doctor_row

        return doctor_row()
    except Exception as exc:
        return _row(
            "llm-models",
            False,
            f"WARNING: could not scan local LLM files ({exc})",
            fix="python -m master_agent automatic-install --yes",
        )


def check_model_folders() -> dict[str, Any]:
    """Comfy-style subfolders under MODELS_DIR. Scan only; ``setup --fix`` creates them."""
    from master_agent.config import MODELS_DIR
    from master_agent.models.layout import SUBFOLDER_NAMES, missing_model_folders

    missing = missing_model_folders(MODELS_DIR)
    if not missing:
        return _row("model-folders", True, f"all {len(SUBFOLDER_NAMES)} folders exist in {MODELS_DIR}")
    return _row(
        "model-folders",
        False,
        f"{len(missing)} missing in {MODELS_DIR}: " + ", ".join(p.name for p in missing),
        fix="python -m master_agent setup --fix   # makes the empty folders; downloads nothing",
    )


def _managed_comfy_state() -> tuple[str, Path | None]:
    try:
        from master_agent.automatic_install import comfy_install_state, managed_workspace

        workspace = managed_workspace()
        return comfy_install_state(workspace), workspace
    except Exception:
        return "unknown", None


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
    state, workspace = _managed_comfy_state()
    detail = f"not reachable at {url}"
    if workspace is not None:
        detail += f"; Buddy's ComfyUI folder {workspace} (from {_managed_comfy_source()}) is {state}"
    if portable.is_file():
        hint = str(portable)
    elif state == "installed":
        hint = f"ComfyUI is installed in {workspace}. Start it: python -m master_agent comfy start"
        failed_log = _last_launch_log()
        if failed_log:
            hint += f". The last start failed; its output is in {failed_log}"
    elif state in {"absent", "partial"}:
        from master_agent.install_report import FULL_INSTALL_CMD

        hint = (
            f"ComfyUI is not installed yet. Install it with LTX 2.3: {FULL_INSTALL_CMD} "
            "(already installed elsewhere? set MANAGED_COMFY_ROOT to that folder in .env; "
            "or start your own ComfyUI with --port 8188 and set COMFYUI_URL)"
        )
    else:
        hint = "start ComfyUI with --port 8188 (or set COMFYUI_URL)"
    return _row("comfyui", False, detail, fix=hint)


def check_comfy_node_deps() -> dict[str, Any]:
    """gguf / colour-science importable in the managed ComfyUI's own python. Read-only."""
    state, workspace = _managed_comfy_state()
    if workspace is None or state != "installed":
        return _row("comfy-node-deps", True, f"not checked: Buddy's ComfyUI folder is {state}")
    try:
        from master_agent.automatic_install import comfy_python, missing_comfy_packages

        py, flags = comfy_python(workspace)
        if py is None:
            return _row("comfy-node-deps", True, f"not checked: no ComfyUI python in {workspace}")
        missing = missing_comfy_packages(py, flags)
    except Exception as exc:
        return _row("comfy-node-deps", True, f"not checked ({type(exc).__name__}: {exc})")
    if not missing:
        return _row("comfy-node-deps", True, f"gguf, colour-science importable in {py}")
    from master_agent.install_report import FULL_INSTALL_CMD

    return _row(
        "comfy-node-deps",
        False,
        f"missing in ComfyUI's python {py}: {', '.join(missing)}. "
        "ComfyUI-LTXVideo / ComfyUI-GGUF will not load, so video generation is blocked",
        fix=f"{FULL_INSTALL_CMD} (installs them into ComfyUI's python and skips everything already done)",
    )


def _managed_comfy_source() -> str:
    try:
        from master_agent.comfy.root import resolve_managed_comfy_root

        return resolve_managed_comfy_root().source
    except Exception:
        return "unknown"


def _last_launch_log() -> str:
    try:
        from master_agent.comfy.tower import load_state

        st = load_state()
    except Exception:
        return ""
    failure = (st.extra or {}).get("last_launch_failure") or {}
    return str(failure.get("log") or "") if st.last_error else ""


def dest_dir(role: str) -> str:
    from master_agent.models.weights import dest_dir as _dest_dir

    return _dest_dir(role)


def _into(weights: list[Any]) -> str:
    """`` (into <MODELS_DIR>/<folder>/, ...)`` for the folders the missing files go in."""
    dirs = dict.fromkeys(dest_dir(w.dest_folder) for w in weights)
    return f" (into {', '.join(dirs)})" if dirs else ""


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
    detail = f"missing {names}{more}{_into(status.missing_mandatory)}"
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
    from master_agent.models.deferred import DOWNLOAD_CMD, REGISTER_CMD, read_marker

    marker = read_marker()
    if marker is not None:
        when = marker.get("deferred_at") or "earlier"
        row = _row(
            "ltx23-weights",
            False,
            f"deferred ({marker.get('reason') or '--skip-weights'}, {when}); "
            f"no LTX 2.3 GGUF yet. It goes in {dest_dir('diffusion_models')}",
            fix=f"{REGISTER_CMD}   # or: {DOWNLOAD_CMD}",
        )
        row["deferred"] = True
        return row
    return _row(
        "ltx23-weights",
        False,
        "no compatible LTX 2.3 GGUF after inventory "
        "(sulphur_dev-Q3_K_S.gguf and other LTX 2.3 Q3/Q4/Q5 GGUF count). "
        f"It goes in {dest_dir('diffusion_models')}",
        fix="python -m master_agent download-models --ltx23   # lists the LTX 2.3 pack; --yes fetches. Doctor does not download",
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
    detail = f"missing {names}{more}{_into(status.missing_mandatory)}"
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


def check_budget() -> dict[str, Any]:
    """Shift budget, read-only. A HOLD is ``paused``, never a doctor failure."""
    from master_agent.control.budget import format_shift_summary, shift_summary

    line = format_shift_summary(shift_summary())
    return _row("budget", True, line.removeprefix("budget: "))


def snapshot() -> list[dict[str, Any]]:
    return [
        check_python(),
        check_venv(),
        check_pip(),
        check_playwright(),
        check_env(),
        check_ffmpeg(),
        check_model_folders(),
        check_llm_models(),
        check_comfy(),
        check_comfy_node_deps(),
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
        check_budget(),
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
# Missing LLM GGUFs are WARN (a link and manual steps), not INFO and not a
# quiet skip. They do not change the exit code: video generation can still run.
_OPTIONAL_INFO_ROWS = frozenset(
    {
        "heartlib",
        "heartmula-weights",
        "heartmula-comfy",
        "hardware",
        "pack-pins",
        "model-folders",
    }
)
_WARN_ROWS = frozenset({"llm-models"})

# Rows ``setup --fix`` can actually repair. Everything else needs automatic
# install, download-models, or the user.
_FIXED_BY_SETUP_FIX = frozenset({"venv", "pip", "playwright", "env", "ffmpeg", "model-folders"})


def heartmula_rows() -> list[dict[str, Any]]:
    from master_agent.heartmula.doctor import heartmula_doctor_rows

    return heartmula_doctor_rows()


def classify_row(row: dict[str, Any], *, light: bool = False, weight_optional: bool = False) -> str:
    """OK | INFO | FAIL for one doctor row. ``light`` is the plain install.py / setup --fix view."""
    from master_agent.install_report import FAIL, INFO, LIGHT_INSTALL_NOT_INCLUDED, OK, WARN

    name = row["name"]
    if row["ok"]:
        return OK
    if name in _WARN_ROWS:
        return WARN
    if name in _OPTIONAL_INFO_ROWS or row.get("deferred"):
        return INFO
    if weight_optional and name in _WEIGHT_ROWS:
        return INFO
    if light and name in LIGHT_INSTALL_NOT_INCLUDED:
        return INFO
    return FAIL


def _row_reason(name: str, *, light: bool) -> str:
    from master_agent.install_report import INFO_ROW_REASONS, LIGHT_INSTALL_NOT_INCLUDED

    if name in INFO_ROW_REASONS:
        return INFO_ROW_REASONS[name]
    if light and name in LIGHT_INSTALL_NOT_INCLUDED:
        return LIGHT_INSTALL_NOT_INCLUDED[name]
    return ""


def print_report(rows: list[dict[str, Any]], *, weight_optional: bool = False, light: bool = False) -> int:
    """Print doctor rows. Only FAIL rows count toward the exit code.

    ``light`` (setup --fix, i.e. plain install.py) downgrades the rows the
    light install never provides (ComfyUI, weight packs) to INFO, and ends
    with the ``Install OK`` / ``Install FAILED`` summary line.
    """
    from master_agent.install_report import FAIL, INFO, OK, WARN, safe_to_ignore, summary_line

    print("VIDEO BUDDY setup")
    failed: list[str] = []
    counts = {OK: 0, INFO: 0, WARN: 0}
    for row in rows:
        level = classify_row(row, light=light, weight_optional=weight_optional)
        name = row["name"]
        detail = row["detail"]
        if weight_optional and not row["ok"] and name in _WEIGHT_ROWS:
            detail = f"{detail} (--use-existing: not downloading)"
        if level == FAIL:
            failed.append(name)
            mark = "NEED"
        else:
            counts[level] += 1
            mark = "OK  " if level == OK else ("WARN" if level == WARN else "INFO")
        print(f"  {mark}  {name:<18} {detail}")
        if level == WARN:
            print("        WARNING: local LLM files are missing. This is not safe to ignore.")
            if "projector MISSING" in str(detail):
                print("        WARNING: the brain GGUF is on disk but the vision projector is not.")
        if level == INFO:
            if row.get("deferred"):
                from master_agent.models.deferred import REASON as why
            else:
                why = _row_reason(name, light=light)
            if why:
                print(f"        {safe_to_ignore(why)}")
        use_existing = weight_optional and name in _WEIGHT_ROWS
        if level != OK and row.get("fix") and not use_existing:
            print(f"        -> {row['fix']}")
    print()
    notes = counts[INFO] + counts[WARN]
    if light:
        print(summary_line(ok=counts[OK], notes=notes, failed=failed))
    elif failed:
        fixable = [n for n in failed if n in _FIXED_BY_SETUP_FIX]
        rest = [n for n in failed if n not in _FIXED_BY_SETUP_FIX]
        print(f"{len(failed)} check(s) need work.")
        if fixable:
            print(f"  setup --fix handles: {', '.join(fixable)}. Run: python -m master_agent setup --fix")
        if rest:
            from master_agent.install_report import FULL_INSTALL_CMD

            print(f"  setup --fix does not handle: {', '.join(rest)}. Each row's -> line says what does.")
            if "comfyui" in rest or "ltx23-weights" in rest:
                print(f"  ComfyUI + LTX 2.3 for a new machine: {FULL_INSTALL_CMD}")
    elif counts[WARN]:
        print(
            f"Video checks passed with {counts[WARN]} warning(s): "
            + ", ".join(n for n in (row["name"] for row in rows) if n == "llm-models")
            + ". See the WARN row above. Local LLM features stay off until those files are in place."
        )
    else:
        print("All checked dependencies are ready.")
    return 0 if not failed else 1


def ensure_venv() -> None:
    if (VENV / "pyvenv.cfg").is_file():
        return
    print(f"creating venv {VENV}")
    code, out = _run([sys.executable, "-m", "venv", str(VENV)], timeout=180)
    if code != 0:
        raise RuntimeError(f"venv failed: {out}")


def _nonfatal(what: str, why: str, tail: str = "") -> None:
    from master_agent.install_report import NONFATAL, safe_to_ignore

    print(f"{NONFATAL}  {what}")
    if tail:
        print(f"        {tail.strip().splitlines()[-1][:300]}")
    print(f"        {safe_to_ignore(why)}")


def install_pip() -> None:
    py = _py()
    print("installing Python packages...")
    code, out = _run([py, "-m", "pip", "install", "--upgrade", "pip"], timeout=180)
    if code != 0:
        _nonfatal("pip self-upgrade", "the pip already in .venv still installs requirements.txt", out)
    code, out = _run([py, "-m", "pip", "install", "-r", str(REQ)], timeout=900)
    if code != 0:
        raise RuntimeError(f"pip install failed:\n{out[-800:]}")


def install_model_folders() -> None:
    """Empty Comfy-style subfolders under MODELS_DIR. Never downloads."""
    from master_agent.config import MODELS_DIR
    from master_agent.models.layout import CREATED, ensure_model_folders

    try:
        made = [a.path for a in ensure_model_folders(MODELS_DIR) if a.action == CREATED]
    except OSError as exc:
        from master_agent.install_report import INFO_ROW_REASONS

        _nonfatal(f"could not create model folders in {MODELS_DIR} ({exc})", INFO_ROW_REASONS["model-folders"])
        return
    if made:
        print(f"created {len(made)} model folder(s) in {MODELS_DIR}: " + ", ".join(p.name for p in made))


def install_playwright() -> None:
    py = _py()
    print("installing Playwright Chromium...")
    code, out = _run([py, "-m", "playwright", "install", "chromium"], timeout=600)
    if code != 0:
        from master_agent.install_report import STEP_REASONS

        _nonfatal(
            f"playwright chromium (retry: {py} -m playwright install chromium)",
            STEP_REASONS["Playwright Chromium"],
            out,
        )


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
    print("FAIL  could not auto-install ffmpeg (Buddy needs it to write and check video). " + check_ffmpeg()["fix"])


def install_llm_models() -> None:
    """Warn loudly when the GGUFs are missing. The light install does not download them.

    ``automatic-install --yes`` is the command that fetches them. This function
    does not fail ``setup --fix``: video deps are a separate step, and the
    warning is the doctor's WARN row plus ``LAUNCH_NOTES.txt``.
    """
    try:
        from master_agent.automatic_install import step_llm_models

        result = step_llm_models(True, skip=False, fetch=False)
    except Exception as exc:
        print(f"WARNING: local LLM check failed ({exc}). This is not a silent skip.")
        print("        Run: python -m master_agent automatic-install --yes")
        return
    for line in result.lines:
        print(line)
    if result.status == "FAILED":
        print("WARNING: local LLM files are missing. See LAUNCH_NOTES.txt. Video install continues.")


def fix() -> int:
    """The light install. Does not run ``comfy install`` or ``comfy update`` or fetch weights."""
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
        install_model_folders()
        install_llm_models()
    except RuntimeError as exc:
        print(f"FAIL  {exc}")
        return 1
    print()
    return print_report(snapshot(), light=True)


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
        rc = fix()
        if mode not in {"download"} and not fix_models:
            return rc
    elif do_fix and mode == "scan":
        print("--scan-only: --fix ignored. Nothing installed or downloaded.")
        return print_report(snapshot(), weight_optional=False)
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
