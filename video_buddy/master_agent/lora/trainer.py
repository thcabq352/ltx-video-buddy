"""Run ai-toolkit Flux LoRA training for a character (subprocess behind a seam).

Preflight checks character.json + dataset + ai-toolkit install + flux unet,
writes the YAML config, then runs ai-toolkit's run.py through the injectable
``runner(cmd, log_path) -> returncode`` seam (default: subprocess streaming
stdout to the character's train.log). On success the newest output
safetensors is copied into MODELS_DIR/loras.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from master_agent.config import (
    AI_TOOLKIT_DIR,
    CHARACTERS_DIR,
    FLUX_UNET,
    LORA_DEFAULTS,
    MODELS_DIR,
    resolve_model_path,
)
from master_agent.lora.config_gen import render_train_config

_SETUP_HINT = (
    "ai-toolkit not found. Set it up with: "
    "git clone https://github.com/ostris/ai-toolkit ai-toolkit && "
    "python -m venv ai-toolkit/venv && "
    "ai-toolkit/venv/Scripts/python -m pip install -r ai-toolkit/requirements.txt"
)


def _toolkit_python() -> Path:
    win = AI_TOOLKIT_DIR / "venv" / "Scripts" / "python.exe"
    if win.is_file() or sys.platform.startswith("win"):
        return win
    return AI_TOOLKIT_DIR / "venv" / "bin" / "python"


def _default_runner(cmd: list[str], log_path: Path) -> int:
    """Stream ai-toolkit stdout to the log file; return the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(AI_TOOLKIT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            try:
                print(line, end="")
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode("ascii"), end="")
        return proc.wait()


def _log_tail(log_path: Path, chars: int = 2000) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no training log)"
    return text[-chars:]


def _attempt_number(name: str) -> int:
    """1 + count of already-trained loras for this character."""
    loras_dir = MODELS_DIR / "loras"
    if not loras_dir.is_dir():
        return 1
    return 1 + len(list(loras_dir.glob(f"{name}_r*.safetensors")))


def _log_to_kb(name: str, attempt: int, result: dict[str, Any], steps: int) -> None:
    """Best-effort KB record; never fails the stage."""
    try:
        from master_agent.kb.store import COLLECTION_LORA_RUNS, kb_available, upsert_docs

        if not kb_available():
            return
        ts = datetime.now(timezone.utc).isoformat()
        text = (
            f"lora train {name} attempt {attempt}: ok={result.get('ok')} "
            f"steps={steps} lora={result.get('lora_path', '')} "
            f"error={str(result.get('error', ''))[:200]}"
        )
        upsert_docs(
            COLLECTION_LORA_RUNS,
            ids=[f"lora:{name}:attempt:{attempt}"],
            texts=[text],
            metadatas=[
                {
                    "name": name,
                    "attempt": attempt,
                    "steps": int(steps),
                    "ok": bool(result.get("ok")),
                    "timestamp": ts,
                }
            ],
        )
    except Exception:
        pass


def train_lora(
    name: str,
    overrides: dict | None = None,
    resume: bool = True,
    client=None,
    runner=None,
) -> dict:
    """Train the character LoRA; returns {"ok", "lora_path", "steps", "log"}."""
    char_dir = CHARACTERS_DIR / name
    char_json = char_dir / "character.json"
    dataset_dir = char_dir / "dataset"
    if not char_json.is_file():
        return {"ok": False, "error": f"missing character.json for '{name}' — run the CCC stage first"}
    if not dataset_dir.is_dir() or not list(dataset_dir.glob("*.png")):
        return {"ok": False, "error": f"missing dataset images for '{name}' — run build_dataset first"}
    if not (AI_TOOLKIT_DIR / "run.py").is_file():
        return {"ok": False, "error": _SETUP_HINT}
    if resolve_model_path(FLUX_UNET) is None:
        return {"ok": False, "error": f"flux unet not found: {FLUX_UNET} (expected under models/diffusion_models)"}

    if client is not None:
        try:
            client.free_memory()
        except Exception as e:
            print(f"[lora] free_memory warning: {e}")

    import json

    character = json.loads(char_json.read_text(encoding="utf-8"))
    config_text = render_train_config(character, overrides)
    config_dir = AI_TOOLKIT_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{name}.yml").write_text(config_text, encoding="utf-8")

    steps = int((overrides or {}).get("steps") or LORA_DEFAULTS["steps"])
    rank = int((overrides or {}).get("rank") or LORA_DEFAULTS["rank"])
    log_path = char_dir / "train.log"
    cmd = [str(_toolkit_python()), "run.py", f"config/{name}.yml"]
    run = runner or _default_runner
    rc = run(cmd, log_path)

    attempt = _attempt_number(name)
    if rc != 0:
        result = {"ok": False, "error": f"training exited rc={rc}: {_log_tail(log_path)}", "log": str(log_path)}
        _log_to_kb(name, attempt, result, steps)
        return result

    out_dir = AI_TOOLKIT_DIR / "output" / name
    candidates = sorted(out_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime) if out_dir.is_dir() else []
    if not candidates:
        result = {"ok": False, "error": f"no safetensors produced under {out_dir}", "log": str(log_path)}
        _log_to_kb(name, attempt, result, steps)
        return result

    loras_dir = MODELS_DIR / "loras"
    loras_dir.mkdir(parents=True, exist_ok=True)
    dest = loras_dir / f"{name}_r{rank}.safetensors"
    shutil.copy2(candidates[-1], dest)
    result = {"ok": True, "lora_path": str(dest), "steps": steps, "log": str(log_path)}
    _log_to_kb(name, attempt, result, steps)
    return result
