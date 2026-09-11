"""Diagnose hull: 9-frame short fire that records sec/step before any scale."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from master_agent.config import (
    DIAGNOSE_FRAMES,
    DIAGNOSE_HEIGHT,
    DIAGNOSE_SEED,
    DIAGNOSE_STEPS,
    DIAGNOSE_STEPS_MAX,
    DIAGNOSE_STEPS_MIN,
    DIAGNOSE_WIDTH,
    OUTPUTS_DIR,
    STATE_DIR,
)
from master_agent.judge.probe import TINY_FILE_BYTES

MIN_DIAGNOSE_FRAMES = 3
HULL_FILENAME = "diagnose_hull.json"


class ScaleRefused(RuntimeError):
    """Scale is blocked until a diagnose hull has recorded sec/step."""


class DiagnoseFailed(RuntimeError):
    """Diagnose output is junk (<100KB or <3 frames) or lint/fire failed."""


def clamp_diagnose_steps(steps: int | None = None) -> int:
    value = DIAGNOSE_STEPS if steps is None else int(steps)
    return max(DIAGNOSE_STEPS_MIN, min(DIAGNOSE_STEPS_MAX, value))


def hull_path() -> Path:
    return STATE_DIR / "control" / HULL_FILENAME


def load_hull(path: Path | None = None) -> dict[str, Any] | None:
    target = Path(path) if path is not None else hull_path()
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def hull_has_sec_per_step(hull: dict[str, Any] | None = None, *, path: Path | None = None) -> bool:
    data = hull if hull is not None else load_hull(path)
    if not data:
        return False
    try:
        return float(data.get("sec_per_step")) >= 0.0
    except (TypeError, ValueError):
        return False


def record_hull(
    *,
    sec_per_step: float,
    wall_s: float,
    steps: int,
    frames: int,
    path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "sec_per_step": round(float(sec_per_step), 4),
        "wall_s": round(float(wall_s), 4),
        "steps": int(steps),
        "frames": int(frames),
    }
    if extra:
        payload.update(extra)
    target = Path(path) if path is not None else hull_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return payload


def refuse_scale_until_hull(
    *,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    hull_file: Path | None = None,
) -> None:
    if hull_has_sec_per_step(path=hull_file):
        return
    w = int(width or 0)
    h = int(height or 0)
    f = int(frames or 0)
    scaled = w > DIAGNOSE_WIDTH or h > DIAGNOSE_HEIGHT or f > DIAGNOSE_FRAMES
    if scaled:
        raise ScaleRefused(
            "refuse scale until diagnose hull has recorded sec/step; "
            'run: python -m master_agent diagnose --variant base --prompt "garden proof"'
        )


def diagnose_preset(
    *,
    prompt: str = "garden proof",
    variant: str = "base",
    steps: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    return {
        "variant": variant or "base",
        "prompt": prompt or "garden proof",
        "frames": DIAGNOSE_FRAMES,
        "steps": clamp_diagnose_steps(steps),
        "seed": DIAGNOSE_SEED if seed is None else int(seed),
        "width": DIAGNOSE_WIDTH,
        "height": DIAGNOSE_HEIGHT,
    }


def prepare_diagnose_workflow(
    *,
    prompt: str = "garden proof",
    variant: str = "base",
    steps: int | None = None,
    seed: int | None = None,
    prepare_fn: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preset = diagnose_preset(prompt=prompt, variant=variant, steps=steps, seed=seed)
    if prepare_fn is None:
        from master_agent.comfy.workflow_patcher import load_and_patch_workflow

        prepare_fn = load_and_patch_workflow
    workflow, meta = prepare_fn(
        preset["variant"],
        prompt=preset["prompt"],
        frames=preset["frames"],
        steps=preset["steps"],
        seed=preset["seed"],
        width=preset["width"],
        height=preset["height"],
    )
    meta = dict(meta or {})
    meta.update(preset)
    return workflow, meta


def _probe_output(video_path: str | Path | None) -> dict[str, Any]:
    from master_agent.judge.probe import probe_video

    return probe_video(video_path)


def _is_junk(probe: dict[str, Any]) -> bool:
    size = int(probe.get("size_bytes") or 0)
    frames = probe.get("frames")
    if size < TINY_FILE_BYTES:
        return True
    if frames is not None and int(frames) < MIN_DIAGNOSE_FRAMES:
        return True
    return False


def run_diagnose(
    *,
    prompt: str = "garden proof",
    variant: str = "base",
    steps: int | None = None,
    seed: int | None = None,
    prepare_only: bool = False,
    object_info: dict[str, Any] | None = None,
    fire: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    probe_fn: Callable[[str | None], dict[str, Any]] | None = None,
    budget: Any | None = None,
    hull_file: Path | None = None,
    copy_dir: Path | None = None,
    log: Callable[[str], None] = print,
    prepare_fn: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Prepare + lint, then optional short fire. Never increments shift budget."""
    refuse_scale_until_hull(
        width=width,
        height=height,
        frames=DIAGNOSE_FRAMES if width or height else None,
        hull_file=hull_file,
    )
    used_before = None if budget is None else float(budget.used)
    workflow, meta = prepare_diagnose_workflow(
        prompt=prompt,
        variant=variant,
        steps=steps,
        seed=seed,
        prepare_fn=prepare_fn,
    )
    steps_used = int(meta["steps"])
    log(
        f"DIAGNOSE  variant={meta['variant']} frames={meta['frames']} "
        f"steps={steps_used} seed={meta['seed']}"
    )

    if object_info is None:
        from master_agent.comfy.client import ComfyClient

        object_info, source = ComfyClient().load_object_info(prefer_live=True)
        log(f"object_info: {source}")

    from master_agent.comfy.cli_run import lint_or_raise

    lint_or_raise(workflow, object_info, file_label="diagnose")
    log("lint: PASS")

    result: dict[str, Any] = {
        "ok": True,
        "status": "prepared",
        "charged_budget": False,
        "meta": meta,
        "frames": meta["frames"],
        "steps": steps_used,
        "seed": meta["seed"],
    }
    if prepare_only:
        log("prepare-only: no GPU fire, shift budget untouched")
        result["used_delta"] = 0.0
        return result

    if fire is None:
        from master_agent.comfy.cli_run import execute_prepared

        fire = lambda wf: execute_prepared(wf, run_id="diagnose")

    t0 = time.perf_counter()
    fired = fire(workflow) or {}
    wall_s = float(fired.get("wall_s") or (time.perf_counter() - t0))
    sec_per_step = wall_s / max(steps_used, 1)
    log(f"wall_s={wall_s:.3f} sec/step={sec_per_step:.3f}")

    video_path = fired.get("video_path")
    dest = video_path
    if video_path:
        src = Path(video_path)
        out_root = Path(copy_dir) if copy_dir is not None else OUTPUTS_DIR
        out_root.mkdir(parents=True, exist_ok=True)
        if src.is_file() and src.parent.resolve() != out_root.resolve():
            dest_path = out_root / src.name
            dest_path.write_bytes(src.read_bytes())
            dest = str(dest_path)
        log(f"output: {dest}")

    probe = (probe_fn or _probe_output)(dest)
    frames_out = probe.get("frames")
    log(f"ffprobe frames={frames_out} size_bytes={probe.get('size_bytes')}")
    if _is_junk(probe):
        result.update(
            {
                "ok": False,
                "status": "FAIL",
                "reason": "junk output (<100KB or <3 frames)",
                "probe": probe,
                "wall_s": wall_s,
                "sec_per_step": sec_per_step,
                "video_path": dest,
            }
        )
        if used_before is not None:
            result["used_delta"] = float(budget.used) - used_before
        raise DiagnoseFailed(result["reason"])

    hull = record_hull(
        sec_per_step=sec_per_step,
        wall_s=wall_s,
        steps=steps_used,
        frames=int(frames_out or meta["frames"]),
        path=hull_file,
        extra={"variant": meta["variant"], "video_path": dest},
    )
    log(f"HULL recorded sec/step={hull['sec_per_step']}")
    result.update(
        {
            "status": "done",
            "wall_s": wall_s,
            "sec_per_step": sec_per_step,
            "video_path": dest,
            "probe": probe,
            "hull": hull,
        }
    )
    if used_before is not None:
        result["used_delta"] = float(budget.used) - used_before
        result["charged_budget"] = result["used_delta"] != 0.0
    else:
        result["used_delta"] = 0.0
    return result
