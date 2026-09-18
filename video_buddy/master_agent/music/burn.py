"""Comfy/LTX clip burn loop — one unique motion clip per beat window.

Reuses ``prepare_run`` (TeaCache inject-when-registered) and
``execute_prepared``. Dry-run writes unique motion placeholders +
``buddy.clip.provenance/v1`` sidecars and never queues GPU.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from master_agent.music.plan import BeatPlan, MV_FPS
from master_agent.music.still_hold import (
    MOTION_MARKER,
    StillHoldError,
    assert_clip_not_still_hold,
    assert_window_not_still_hold,
)
from master_agent.provenance import (
    CLIP_PROVENANCE_SCHEMA,
    persist_clip_provenance,
    read_clip_provenance,
    write_clip_provenance,
)

DEFAULT_MV_VARIANT = "ltx25_t2v_i2v"
MOTION_VERBS = (
    "handheld push-in, subject in motion",
    "slow dolly left, hair and cloth moving",
    "orbiting camera, walking subject",
    "crane rise, crowd and lights shifting",
    "whip-pan into a moving close-up",
    "steadicam follow, feet hitting the beat",
)


class BurnError(RuntimeError):
    """A window failed to burn a unique motion clip."""


def seed_for_window(base_seed: int, index: int) -> int:
    return (int(base_seed) + int(index) * 1009) & 0xFFFFFFFF


def resolve_mv_variant(variant: str | None, image: str | Path | None) -> str:
    """I2V when a still / character lock is provided; T2V otherwise.

    ``ltx25_t2v_i2v`` is the same graph for both. H3 T2V flips to ``h3_i2v``.
    """
    chosen = (variant or DEFAULT_MV_VARIANT).strip() or DEFAULT_MV_VARIANT
    if image and chosen in {"h3_t2v", "fl2va"}:
        return "h3_i2v"
    return chosen


def window_motion_prompt(brief: str, window: dict[str, Any]) -> str:
    """Force real motion language so the burn cannot be a still-hold."""
    verb = MOTION_VERBS[int(window.get("index") or 0) % len(MOTION_VERBS)]
    label = window.get("label") or "verse"
    idx = window.get("index", 0)
    return (
        f"{brief}, {label}, {verb}, window {idx}, "
        "no still hold, no freeze-frame, real motion"
    )


def clip_path_for_window(out_dir: str | Path, window: dict[str, Any]) -> Path:
    idx = int(window["index"])
    return Path(out_dir) / "clips" / f"window-{idx:03d}.mp4"


def write_dry_motion_clip(
    dest: str | Path,
    window: dict[str, Any],
    prompt: str,
) -> Path:
    """Unique-bytes motion placeholder. Not a still. No GPU."""
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = MOTION_MARKER + b"\n" + json.dumps(
        {
            "window": window.get("index"),
            "start_s": window.get("start_s"),
            "end_s": window.get("end_s"),
            "start_frame": window.get("start_frame"),
            "end_frame": window.get("end_frame"),
            "prompt": prompt,
        },
        sort_keys=True,
    ).encode("utf-8")
    path.write_bytes(payload)
    return path


def _write_window_provenance(
    *,
    clip: Path,
    brief: str,
    prompt: str,
    variant: str,
    seed: int,
    window: dict[str, Any],
    image_name: str | None,
    width: int,
    height: int,
    run_id: str,
    steps: int | None = None,
    cfg: float | None = None,
) -> dict[str, Any]:
    from types import SimpleNamespace

    idx = int(window["index"])
    st = SimpleNamespace(
        request=brief,
        prompt=prompt,
        run_id=run_id,
        variant=variant,
        seed=seed,
        steps=steps,
        cfg=cfg,
        width=width,
        height=height,
        duration_s=float(window["end_s"]) - float(window["start_s"]),
        fps=MV_FPS,
        shot_index=idx + 1,
        shot_id=f"window-{idx}",
        output_dir=str(clip.parent),
        planned_clip=str(clip),
        video_path=str(clip),
        image_name=image_name,
        kind="music_video",
        music_bed_attached=True,
    )
    payload = persist_clip_provenance(st, path=clip)
    payload["schema"] = CLIP_PROVENANCE_SCHEMA
    payload["window_id"] = idx
    lineage = dict(payload.get("lineage") or {})
    lineage["window_id"] = idx
    lineage["shot_id"] = f"window-{idx}"
    lineage["attempt_id"] = f"window-{idx}.a{int(getattr(st, 'attempt', 1) or 1)}"
    payload["lineage"] = lineage
    write_clip_provenance(clip, payload)
    return payload


def prepare_window_graph(
    window: dict[str, Any],
    *,
    prompt: str,
    variant: str,
    seed: int,
    width: int,
    height: int,
    image_name: str | None,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Patch the existing LTX generate path (TeaCache via ``prepare_run``)."""
    from master_agent.comfy.cli_run import prepare_run

    duration_s = max(float(window["end_s"]) - float(window["start_s"]), 1.0)
    idx = int(window["index"])
    return prepare_run(
        "generate",
        variant=variant,
        prompt=prompt,
        object_info=object_info,
        duration_s=duration_s,
        seed=seed,
        width=width,
        height=height,
        image_name=image_name,
        filename_prefix=f"mv_w{idx:03d}",
    )


def burn_windows(
    plan: BeatPlan | dict[str, Any],
    *,
    brief: str,
    out_dir: str | Path,
    variant: str | None = None,
    image: str | Path | None = None,
    seed: int = 7,
    width: int = 768,
    height: int = 512,
    dry_run: bool = False,
    run_id: str = "mv",
    object_info: dict[str, Any] | None = None,
    execute: Optional[Callable[..., dict[str, Any]]] = None,
    upload_image: Optional[Callable[[Path], str]] = None,
    log=print,
) -> list[dict[str, Any]]:
    """Burn one unique motion clip per window. Hard-fail still-holds.

    ``dry_run=True`` writes unique motion placeholders + provenance and still
    calls ``prepare_run`` (graph patch / TeaCache) so wiring is exercised
    without Comfy or a GPU.
    """
    data = plan.to_dict() if isinstance(plan, BeatPlan) else plan
    windows = list(data.get("windows") or [])
    if not windows:
        raise StillHoldError("beat plan has no windows")
    chosen = resolve_mv_variant(variant, image)
    image_name = Path(image).name if image else None
    dest_root = Path(out_dir)
    dest_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    if image and not dry_run and upload_image is None:
        from master_agent.comfy.client import ComfyClient

        client = ComfyClient()
        image_name = client.upload_image(Path(image))
        log(f"uploaded character lock → {image_name}")
    elif image and upload_image is not None:
        image_name = upload_image(Path(image))

    for window in windows:
        assert_window_not_still_hold(window, fps=int(data.get("fps") or MV_FPS))
        idx = int(window["index"])
        prompt = window_motion_prompt(brief, window)
        win_seed = seed_for_window(seed, idx)
        clip = clip_path_for_window(dest_root, window)
        log(f"burn window {idx}: {window['start_s']:.2f}-{window['end_s']:.2f}s {chosen}")

        # Always go through prepare_run so TeaCache / LTX graph path is the
        # same as diagnose / draft / generate. Dry-run skips the GPU queue.
        try:
            wf = prepare_window_graph(
                window,
                prompt=prompt,
                variant=chosen,
                seed=win_seed,
                width=width,
                height=height,
                image_name=image_name,
                object_info=object_info,
            )
        except Exception as exc:
            if dry_run:
                log(f"prepare_run skipped for window {idx} ({exc})")
                wf = {}
            else:
                raise BurnError(f"window {idx}: prepare_run failed: {exc}") from exc

        if dry_run:
            write_dry_motion_clip(clip, window, prompt)
        else:
            runner = execute
            if runner is None:
                from master_agent.comfy.cli_run import execute_prepared

                runner = execute_prepared
            rec = runner(wf, variant=chosen, run_id=f"{run_id}_w{idx:03d}")
            src = rec.get("video_path")
            if not src or not Path(src).is_file():
                raise BurnError(f"window {idx}: Comfy produced no video")
            clip.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, clip)

        assert_clip_not_still_hold(clip, window, dry_run=dry_run)
        provenance = _write_window_provenance(
            clip=clip,
            brief=brief,
            prompt=prompt,
            variant=chosen,
            seed=win_seed,
            window=window,
            image_name=image_name,
            width=width,
            height=height,
            run_id=run_id,
        )
        loaded = read_clip_provenance(clip)
        if not loaded or loaded.get("schema") != CLIP_PROVENANCE_SCHEMA:
            raise BurnError(f"window {idx}: provenance sidecar missing")
        records.append(
            {
                "index": idx,
                "clip": str(clip),
                "prompt": prompt,
                "variant": chosen,
                "seed": win_seed,
                "nodes": len(wf) if isinstance(wf, dict) else 0,
                "hash": provenance.get("hash"),
                "window_id": idx,
                "sidecar": str(clip.with_name(f"{clip.stem}.buddy.json")),
            }
        )
    return records
