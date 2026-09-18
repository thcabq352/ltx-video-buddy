"""Top-level music-video render: plan → Comfy burn → unique → Remotion."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from master_agent.music.burn import burn_windows
from master_agent.music.plan import (
    BEAT_PLAN_SCHEMA,
    BeatPlan,
    build_beat_plan,
    beat_plan_from_dict,
    read_beat_plan,
    write_beat_plan,
)
from master_agent.music.remotion import DEFAULT_OUT_NAME, assemble_remotion
from master_agent.music.unique import DuplicateClipError, check_unique_clips

DEFAULT_OUT = Path("out") / DEFAULT_OUT_NAME


def render_music_video(
    audio: str | Path,
    *,
    out: str | Path | None = None,
    prompt: str = "music video",
    image: str | Path | None = None,
    variant: str | None = None,
    dry_run: bool = False,
    plan: BeatPlan | dict[str, Any] | str | Path | None = None,
    seed: int | None = None,
    width: int = 768,
    height: int = 512,
    fps: int = 30,
    work_dir: str | Path | None = None,
    log=print,
) -> dict[str, Any]:
    """Run plan → unique Comfy/LTX burns → uniqueness gate → Remotion stitch.

    ``dry_run=True`` mocks burns (unique motion placeholders) and writes
    Remotion props/command without GPU, Comfy, or Node.
    """
    audio_path = Path(audio)
    out_path = Path(out) if out else DEFAULT_OUT
    run_id = uuid.uuid4().hex[:12]
    dest = Path(work_dir) if work_dir else out_path.parent / f"mv-{run_id}"
    dest.mkdir(parents=True, exist_ok=True)
    base_seed = 7 if seed is None else int(seed)

    if isinstance(plan, (str, Path)):
        beat = beat_plan_from_dict(read_beat_plan(plan))
    elif isinstance(plan, BeatPlan):
        beat = plan
    elif isinstance(plan, dict):
        beat = beat_plan_from_dict(plan)
    else:
        log(f"planning beats from {audio_path}")
        beat = build_beat_plan(audio_path, fps=fps)
    if beat.audio_path is None:
        beat.audio_path = str(audio_path)

    plan_path = write_beat_plan(beat, dest / "beat_plan.json")
    log(
        f"beat plan: {beat.bpm:.0f} BPM, {len(beat.windows)} windows, "
        f"{beat.duration_s:.2f}s @ {beat.fps}fps"
    )

    burns = burn_windows(
        beat,
        brief=prompt,
        out_dir=dest,
        variant=variant,
        image=image,
        seed=base_seed,
        width=width,
        height=height,
        dry_run=dry_run,
        run_id=run_id,
        log=log,
    )
    clip_pairs = [(int(row["index"]), Path(row["clip"])) for row in burns]
    try:
        hashes = check_unique_clips(clip_pairs)
    except DuplicateClipError as exc:
        log(f"FAIL  uniqueness gate: {exc}")
        raise

    clips = {int(row["index"]): Path(row["clip"]) for row in burns}
    remotion = assemble_remotion(
        beat,
        clips,
        audio_path,
        out_path=out_path,
        work_dir=dest,
        dry_run=dry_run,
        log=log,
    )

    record: dict[str, Any] = {
        "ok": True,
        "kind": "music_video_mv",
        "engine": "comfy-ltx",
        "schema": BEAT_PLAN_SCHEMA,
        "run_id": run_id,
        "dry_run": dry_run,
        "audio": str(audio_path),
        "out": str(out_path),
        "work_dir": str(dest),
        "plan_path": str(plan_path),
        "beat_plan": beat.to_dict(),
        "burns": burns,
        "hashes": hashes,
        "remotion": remotion,
        "variant": burns[0]["variant"] if burns else variant,
        "image": str(image) if image else None,
        "prompt": prompt,
    }
    rec_path = dest / "mv-record.json"
    rec_path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    record["record_path"] = str(rec_path)
    log(f"mv record: {rec_path}")
    if dry_run:
        log(f"DRY-RUN  plan+unique+remotion wired → {remotion.get('props_path')}")
    else:
        log(f"DONE  {out_path}")
    return record
