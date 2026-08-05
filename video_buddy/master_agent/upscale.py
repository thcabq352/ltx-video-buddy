"""Post-stage video upscaling via ComfyUI.

Two methods:
  rtx     — NVIDIA RTX Video Super Resolution (fast, fixed-function hardware
            path via the RTX nodes pack). Patches the validated Mickmumpitz
            RTX-SR workflow: node 12 (VHS_LoadVideo) + node 14 (resolution).
  seedvr2 — SeedVR2 diffusion upscaler (slow, production-grade). Minimal
            VHS_LoadVideo -> SeedVR2 -> VHS_VideoCombine chain using the
            models/SEEDVR2/ weights.

Both reuse the Orchestrator's submit/poll/resolve machinery — nothing new.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from master_agent.comfy.client import ComfyClient, ComfyClientError
from master_agent.comfy.validator import validate_workflow
from master_agent.config import OUTPUTS_DIR, WORKFLOWS_DIR

RTX_WORKFLOW = (
    WORKFLOWS_DIR
    / "AI-RENDERING-EXAMPLE FILES"
    / "260330_MICKMUMPITZ_NVIDIA-RTX-SUPER-RESOLUTION_1-0_api.json"
)
SEEDVR2_WORKFLOW = WORKFLOWS_DIR / "upscale_seedvr2_api.json"


def _patch_workflow(
    method: str,
    video_name: str,
    *,
    run_id: str,
    rtx_resolution: str,
    seedvr2_resolution: int,
) -> dict:
    if method == "rtx":
        wf = json.loads(RTX_WORKFLOW.read_text(encoding="utf-8"))
        wf["12"]["inputs"]["video"] = video_name
        wf["14"]["inputs"]["resolution"] = rtx_resolution
        wf["17"]["inputs"]["filename_prefix"] = f"rtx_upscale_{run_id}"
        return wf
    if method == "seedvr2":
        wf = json.loads(SEEDVR2_WORKFLOW.read_text(encoding="utf-8"))
        wf["1"]["inputs"]["video"] = video_name
        wf["4"]["inputs"]["resolution"] = seedvr2_resolution
        wf["6"]["inputs"]["filename_prefix"] = f"seedvr2_upscale_{run_id}"
        return wf
    raise ValueError(f"unknown upscale method: {method!r} (rtx | seedvr2)")


def upscale_video(
    video_path: str | Path,
    *,
    method: str = "rtx",
    run_id: Optional[str] = None,
    rtx_resolution: str = "2160p",
    seedvr2_resolution: int = 1080,
    client: Optional[ComfyClient] = None,
    log=print,
) -> Path:
    """Upscale a video through ComfyUI; returns the resolved output path."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    client = client or ComfyClient()
    run_id = run_id or video_path.stem[:12]

    video_name = client.upload_image(video_path)
    log(f"upscale [{method}]: uploaded {video_name}")
    wf = _patch_workflow(
        method,
        video_name,
        run_id=run_id,
        rtx_resolution=rtx_resolution,
        seedvr2_resolution=seedvr2_resolution,
    )

    object_info, source = client.load_object_info(prefer_live=True)
    report = validate_workflow(
        wf, object_info, file_label=f"upscale:{method}", object_info_source=source
    )
    if not report.ok:
        details = "; ".join(str(i) for i in report.errors[:5])
        raise RuntimeError(f"upscale workflow validation failed: {details}")

    client.free_memory()
    prompt_id = client.queue_prompt(wf)
    log(f"upscale [{method}]: queued prompt_id={prompt_id}")
    try:
        entry = client.wait_for_prompt(prompt_id)
    except ComfyClientError as e:
        raise RuntimeError(f"upscale job failed: {e}") from e
    files = ComfyClient.extract_video_files(entry)
    if not files:
        raise RuntimeError("upscale job completed but produced no video files")

    out_dir = OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ComfyClient.resolve_output_path(files[0])
    dest = out_dir / f"{video_path.stem}_{method}{video_path.suffix}"
    shutil.copy2(src, dest)
    log(f"upscale [{method}]: {dest}")
    return dest
