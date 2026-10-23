"""``flux_ltx_i2v``: a Klein 4B still, then an LTX 2.5 image-to-video clip.

The slug stays. Its still generator is FLUX.2 [klein] 4B (Apache 2.0), not a
Flux.1 GGUF. The video stage is ``ltx25_t2v_i2v``.
"""

from __future__ import annotations

from typing import Any

from master_agent.models.klein import DIFFUSION_FILE, LICENSE, LICENSE_LINE

SLUG = "flux_ltx_i2v"
STILL_VARIANT = "flux"
VIDEO_VARIANT = "ltx25_t2v_i2v"


def plan(request: str = "") -> dict[str, Any]:
    return {
        "slug": SLUG,
        "request": request,
        "still_variant": STILL_VARIANT,
        "still_generator": DIFFUSION_FILE,
        "video_variant": VIDEO_VARIANT,
        "license": LICENSE,
        "note": LICENSE_LINE,
    }


def uses_klein_4b(spec: dict[str, Any] | None = None) -> bool:
    row = spec or plan()
    name = str(row.get("still_generator") or "")
    banned = ("flux1", "flux.1", "9b", "schnell", "kontext")
    return "flux-2-klein-4b" in name and not any(token in name.lower() for token in banned)


def begin(
    request: str,
    *,
    dry_run: bool,
    image_name: str | None,
    seed: int | None,
    width: int,
    height: int,
) -> tuple[str, str | None]:
    """Point the video stage at LTX 2.5. A live run renders the Klein still first."""
    if dry_run or image_name:
        return VIDEO_VARIANT, image_name
    return VIDEO_VARIANT, render_still(
        request, seed=seed, width=width, height=height
    )


def render_still(
    request: str,
    *,
    seed: int | None,
    width: int,
    height: int,
) -> str:
    """Queue one Klein 4B still and return the image filename Comfy saved."""
    from master_agent.comfy.client import ComfyClient
    from master_agent.comfy.workflow_patcher import load_and_patch_workflow

    workflow, _meta = load_and_patch_workflow(
        STILL_VARIANT,
        prompt=request or "a still frame",
        width=width,
        height=height,
        seed=seed,
        steps=4,
    )
    client = ComfyClient()
    prompt_id = client.queue_prompt(workflow)
    entry = client.wait_for_prompt(prompt_id)
    outputs = entry.get("outputs") or {}
    for node in outputs.values():
        if not isinstance(node, dict):
            continue
        for image in node.get("images") or []:
            if isinstance(image, dict) and image.get("filename"):
                return str(image["filename"])
    raise RuntimeError("flux_ltx_i2v: the Klein 4B still produced no image")
