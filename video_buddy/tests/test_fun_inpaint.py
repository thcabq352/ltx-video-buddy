"""Wan Fun Inpaint API template: mask source + patcher fields. No GPU.

Run: python -m pytest tests/test_fun_inpaint.py -q
"""

from __future__ import annotations

import json

from master_agent.comfy.cli_run import resolve_template
from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.config import WORKFLOW_FILES, WORKFLOWS_DIR
from master_agent.orchestrator.director import rule_based_variant


def _node(wf: dict, class_type: str) -> tuple[str, dict]:
    matches = [
        (nid, node)
        for nid, node in wf.items()
        if isinstance(node, dict) and node.get("class_type") == class_type
    ]
    assert len(matches) == 1, class_type
    return matches[0]


def test_template_wires_mask_into_fun_inpaint():
    path = WORKFLOWS_DIR / WORKFLOW_FILES["wan_fun_inpaint"]
    assert path.name == "wan_fun_inpaint_api.json"
    assert resolve_template("wan_fun_inpaint") == path
    wf = json.loads(path.read_text(encoding="utf-8"))
    _image_id, image = _node(wf, "LoadImage")
    mask_id, mask = _node(wf, "LoadImageMask")
    fun_id, fun = _node(wf, "WanFunInpaintToVideo")
    _noise_id, noise = _node(wf, "SetLatentNoiseMask")
    _sampler_id, sampler = _node(wf, "KSampler")
    assert fun["inputs"]["start_image"] == ["6", 0]
    assert image["inputs"]["image"] == "fun_inpaint_start.png"
    assert mask["inputs"]["channel"] == "red"
    assert noise["inputs"]["mask"] == [mask_id, 0]
    assert noise["inputs"]["samples"] == [fun_id, 2]
    assert sampler["inputs"]["positive"] == [fun_id, 0]
    assert sampler["inputs"]["latent_image"][0] == "9"
    blob = json.dumps(wf).lower()
    assert "k3nk" not in blob


def test_patcher_sets_prompt_image_mask_and_length():
    wf, meta = load_and_patch_workflow(
        "wan_fun_inpaint",
        prompt="repair the torn poster",
        image_name="plate.png",
        mask_name="hole.png",
        frames=41,
        seed=7,
        width=832,
        height=480,
    )
    texts = [
        node["inputs"]["text"]
        for node in wf.values()
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode"
    ]
    assert "repair the torn poster" in texts
    _iid, image = _node(wf, "LoadImage")
    _mid, mask = _node(wf, "LoadImageMask")
    _fid, fun = _node(wf, "WanFunInpaintToVideo")
    assert image["inputs"]["image"] == "plate.png"
    assert mask["inputs"]["image"] == "hole.png"
    assert fun["inputs"]["length"] == 41
    assert fun["inputs"]["width"] == 768
    assert fun["inputs"]["height"] == 480
    assert meta["frames"] == 41
    assert meta["seed"] == 7


def test_default_mask_png_is_rgb_with_a_white_center():
    from master_agent.comfy.fun_inpaint import default_inpaint_mask_png

    png = default_inpaint_mask_png(16)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in png and b"IEND" in png
    # 8-bit RGB (color type 2) is the 10th IHDR byte after width/height.
    ihdr = png.split(b"IHDR", 1)[1][:13]
    assert ihdr[8] == 8
    assert ihdr[9] == 2


def test_ensure_uploads_placeholder_and_keeps_a_supplied_mask():
    from master_agent.comfy.fun_inpaint import (
        FUN_INPAINT_MASK_NAME,
        ensure_fun_inpaint_mask,
    )

    wf, _meta = load_and_patch_workflow(
        "wan_fun_inpaint",
        prompt="repair the torn poster",
        image_name="fun_inpaint_start.png",
        seed=1,
        frames=17,
    )
    seen: list[str] = []

    def _upload(path):
        seen.append(path.name)
        assert path.read_bytes().startswith(b"\x89PNG")
        return path.name

    uploaded = ensure_fun_inpaint_mask(wf, _upload)
    assert uploaded == FUN_INPAINT_MASK_NAME
    assert seen == [FUN_INPAINT_MASK_NAME]
    _mid, mask = _node(wf, "LoadImageMask")
    assert mask["inputs"]["image"] == FUN_INPAINT_MASK_NAME

    supplied, _meta = load_and_patch_workflow(
        "wan_fun_inpaint",
        prompt="repair the torn poster",
        image_name="plate.png",
        mask_name="hole.png",
        seed=1,
        frames=17,
    )
    assert ensure_fun_inpaint_mask(supplied, _upload) is None
    assert seen == [FUN_INPAINT_MASK_NAME]
    _mid, mask = _node(supplied, "LoadImageMask")
    assert mask["inputs"]["image"] == "hole.png"


def test_director_routes_inpaint_and_leaves_wan22():
    assert rule_based_variant("fun inpaint the torn poster") == "wan_fun_inpaint"
    assert rule_based_variant("please inpaint this plate") == "wan_fun_inpaint"
    assert rule_based_variant("heal mask on the face") == "wan_fun_inpaint"
    assert rule_based_variant("wan 2.2 photoreal film grain") == "wan22"
    assert "wan_fun_inpaint" in WORKFLOW_FILES
