"""Four LTX-2.3 render variants resolve to the on-disk dev fp8 checkpoint.

Node 4826 is LTXAVTextEncoderLoader. These variants used to miss MODEL_FILES
and fall back to base, which sprayed the 10Eros bake (and overwrote a JSON
edit of the truncated distilled fp8 name). The same graphs must keep their
authored sampler and positive-prompt link, write the run prompt into node
6203, and must not gain a TeaCache node.
No GPU.

Run: python -m pytest tests/test_ltx23_dev_checkpoint.py -q
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from master_agent.comfy.cli_run import prepare_run
from master_agent.comfy.workflow_patcher import (
    load_and_patch_workflow,
    load_workflow_template,
)
from master_agent.config import DEFAULT_ALL_IN_ONE_CKPT, MODEL_FILES

DEV_FP8 = "ltx-2.3-22b-dev-fp8.safetensors"
TRUNCATED_DISTILLED = "ltx-2.3-22b-distilled-fp8.safetensors"
BAKE_10EROS = "LTX2.3_DISTILLED-1.1_BAKED_LTX_10Eros_v14_r768.safetensors"
DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
RENDER_VARIANTS = (
    "ltx23_lipsync_v08",
    "air_render_030",
    "air_render_050",
    "air_render_businesswoman",
)


def _resolve_on_tower(name: str | None) -> Path | None:
    """Dev fp8 is on the tower; the truncated file and the 10Eros bake are not chosen."""
    if name == DEV_FP8:
        return Path("checkpoints") / DEV_FP8
    return None


def _ckpt_names(workflow: dict) -> list[str]:
    names: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ckpt = (node.get("inputs") or {}).get("ckpt_name")
        if isinstance(ckpt, str):
            names.append(ckpt)
    return names


def _lora_names(workflow: dict) -> list[str]:
    names: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        lora = (node.get("inputs") or {}).get("lora_name")
        if isinstance(lora, str):
            names.append(lora)
    return names


def test_render_variants_have_their_own_checkpoint_entry():
    assert MODEL_FILES["base"]["checkpoint"] == DEFAULT_ALL_IN_ONE_CKPT
    assert DEFAULT_ALL_IN_ONE_CKPT == BAKE_10EROS
    for variant in RENDER_VARIANTS:
        files = MODEL_FILES[variant]
        assert files.get("checkpoint") == DEV_FP8
        assert TRUNCATED_DISTILLED not in files.values()
        assert BAKE_10EROS not in files.values()
        # A lora key would replace the graph's distilled LoRA and IC-LoRAs.
        assert "lora" not in files


@pytest.mark.parametrize("variant", RENDER_VARIANTS)
def test_patched_node_4826_uses_dev_fp8(variant: str):
    template = load_workflow_template("ltx23_lipsync_v08")
    assert template["4826"]["class_type"] == "LTXAVTextEncoderLoader"
    assert template["4826"]["inputs"]["ckpt_name"] == TRUNCATED_DISTILLED

    try:
        own = load_workflow_template(variant)
    except FileNotFoundError:
        own = None

    def _load(requested: str):
        if requested == variant and own is not None:
            return copy.deepcopy(own)
        if requested == variant:
            # air_render JSON is gitignored; it is the same node-4826 graph.
            return copy.deepcopy(template)
        return load_workflow_template(requested)

    with (
        patch(
            "master_agent.comfy.workflow_patcher.resolve_model_path",
            side_effect=_resolve_on_tower,
        ),
        patch(
            "master_agent.comfy.workflow_patcher.load_workflow_template",
            side_effect=_load,
        ),
    ):
        workflow, meta = load_and_patch_workflow(
            variant,
            prompt="clay figure speaks",
            seed=1,
            duration_s=1.0,
        )

    assert meta["checkpoint"] == DEV_FP8
    assert meta["checkpoint_preferred"] == DEV_FP8
    node = workflow["4826"]
    assert node["class_type"] == "LTXAVTextEncoderLoader"
    assert node["inputs"]["ckpt_name"] == DEV_FP8
    assert node["inputs"]["text_encoder"] == "gemma_3_12B_it_fp4_mixed.safetensors"
    ckpts = _ckpt_names(workflow)
    assert ckpts
    assert set(ckpts) == {DEV_FP8}
    assert TRUNCATED_DISTILLED not in ckpts
    assert BAKE_10EROS not in ckpts
    assert DISTILLED_LORA in _lora_names(workflow)
    assert BAKE_10EROS not in _lora_names(workflow)


def _teacache_object_info() -> dict:
    return {
        "TeaCache": {
            "input": {
                "required": {
                    "model": ["MODEL", {}],
                    "model_type": [["ltxv", "flux", "hidream"], {}],
                    "rel_l1_thresh": ["FLOAT", {"default": 0.3}],
                    "start_percent": ["FLOAT", {"default": 0.0}],
                    "end_percent": ["FLOAT", {"default": 1.0}],
                    "cache_device": [["cuda", "cpu"], {}],
                }
            },
            "output": ["MODEL"],
        }
    }


def _patched_render(variant: str, **kwargs):
    """Patch a render variant. Missing air_render JSON uses the v08 graph."""
    template = load_workflow_template("ltx23_lipsync_v08")
    try:
        own = load_workflow_template(variant)
    except FileNotFoundError:
        own = None

    def _load(requested: str):
        if requested == variant and own is not None:
            return copy.deepcopy(own)
        if requested == variant:
            return copy.deepcopy(template)
        return load_workflow_template(requested)

    with (
        patch(
            "master_agent.comfy.workflow_patcher.resolve_model_path",
            side_effect=_resolve_on_tower,
        ),
        patch(
            "master_agent.comfy.workflow_patcher.load_workflow_template",
            side_effect=_load,
        ),
    ):
        return load_and_patch_workflow(variant, **kwargs)


@pytest.mark.parametrize("variant", RENDER_VARIANTS)
def test_render_variants_keep_authored_sampler_prompt_link_and_skip_teacache(variant: str):
    raw = load_workflow_template("ltx23_lipsync_v08")
    authored = raw["6203"]["inputs"]["value"]
    assert authored.startswith("A photorealistic cinematic shot")
    assert raw["6202:5834"]["inputs"]["text"] == ["6203", 0]
    assert raw["6202:5831"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"

    caller = "a gator in a swamp, speaking to camera"
    workflow, meta = _patched_render(
        variant,
        prompt=caller,
        seed=1,
        duration_s=1.0,
        object_info=_teacache_object_info(),
    )

    assert meta["checkpoint"] == DEV_FP8
    teas = [
        node
        for node in workflow.values()
        if isinstance(node, dict) and node.get("class_type") == "TeaCache"
    ]
    assert teas == []
    assert workflow["6202:5831"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"
    assert workflow["6202:5834"]["inputs"]["text"] == ["6203", 0]
    assert workflow["6203"]["inputs"]["value"] == caller
    assert authored not in workflow["6203"]["inputs"]["value"]


def test_render_variant_sampler_changes_only_when_config_asks():
    workflow, _meta = _patched_render(
        "ltx23_lipsync_v08",
        prompt="clay figure speaks",
        seed=1,
        duration_s=1.0,
        sampler_name="euler",
    )
    assert workflow["6202:5831"]["inputs"]["sampler_name"] == "euler"
    assert workflow["6202:5834"]["inputs"]["text"] == ["6203", 0]


def test_prepare_run_generate_does_not_insert_teacache_on_lipsync_v08():
    info = _teacache_object_info()
    with patch(
        "master_agent.comfy.workflow_patcher.resolve_model_path",
        side_effect=_resolve_on_tower,
    ):
        wf = prepare_run(
            "generate",
            variant="ltx23_lipsync_v08",
            prompt="clay figure speaks",
            object_info=info,
            duration_s=1.0,
            seed=1,
        )
    assert all(
        not (isinstance(node, dict) and node.get("class_type") == "TeaCache")
        for node in wf.values()
    )
    assert wf["6202:5831"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"
    assert wf["6202:5834"]["inputs"]["text"] == ["6203", 0]
