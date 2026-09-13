"""16GB (RTX 5060 Ti) policy: catalog, doctor, patcher defaults, baked graphs.

Run: python -m pytest tests/test_vram_policy.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

from master_agent.comfy.catalog import list_catalog_items
from master_agent.comfy.cli_run import unwrap_workflow
from master_agent.comfy.workflow_patcher import (
    _find_nodes_by_class,
    load_and_patch_workflow,
    load_workflow_template,
)
from master_agent.config import QUALITY_PROFILES, WORKFLOWS_DIR
from master_agent.models.vram_policy import (
    FAMILY_PRESETS,
    VRAM_CLASS_MORE,
    VRAM_CLASS_OFFLOAD,
    VRAM_CLASS_SAFE,
    apply_16gb_size,
    family_for_variant,
    format_policy_table,
    policy_rows,
    preset_for_variant,
    vram_warnings,
)
from master_agent.setup import check_vace_weights, check_vram_policy, snapshot


def test_quality_profiles_include_16gb():
    assert "16gb" in QUALITY_PROFILES
    assert QUALITY_PROFILES["16gb"]["max_width"] == 768
    assert QUALITY_PROFILES["16gb"]["segment_max_s"] == 3.0


def test_policy_table_covers_director_families():
    rows = {r["variant"]: r for r in policy_rows()}
    families = {r["family"] for r in policy_rows()}
    for family in ("ltx25", "h3", "ltx23", "wan", "vace", "image", "movie", "ccc"):
        assert family in families, family
    assert rows["ltx25_t2v_i2v"]["vram_class"] == VRAM_CLASS_SAFE
    assert "GGUF" in rows["ltx25_t2v_i2v"]["default_pack"]
    assert rows["ltx25_t2v_i2v_two_stage"]["vram_class"] == VRAM_CLASS_OFFLOAD
    assert rows["h3_t2v"]["vram_class"] == VRAM_CLASS_SAFE
    assert rows["wan22"]["vram_class"] == VRAM_CLASS_OFFLOAD
    assert rows["vb_movie_builder"]["vram_class"] == VRAM_CLASS_MORE
    assert rows["vb_ccc_adv"]["vram_class"] == VRAM_CLASS_MORE
    text = format_policy_table()
    assert "GGUF first" in text
    assert "wan22" in text


def test_aliases_inherit_family_policy():
    assert family_for_variant("t2v_i2v") == "ltx25"
    assert family_for_variant("fl2va") == "h3"
    assert family_for_variant("vb_wan22_vid") == "wan"
    assert preset_for_variant("ltx25_t2v_i2v_two_stage").vram_class == VRAM_CLASS_OFFLOAD
    assert preset_for_variant("k3nk_wan_aio").vram_class == VRAM_CLASS_MORE


def test_apply_16gb_size_remaps_wan_and_krea_defaults():
    assert apply_16gb_size("wan22", 768, 512) == (640, 384)
    assert apply_16gb_size("krea2_img", 768, 512) == (1024, 576)
    # explicit sizes stay
    assert apply_16gb_size("wan22", 1280, 720) == (1280, 720)
    assert apply_16gb_size("ltx25_t2v_i2v", 768, 512) == (768, 512)


def test_patcher_wan_defaults_are_16gb_hull():
    wf, meta = load_and_patch_workflow(
        "wan22",
        prompt="photoreal street",
        duration_s=5.0,
        seed=1,
    )
    assert meta["width"] == 640
    assert meta["height"] == 384
    assert meta["frames"] <= 49
    assert meta["steps"] == 8
    assert meta["cfg"] == 1.0
    assert wf["5"]["inputs"]["width"] == 640
    assert wf["5"]["inputs"]["length"] == meta["frames"]


def test_patcher_ltx25_caps_duration_on_16gb():
    _wf, meta = load_and_patch_workflow(
        "ltx25_t2v_i2v",
        prompt="neon rain",
        duration_s=8.0,
        seed=1,
    )
    assert meta["width"] == 768
    assert meta["height"] == 512
    assert meta["frames"] <= 73  # 3s @ 24fps, 8n+1
    assert meta["cfg"] == 1.0


def test_patcher_h3_still_16gb():
    _wf, meta = load_and_patch_workflow(
        "h3_t2v",
        prompt="stereo city",
        duration_s=5.0,
        seed=1,
    )
    assert meta["width"] == 1152
    assert meta["height"] == 640
    assert meta["steps"] == 4
    assert meta["cfg"] == 1.0
    assert meta["frames"] == 124


def test_baked_ltx25_graphs_are_768x512():
    for rel in (
        "ltx-2.5/LTX-2.5_T2V_I2V_Single_Stage_Distilled_api.json",
        "ltx-2.5/LTX-2.5_FLF2V_api.json",
    ):
        data = json.loads((WORKFLOWS_DIR / rel).read_text(encoding="utf-8"))
        wf = unwrap_workflow(data)
        latents = _find_nodes_by_class(wf, "EmptyLTXVLatentVideo")
        assert latents, rel
        inputs = latents[0][1]["inputs"]
        assert inputs["width"] == 768
        assert inputs["height"] == 512
        assert inputs["length"] <= 25


def test_baked_wan_graph_is_640x384():
    wf = load_workflow_template("wan22")
    latent = wf["5"]["inputs"]
    assert latent["width"] == 640
    assert latent["height"] == 384
    assert latent["length"] == 33


def test_vace_api_defaults_to_gguf_loader():
    for slug in ("vb_aivfx_adv", "vb_aivfx_adv_13"):
        wf = load_workflow_template(slug)
        ggufs = _find_nodes_by_class(wf, "UnetLoaderGGUF")
        assert ggufs, slug
        name = ggufs[0][1]["inputs"]["unet_name"]
        assert name.endswith("Q4_K_M.gguf")


def test_krea_and_ccc41_default_nvfp4():
    krea = load_workflow_template("krea2_img")
    unets = _find_nodes_by_class(krea, "UNETLoader")
    assert unets[0][1]["inputs"]["unet_name"] == "krea2_turbo_nvfp4.safetensors"
    assert krea["6"]["inputs"]["width"] == 1024
    assert krea["6"]["inputs"]["height"] == 576
    ccc = load_workflow_template("vb_ccc41_krea2")
    names = [
        n["inputs"].get("unet_name")
        for _nid, n in _find_nodes_by_class(ccc, "UNETLoader")
    ]
    assert "krea2_turbo_nvfp4.safetensors" in names
    assert "krea2_turbo_fp8_scaled.safetensors" not in names


def test_catalog_exposes_vram_fields():
    items = {i["id"]: i for i in list_catalog_items() if i.get("kind") == "variant"}
    assert items["ltx25_t2v_i2v"]["vram_class"] == VRAM_CLASS_SAFE
    assert "GGUF" in items["ltx25_t2v_i2v"]["default_pack"]
    assert items["wan22"]["vram_class"] == VRAM_CLASS_OFFLOAD
    assert items["vb_movie_builder"]["vram_class"] == VRAM_CLASS_MORE
    assert items["h3_t2v"]["vram_class"] == VRAM_CLASS_SAFE


def test_prepare_warnings_for_heavy_graphs():
    more = vram_warnings("vb_movie_builder")
    assert more
    assert "needs_more_vram" in more[0]
    off = vram_warnings("wan22")
    assert off
    assert "offload" in off[0]
    safe = vram_warnings("h3_t2v")
    assert safe == []


def test_doctor_rows_include_vram_policy():
    row = check_vram_policy()
    assert row["ok"] is True
    assert "GGUF" in row["detail"]
    vace = check_vace_weights()
    assert vace["ok"] is True
    names = [r["name"] for r in snapshot()]
    assert "vram-policy" in names
    assert "vace-weights" in names
    assert "ltx25-weights" in names
    assert "h3-weights" in names


def test_no_invented_wan_t2v_gguf():
    """Wan T2V pack stays the documented fp8 dual UNET — no invented GGUF name."""
    from master_agent.config import MODEL_FILES

    wan = MODEL_FILES["wan22"]
    assert "fp8_scaled" in wan["checkpoint_high"]
    assert not wan["checkpoint_high"].lower().endswith(".gguf")
