"""Capability gap matrix + template slug wiring. No GPU.

Run: python -m pytest tests/test_capabilities.py tests/test_comfy_cli.py -q
"""

from __future__ import annotations

from pathlib import Path

from master_agent.comfy.capabilities import (
    CAPABILITY_CATALOG,
    TOWER_LIVE_YES,
    format_matrix,
    probe_capabilities,
)
from master_agent.comfy.cli_run import list_templates, resolve_template
from master_agent.config import WORKFLOW_FILES, WORKFLOWS_DIR


def test_wan22_workflow_file_exists():
    rel = WORKFLOW_FILES["wan22"]
    assert "MICKMUMPITZ" not in rel
    assert (WORKFLOWS_DIR / rel).is_file()


def test_resolve_template_wan22_and_manifest_slug():
    wan = resolve_template("wan22")
    assert wan.name == "260713_VIDEO-BUDDY_WAN-2-2-VID_1-0_api.json"
    assert wan.is_file()
    aivfx = resolve_template("vb_aivfx_adv")
    assert aivfx.name == "260330_VIDEO-BUDDY_AI-VFX_1-0_ADV_api.json"
    assert aivfx.is_file()
    flux = resolve_template("flux")
    assert flux.name == "flux_t2i.json"


def test_list_templates_includes_director_and_manifest():
    items = list_templates()
    by_id = {i["id"]: i for i in items}
    assert by_id["wan22"]["kind"] == "variant"
    assert by_id["vb_aivfx_adv"]["kind"] == "manifest"
    assert Path(WORKFLOWS_DIR / by_id["wan22"]["path"]).is_file()


def test_probe_marks_fun_inpaint_present_but_unwired():
    info = {"WanFunInpaintToVideo": {}, "EmptyLTXVLatentVideo": {}}
    rows = {r.capability.id: r for r in probe_capabilities(info)}
    fun = rows["wan_fun_inpaint"]
    assert fun.in_object_info == ["WanFunInpaintToVideo"]
    assert fun.verdict == "no"
    assert "no graph" in fun.gap.lower() or "no graph" in fun.capability.notes.lower()
    ltx = rows["ltx_t2v"]
    assert ltx.verdict == "yes"
    assert "director" in ltx.capability.surfaces


def test_probe_does_not_claim_missing_lanpaint():
    rows = {r.capability.id: r for r in probe_capabilities({})}
    assert rows["lanpaint"].verdict == "no"
    assert rows["lanpaint"].in_object_info == []


def test_k3nk_does_not_match_aio_preprocessor():
    rows = {
        r.capability.id: r
        for r in probe_capabilities({"AIO_Preprocessor": {}, "RecraftTextToImageNode": {}})
    }
    assert rows["k3nk_wan_aio"].in_object_info == []
    assert rows["raft"].in_object_info == []


def test_teacache_is_bypass_not_inject():
    tea = next(c for c in CAPABILITY_CATALOG if c.id == "wan_teacache")
    assert "bypass" in tea.surfaces
    assert "inject" not in tea.surfaces
    assert "CACHEARGS" in tea.notes
    assert "TeaCache" in tea.class_types


def test_catalog_uses_live_tower_exact_names():
    by_id = {c.id: c for c in CAPABILITY_CATALOG}
    assert by_id["lanpaint"].class_types == ("LanPaint_KSampler",)
    assert by_id["video_noise_warp"].class_types == ("GetWarpedNoiseFromVideo",)
    catalog_names = {ct for c in CAPABILITY_CATALOG for ct in c.class_types}
    for name in TOWER_LIVE_YES:
        assert name in catalog_names, name


def test_probe_hits_lanpaint_ksampler_and_warp():
    info = {"LanPaint_KSampler": {}, "GetWarpedNoiseFromVideo": {}}
    rows = {r.capability.id: r for r in probe_capabilities(info)}
    assert rows["lanpaint"].in_object_info == ["LanPaint_KSampler"]
    assert rows["video_noise_warp"].in_object_info == ["GetWarpedNoiseFromVideo"]
    assert rows["lanpaint"].verdict == "partial"


def test_format_matrix_has_header():
    rows = probe_capabilities({"EmptyLTXVLatentVideo": {}})
    text = format_matrix(rows, source="test")
    assert "Capability" in text
    assert "Director allowlist" in text
    assert "wan22" in text
