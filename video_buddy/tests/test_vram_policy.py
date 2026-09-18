"""Shared 16GB-class pack policy. No GPU. No invented Hub filenames.

Run: python -m pytest tests/test_vram_policy.py -q
"""

from __future__ import annotations

from pathlib import Path

import yaml

from master_agent.config import (
    H3_DEFAULT_CFG,
    H3_DEFAULT_STEPS,
    H3_MAX_DURATION_S,
    H3_MAX_MP,
    WORKFLOWS_DIR,
    load_manifest_workflow_files,
)
from master_agent.control.cost import estimate_cost
from master_agent.models.vram_policy import (
    ATTESTED_FILENAMES,
    HEAVY_SLUGS,
    NVFP4_MIN_VRAM_GB,
    TARGET_GPU,
    TARGET_VRAM_GB,
    family_for_slug,
    format_vram_table,
    pack_kind,
    preference_order,
    prepare_warning,
    safer_alternate,
    vram_class,
    workflow_row,
    workflow_vram_rows,
)
from master_agent.models.weights import (
    H3_FL2VA_PREFERENCE,
    TRANSFORMER_PREFERENCE,
    bundle_for_variant,
    h3_transformer_preference_order,
    transformer_preference_order,
)
from master_agent.setup import check_vram_policy, snapshot


def _manifest_slugs() -> list[str]:
    return list(load_manifest_workflow_files())


def test_target_is_rtx_5060_ti_16gb():
    assert TARGET_VRAM_GB == 16.0
    assert "5060" in TARGET_GPU
    assert NVFP4_MIN_VRAM_GB == 14.0


def test_every_manifest_slug_has_a_policy_row():
    slugs = _manifest_slugs()
    assert slugs
    rows = {row.slug: row for row in workflow_vram_rows()}
    missing = [s for s in slugs if s not in rows]
    assert not missing, f"16GB policy missing manifest slugs: {missing}"
    for slug in slugs:
        row = workflow_row(slug)
        assert row.family
        assert row.default_pack
        assert row.vram_class in {"safe", "tight", "heavy", "external", "cpu"}
        assert row.expected_vram_gb > 0 or row.vram_class == "external"


def test_preference_ladder_is_gguf_then_nvfp4_then_heavier():
    names = (
        "model-bf16.safetensors",
        "model_fp8_scaled.safetensors",
        "model_nvfp4.safetensors",
        "model-Q5_0.gguf",
        "model-Q4_K_M.gguf",
        "model_int8_convrot.safetensors",
    )
    order16 = preference_order(names, vram_gb=16)
    assert order16[0].endswith("Q4_K_M.gguf")
    assert order16[1].endswith("Q5_0.gguf")
    assert "nvfp4" in order16[2]
    assert "int8" in order16[3]
    order12 = preference_order(names, vram_gb=12)
    assert order12[0].endswith(".gguf")
    assert not any("nvfp4" in n.lower() for n in order12[:2])


def test_pack_kind_ranks_quantization():
    assert pack_kind("foo-Q4_K.gguf") == "gguf_q4"
    assert pack_kind("foo-Q5_0.gguf") == "gguf_q5"
    assert pack_kind("foo_nvfp4.safetensors") == "nvfp4"
    assert pack_kind("foo_fp8_scaled.safetensors") == "fp8"
    assert pack_kind("foo-bf16.safetensors") == "bf16"


def test_ltx25_and_h3_still_prefer_gguf_q4():
    assert transformer_preference_order(vram_gb=16)[0] == TRANSFORMER_PREFERENCE[0]
    assert transformer_preference_order(vram_gb=16)[0].endswith(".gguf")
    assert transformer_preference_order(vram_gb=16)[1].endswith("nvfp4.safetensors")
    h3 = h3_transformer_preference_order("h3_fl2va", vram_gb=16)
    assert h3[0] == H3_FL2VA_PREFERENCE[0]
    assert h3[0].endswith("Q4_K.gguf")
    assert H3_DEFAULT_CFG == 1.0
    assert H3_DEFAULT_STEPS == 4
    assert H3_MAX_DURATION_S == 12.0
    assert H3_MAX_MP == 0.8


def test_h3_and_ltx25_rows_are_16gb_safe():
    for slug in ("ltx25_t2v_i2v", "h3_t2v", "h3_i2v", "h3_flf", "h3_r2v", "base", "wan22"):
        row = workflow_row(slug)
        assert row.vram_class in {"safe", "tight"}
        assert row.expected_vram_gb <= 14.5
        assert "bf16 dual" not in row.default_pack.lower()


def test_wan_vace_krea_flux_qwen_use_attested_16gb_packs():
    wan = workflow_row("wan22")
    assert "lightx2v" in wan.notes.lower() or "lightx2v" in wan.default_pack.lower()
    assert family_for_slug("wan22") == "wan22"
    vace = workflow_row("vb_aivfx_adv_13")
    assert "Q4_K_M.gguf" in vace.default_pack
    krea = workflow_row("krea2_img")
    assert "nvfp4" in krea.default_pack.lower()
    flux = workflow_row("flux")
    assert pack_kind(flux.default_pack) in {"gguf_q4", "fp8"}
    qwen = workflow_row("vb_qwen_edit_360")
    assert qwen.default_pack.endswith(".gguf")
    for name in (
        wan.default_pack,
        vace.default_pack,
        krea.default_pack,
        flux.default_pack,
        qwen.default_pack,
    ):
        assert Path(name).name in ATTESTED_FILENAMES or name in ATTESTED_FILENAMES


def test_heavy_graphs_are_labeled_and_have_safer_alternate():
    assert HEAVY_SLUGS
    for slug in HEAVY_SLUGS:
        row = workflow_row(slug)
        assert row.vram_class == "heavy", slug
        alt = safer_alternate(slug)
        assert alt, slug
        assert alt != slug
        warn = prepare_warning(slug)
        assert warn
        assert "16GB" in warn or "VRAM" in warn or "heavy" in warn.lower()
        assert vram_class(alt) in {"safe", "tight", "external", "cpu"}


def test_k3nk_is_not_a_default_and_has_no_invented_aio_file():
    rows = workflow_vram_rows()
    for row in rows:
        assert "k3nk" not in row.default_pack.lower()
    assert bundle_for_variant("k3nk") is None
    invented = [n for n in ATTESTED_FILENAMES if "k3nk" in n.lower()]
    assert invented == []


def test_cost_gate_inherits_shared_policy():
    base = estimate_cost("base", frames=9)
    movie = estimate_cost("vb_movie_builder", frames=9)
    ccc = estimate_cost("vb_ccc_adv", frames=9)
    assert movie["vram_gb"] > base["vram_gb"]
    assert ccc["vram_gb"] > base["vram_gb"]
    assert movie["flagged"] is True or ccc["flagged"] is True
    h3 = estimate_cost("h3_t2v", frames=124, threshold_vram_gb=15.0)
    assert h3["vram_gb"] <= 14.5
    assert h3["flagged"] is False


def test_doctor_reports_family_picks():
    row = check_vram_policy()
    assert row["name"] == "vram-policy"
    assert "16GB" in row["detail"] or "GGUF" in row["detail"]
    names = {r["name"] for r in snapshot()}
    assert "vram-policy" in names
    assert "ltx25-weights" in names
    assert "h3-weights" in names


def test_format_vram_table_covers_families():
    table = format_vram_table()
    for family in (
        "LTX 2.3",
        "LTX 2.5",
        "MiniMax H3",
        "Wan 2.2",
        "AI-VFX / VACE",
        "Movie Builder",
        "CCC",
        "Flux / Krea",
    ):
        assert family in table, family
    assert "expected VRAM" in table.lower() or "VRAM" in table


def test_manifest_yaml_carries_vram_metadata():
    path = Path(WORKFLOWS_DIR) / "manifests.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    heavy = data["vb_movie_builder"]
    assert heavy.get("vram_class") == "heavy"
    assert heavy.get("safer_alternate")
    safe = data["ltx25_t2v_i2v"]
    assert safe.get("vram_class") in {"safe", "tight"}
    wan = data["wan22"]
    assert wan.get("vram_class") in {"safe", "tight"}


def test_preference_order_matches_ltx25_and_h3_helpers():
    assert preference_order(TRANSFORMER_PREFERENCE, vram_gb=16)[0].endswith(".gguf")
    assert preference_order(H3_FL2VA_PREFERENCE, vram_gb=16)[0].endswith("Q4_K.gguf")


def test_wan_and_flux_download_targets_are_attested_gguf():
    from master_agent.models.weights import WEIGHT_FILES

    wan_high = WEIGHT_FILES["wan22_high"]
    wan_low = WEIGHT_FILES["wan22_low"]
    flux = WEIGHT_FILES["flux"]
    assert wan_high.filename.endswith("Q4_K_S.gguf")
    assert wan_high.repo_id == "QuantStack/Wan2.2-T2V-A14B-GGUF"
    assert wan_high.repo_filename == "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf"
    assert wan_low.repo_filename == "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_S.gguf"
    assert flux.filename == "flux1-dev-Q4_K_S.gguf"
    assert flux.repo_id == "city96/FLUX.1-dev-gguf"
    assert pack_kind(wan_high.filename) == "gguf_q4"
    assert pack_kind(flux.filename) == "gguf_q4"
    # fp8 remains an accepted local fallback, not the download default
    assert any("fp8" in n for n in wan_high.candidates)
    assert any("fp8" in n for n in flux.candidates)
    light = WEIGHT_FILES["wan22_lightx2v"]
    assert "lightx2v" in light.filename.lower()
    assert light.mandatory is False
