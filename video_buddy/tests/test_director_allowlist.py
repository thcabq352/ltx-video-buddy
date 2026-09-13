"""Director allowlist is every manifests.yaml slug (Scott override).

Run: python -m pytest tests/test_director_allowlist.py -q
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from master_agent.config import WORKFLOW_FILES, WORKFLOWS_DIR, load_workflow_files
from master_agent.orchestrator.director import (
    _allowed_variants,
    choose_variant,
    rule_based_variant,
)


def _manifest_slugs() -> list[str]:
    path = Path(WORKFLOWS_DIR) / "manifests.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        str(slug)
        for slug, meta in data.items()
        if isinstance(meta, dict) and isinstance(meta.get("file"), str) and meta["file"].strip()
    ]


def test_every_manifest_slug_is_in_director_allowlist():
    slugs = _manifest_slugs()
    assert slugs, "manifests.yaml must list at least one workflow"
    allowlist = set(WORKFLOW_FILES)
    missing = [s for s in slugs if s not in allowlist]
    assert not missing, f"director allowlist missing manifest slugs: {missing}"
    derived = load_workflow_files()
    for slug in slugs:
        assert slug in derived
        assert derived[slug] == WORKFLOW_FILES[slug]


def test_workflow_files_keeps_legacy_variant_keys():
    for key in ("base", "eros", "directors", "lipsync", "wan22", "flux"):
        assert key in WORKFLOW_FILES
        assert WORKFLOW_FILES[key].endswith(".json")


def test_allowed_variants_include_every_manifest_slug():
    allowed = _allowed_variants()
    missing = [s for s in _manifest_slugs() if s not in allowed]
    assert not missing, f"LLM allowlist missing: {missing}"


def test_keyword_routing_hits_new_families():
    cases = {
        "possession vfx composite with vace": "vb_aivfx_adv",
        "run the aivfx 1.3 compositor": "vb_aivfx_adv_13",
        "aivfx preprocess sam3 depthcrafter": "vb_aivfx_preprocess",
        "generate an aivfx start-image": "vb_aivfx_startimage",
        "movie builder shot-by-shot feature": "vb_movie_builder",
        "consistent character creator ccc 4.01": "vb_ccc_adv",
        "ccc 4.1 krea2-edit grounded character": "vb_ccc41_krea2",
        "dataset tagger for lora captions": "vb_dataset_tagger",
        "tag review the auto captions": "vb_tag_review",
        "ideogram prompt builder still": "vb_ideogram",
        "qwen-image-edit 360 turnaround": "vb_qwen_edit_360",
        "krea-2 image generation": "krea2_img",
        "flux character sheet": "flux",
        "ai-renderer advanced pass": "vb_ai_renderer_adv",
        "smpl renderer simple": "vb_ai_renderer_smpl",
        "nvidia rtx super resolution": "vb_rtx_superres",
        "bear minimum bar clay scene": "air_render_050",
        "minimax h3 native stereo": "h3_t2v",
        "ltx 2.5 alley push-in": "ltx25_t2v_i2v",
        "wan 2.2 photoreal film grain": "wan22",
    }
    for request, expected in cases.items():
        assert rule_based_variant(request) == expected, request


def test_hard_constraints_win_over_soft_keywords():
    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False):
        assert choose_variant("possession vfx composite", force="base") == ("base", "forced")
        assert choose_variant("movie builder shot-by-shot", has_video=True) == (
            "lipsync",
            "input",
        )
        assert choose_variant("a dub of my clip") == ("lipsync", "rules")
        assert choose_variant("rain on a window") == ("base", "rules")
        assert choose_variant("10eros teaser") == ("eros", "rules")


def test_existing_keyword_routes_still_work():
    assert rule_based_variant("a dub of my clip") == "lipsync"
    assert rule_based_variant("10eros teaser") == "eros"
    assert rule_based_variant("cinematic brand film with three scenes") == "directors"
    assert rule_based_variant("rain on a window") == "base"


def test_aivfx_and_movie_builder_field_maps_apply_prompt():
    from master_agent.comfy.workflow_patcher import _find_nodes_by_class, load_and_patch_workflow

    wf, meta = load_and_patch_workflow("vb_aivfx_adv", prompt="UNIQUE_AIVFX_PROMPT_XYZ")
    assert meta["variant"] == "vb_aivfx_adv"
    builders = _find_nodes_by_class(wf, "IterPromptBuilder")
    assert builders
    assert builders[0][1]["inputs"]["string_1"] == "UNIQUE_AIVFX_PROMPT_XYZ"

    movie, movie_meta = load_and_patch_workflow(
        "vb_movie_builder", prompt="UNIQUE_MOVIE_PROMPT_XYZ"
    )
    assert movie_meta["variant"] == "vb_movie_builder"
    assert movie["13383"]["inputs"]["value"] == "UNIQUE_MOVIE_PROMPT_XYZ"
