"""Default catalog: every shipped MiniMax H3 workflow is listed and patchable.

Run: python -m pytest tests/test_h3_catalog.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from master_agent.comfy.catalog import (
    H3_ALIASES,
    H3_FILES,
    RESEARCH_ALIASES,
    clear_catalog_cache,
    default_variant_ids,
    is_known_variant,
    resolve_variant,
    resolve_workflow_path,
)
from master_agent.comfy.cli_run import list_templates, prepare_run
from master_agent.comfy.workflow_patcher import (
    _find_nodes_by_class,
    load_and_patch_workflow,
    load_workflow_template,
)
from master_agent.config import WORKFLOW_FILES, WORKFLOWS_DIR, snap_h3_frames, snap_ltx_frames


@pytest.fixture(autouse=True)
def _clear_catalog():
    clear_catalog_cache()
    yield
    clear_catalog_cache()


SHIPPED_H3 = list(H3_FILES.keys())


def test_every_shipped_h3_file_exists():
    for rel in H3_FILES.values():
        assert (WORKFLOWS_DIR / rel).is_file(), rel


def test_every_shipped_h3_id_is_in_default_catalog():
    ids = default_variant_ids()
    for vid in SHIPPED_H3:
        assert vid in ids, f"{vid} missing from default catalog"
        assert is_known_variant(vid)
        assert vid in WORKFLOW_FILES


def test_h3_aliases_resolve_without_env_flag():
    for alias, canonical in H3_ALIASES.items():
        assert is_known_variant(alias)
        assert resolve_variant(alias).id == canonical


def test_ltx25_and_wan_entrypoints_still_default():
    ids = default_variant_ids()
    for vid in ("base", "eros", "directors", "lipsync", "wan22", "ltx25_t2v_i2v"):
        assert vid in ids
    for alias, canonical in RESEARCH_ALIASES.items():
        assert resolve_variant(alias).id == canonical


def test_list_templates_marks_h3_as_variants():
    items = list_templates()
    by_id = {i["id"]: i for i in items}
    for vid in SHIPPED_H3:
        assert by_id[vid]["kind"] == "variant"
        assert by_id[vid]["path"].startswith("minimax-h3/")
        assert by_id[vid].get("family") == "h3"


def test_patcher_accepts_every_shipped_h3_template():
    for vid in SHIPPED_H3:
        wf, meta = load_and_patch_workflow(
            vid,
            prompt="slow push into a neon alley",
            negative_prompt="watermark",
            seed=42,
            duration_s=5.0,
            first_image="hero.png",
            last_image="tail.png",
        )
        assert isinstance(wf, dict)
        assert meta["variant"] == vid
        assert snap_h3_frames(meta["frames"]) == meta["frames"]
        assert meta["frames"] >= 5
        assert (meta["frames"] - 5) % 17 == 0
        assert meta["cfg"] == 1.0
        assert meta["steps"] == 4
        texts = [
            node["inputs"].get("prompt")
            for node in wf.values()
            if isinstance(node, dict)
            and isinstance(node.get("inputs"), dict)
            and "prompt" in node["inputs"]
            and not isinstance(node["inputs"]["prompt"], list)
        ]
        assert "slow push into a neon alley" in texts, vid
        for _nid, node in _find_nodes_by_class(wf, "KSampler"):
            assert node["inputs"]["cfg"] == 1.0


def test_h3_cfg_stays_one_even_if_asked_otherwise():
    _wf, meta = load_and_patch_workflow(
        "h3_t2v",
        prompt="hero shot",
        seed=1,
        duration_s=5.0,
        cfg=3.5,
        steps=20,
    )
    assert meta["cfg"] == 1.0
    assert meta["steps"] == 20


def test_local_gguf_rewrites_h3_unet_loader(tmp_path, monkeypatch):
    root = tmp_path / "models"
    gguf = root / "diffusion_models" / "gguf" / "minimax_h3_fl2va_pruned-Q4_K.gguf"
    nvfp4 = root / "diffusion_models" / "minimax_h3_fl2va_pruned_nvfp4.safetensors"
    te = root / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    gguf.parent.mkdir(parents=True)
    te.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    nvfp4.write_bytes(b"nvfp4")
    te.write_bytes(b"te")
    monkeypatch.setattr(
        "master_agent.models.weights.model_search_roots",
        lambda: [root],
    )
    wf, _meta = load_and_patch_workflow(
        "h3_t2v",
        prompt="hero shot",
        seed=1,
        duration_s=5.0,
    )
    ggufs = _find_nodes_by_class(wf, "UnetLoaderGGUF")
    assert ggufs
    assert ggufs[0][1]["inputs"]["unet_name"].endswith(".gguf")
    clips = _find_nodes_by_class(wf, "CLIPLoader")
    assert clips
    assert clips[0][1]["inputs"]["clip_name"].startswith("qwen3vl_32b_minimax_h3")


def test_generate_mode_prepares_each_h3_id():
    for vid in SHIPPED_H3:
        wf = prepare_run("generate", variant=vid, prompt="catalog smoke")
        assert any(isinstance(n, dict) and "class_type" in n for n in wf.values())


def test_resolve_workflow_path_for_alias_and_id():
    a = resolve_workflow_path("fl2va")
    b = resolve_workflow_path("h3_t2v")
    assert a == b
    assert a.is_file()
    assert resolve_workflow_path("ref2va") == resolve_workflow_path("h3_r2v")


def test_load_template_does_not_fall_back_to_base_for_h3():
    raw = load_workflow_template("h3_flf")
    assert _find_nodes_by_class(raw, "MiniMaxH3ImageToVideo")
    assert _find_nodes_by_class(raw, "LoadImage")
    raw_r2v = load_workflow_template("h3_r2v")
    assert _find_nodes_by_class(raw_r2v, "MiniMaxH3ReferenceToVideo")


def test_h3_does_not_use_ltx_frame_law():
    _wf, meta = load_and_patch_workflow(
        "h3_t2v",
        prompt="x",
        seed=1,
        duration_s=5.0,
    )
    assert meta["frames"] != snap_ltx_frames(meta["frames"]) or meta["frames"] == 124
    assert meta["frames"] == 124


def test_h3_r2v_wires_reference_audio():
    from master_agent.config import snap_h3_frames as snap

    wf, meta = load_and_patch_workflow(
        "h3_r2v",
        prompt="slow push into a neon alley",
        seed=1,
        duration_s=5.0,
        image_name="hero.png",
        audio_name="line.wav",
    )
    audios = [
        node["inputs"].get("audio")
        for node in wf.values()
        if isinstance(node, dict) and node.get("class_type") == "LoadAudio"
    ]
    assert "line.wav" in audios
    refs = [
        node
        for node in wf.values()
        if isinstance(node, dict) and node.get("class_type") == "MiniMaxH3ReferenceToVideo"
    ]
    assert refs
    inputs = refs[0]["inputs"]
    assert inputs["ref_audios.ref_audio_0"][1] == 0
    assert inputs["length"] == meta["frames"] == snap(meta["frames"])
    assert "<audio_1>" in inputs["prompt"]
    assert "slow push into a neon alley" in inputs["prompt"]


def test_h3_fl2va_rejects_voice_file():
    from master_agent.orchestrator.talking import media_route_error, preview_media_variant

    err = media_route_error("h3_i2v", has_image=True, has_audio=True)
    assert err
    assert "ltx25_a2v" in err
    assert "h3_r2v" in err
    assert preview_media_variant(
        "lip sync this photo", has_image=True, has_audio=True
    ) == "ltx25_a2v"
    assert preview_media_variant(
        "use hailuo on this photo", has_image=True, has_audio=True
    ) == "h3_r2v"
