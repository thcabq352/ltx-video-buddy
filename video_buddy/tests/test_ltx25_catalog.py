"""Default catalog: every shipped LTX 2.5 workflow is listed and patchable.

Run: python -m pytest tests/test_ltx25_catalog.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from master_agent.comfy.catalog import (
    LTX25_FILES,
    RESEARCH_ALIASES,
    clear_catalog_cache,
    default_variant_ids,
    extra_ltx25_workflow_dirs,
    is_known_variant,
    list_catalog_items,
    resolve_variant,
    resolve_workflow_path,
)
from master_agent.comfy.cli_run import list_templates, prepare_run
from master_agent.comfy.workflow_patcher import (
    _find_nodes_by_class,
    load_and_patch_workflow,
    load_workflow_template,
)
from master_agent.config import WORKFLOW_FILES, WORKFLOWS_DIR, snap_ltx_frames


@pytest.fixture(autouse=True)
def _clear_catalog():
    clear_catalog_cache()
    yield
    clear_catalog_cache()


SHIPPED_LTX25 = list(LTX25_FILES.keys())


def test_every_shipped_ltx25_file_exists():
    for rel in LTX25_FILES.values():
        assert (WORKFLOWS_DIR / rel).is_file(), rel


def test_every_shipped_id_is_in_default_catalog():
    ids = default_variant_ids()
    for vid in SHIPPED_LTX25:
        assert vid in ids, f"{vid} missing from default catalog"
        assert is_known_variant(vid)


def test_research_aliases_resolve_without_env_flag():
    for alias, canonical in RESEARCH_ALIASES.items():
        assert is_known_variant(alias)
        assert resolve_variant(alias).id == canonical


def test_existing_i2v_t2v_entrypoints_still_default():
    ids = default_variant_ids()
    for vid in ("base", "eros", "directors", "lipsync", "wan22"):
        assert vid in ids
        assert vid in WORKFLOW_FILES


def test_list_templates_marks_ltx25_as_variants():
    items = list_templates()
    by_id = {i["id"]: i for i in items}
    for vid in SHIPPED_LTX25:
        assert by_id[vid]["kind"] == "variant"
        assert by_id[vid]["path"].startswith("ltx-2.5/")


def test_patcher_accepts_every_shipped_template():
    for vid in SHIPPED_LTX25:
        wf, meta = load_and_patch_workflow(
            vid,
            prompt="slow push into a neon alley",
            negative_prompt="watermark",
            seed=42,
            duration_s=3.0,
            first_image="hero.png",
            last_image="tail.png",
        )
        assert isinstance(wf, dict)
        assert meta["variant"] == vid
        assert snap_ltx_frames(meta["frames"]) == meta["frames"]
        assert meta["frames"] >= 9
        assert (meta["frames"] - 1) % 8 == 0
        texts = [
            node["inputs"].get("text")
            for node in wf.values()
            if isinstance(node, dict) and isinstance(node.get("inputs"), dict)
            and "text" in node["inputs"]
            and not isinstance(node["inputs"]["text"], list)
        ]
        assert "slow push into a neon alley" in texts, vid


def test_local_gguf_rewrites_unet_loader_gguf(tmp_path, monkeypatch):
    root = tmp_path / "models"
    gguf = root / "diffusion_models" / "gguf" / "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf"
    te = root / "text_encoders" / "gemma4-12b-heretic-ltx25-int8convrot.safetensors"
    gguf.parent.mkdir(parents=True)
    te.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    te.write_bytes(b"te")
    monkeypatch.setattr(
        "master_agent.models.weights.model_search_roots",
        lambda: [root],
    )
    wf, _meta = load_and_patch_workflow(
        "ltx25_t2v_i2v",
        prompt="hero shot",
        seed=1,
        duration_s=2.0,
    )
    ggufs = _find_nodes_by_class(wf, "UnetLoaderGGUF")
    assert ggufs
    assert ggufs[0][1]["inputs"]["unet_name"].endswith(".gguf")
    tes = _find_nodes_by_class(wf, "LTXAVTextEncoderLoader")
    assert tes[0][1]["inputs"]["text_encoder"].startswith("gemma4-12b-heretic")


def test_stub_ckpt_remapped_to_official_transformer():
    wf, _meta = load_and_patch_workflow(
        "ltx25_t2v_i2v",
        prompt="hero shot",
        seed=1,
        duration_s=2.0,
    )
    official = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    stub = "ltx-2.5-22b-distilled.safetensors"
    dumped = str(wf)
    assert stub not in dumped
    assert official in dumped
    unets = _find_nodes_by_class(wf, "UNETLoader")
    assert unets, "CheckpointLoaderSimple should rewrite to UNETLoader"
    assert unets[0][1]["inputs"]["unet_name"] == official
    tes = _find_nodes_by_class(wf, "LTXAVTextEncoderLoader")
    assert tes
    assert tes[0][1]["inputs"]["text_encoder"].startswith("gemma4-12b-with-proj-ltx-2.5")


def test_generate_mode_prepares_each_ltx25_id():
    for vid in SHIPPED_LTX25:
        wf = prepare_run("generate", variant=vid, prompt="catalog smoke")
        assert any(isinstance(n, dict) and "class_type" in n for n in wf.values())


def test_resolve_workflow_path_for_alias_and_id():
    a = resolve_workflow_path("t2v_i2v")
    b = resolve_workflow_path("ltx25_t2v_i2v")
    assert a == b
    assert a.is_file()


def test_load_template_does_not_fall_back_to_base_for_ltx25():
    raw = load_workflow_template("ltx25_flf2v")
    assert _find_nodes_by_class(raw, "LTXVImgToVideo")
    assert _find_nodes_by_class(raw, "LoadImage")


def test_resolve_workflow_falls_back_to_ltx_director_sibling(tmp_path, monkeypatch):
    extra = tmp_path / "ltx_director" / "workflows" / "ltx-2.5"
    extra.mkdir(parents=True)
    name = Path(LTX25_FILES["ltx25_flf2v"]).name
    dest = extra / name
    dest.write_text((WORKFLOWS_DIR / LTX25_FILES["ltx25_flf2v"]).read_text(encoding="utf-8"))
    empty = tmp_path / "empty_workflows"
    empty.mkdir()
    monkeypatch.setattr("master_agent.comfy.catalog._workflows_dir", lambda: empty)
    monkeypatch.setattr("master_agent.comfy.catalog.extra_ltx25_workflow_dirs", lambda: [extra])
    path = resolve_workflow_path("ltx25_flf2v")
    assert path == dest


def test_extra_ltx25_workflow_dirs_finds_sibling_tree(tmp_path, monkeypatch):
    extra = tmp_path / "ltx_director" / "workflows" / "ltx-2.5"
    extra.mkdir(parents=True)
    monkeypatch.setattr("master_agent.config.PROJECT_ROOT", tmp_path / "video_buddy")
    dirs = extra_ltx25_workflow_dirs()
    assert extra.resolve() in [d.resolve() for d in dirs]
