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
    refs = _find_nodes_by_class(raw_r2v, "MiniMaxH3ReferenceToVideo")
    assert refs
    raw_inputs = refs[0][1]["inputs"]
    assert "ref_images" not in raw_inputs
    assert raw_inputs["ref_images.ref_image_0"] == ["13", 0]


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
    for bare in ("ref_images", "ref_audios", "ref_videos", "ref_video_audios"):
        assert bare not in inputs
    assert inputs["ref_images.ref_image_0"][1] == 0
    assert inputs["ref_audios.ref_audio_0"][1] == 0
    image_id = str(inputs["ref_images.ref_image_0"][0])
    assert wf[image_id]["class_type"] == "LoadImage"
    assert wf[image_id]["inputs"]["image"] == "hero.png"
    audio_id = str(inputs["ref_audios.ref_audio_0"][0])
    assert wf[audio_id]["class_type"] == "LoadAudio"
    assert inputs["length"] == meta["frames"] == snap(meta["frames"])
    assert "<audio_1>" in inputs["prompt"]
    assert "slow push into a neon alley" in inputs["prompt"]


def _h3_autogrow(slot_type: str, prefix: str) -> list:
    return [
        "COMFY_AUTOGROW_V3",
        {
            "template": {
                "input": {"required": {prefix.rstrip("_"): [slot_type, {}]}},
                "prefix": prefix,
                "min": 0,
                "max": 9,
            }
        },
    ]


def _object_info_for_graph(workflow: dict) -> dict:
    """Permissive specs for every class, with a live-shaped H3 ref node."""
    info: dict = {}
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not class_type or class_type in info:
            continue
        required = {}
        for key, value in (node.get("inputs") or {}).items():
            if "." in key:
                continue
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                required[key] = ["*", {}]
            elif isinstance(value, bool):
                required[key] = ["BOOLEAN", {}]
            elif isinstance(value, int):
                required[key] = ["INT", {}]
            elif isinstance(value, float):
                required[key] = ["FLOAT", {}]
            else:
                required[key] = ["STRING", {}]
        info[class_type] = {
            "input": {"required": required},
            "output": ["*"] * 8,
        }
    if "LoadImage" in info:
        info["LoadImage"]["output"] = ["IMAGE"]
    if "LoadAudio" in info:
        info["LoadAudio"]["output"] = ["AUDIO"]
    info["MiniMaxH3ReferenceToVideo"] = {
        "input": {
            "required": {
                "clip": ["CLIP", {}],
                "vae": ["VAE", {}],
                "audio_vae": ["VAE", {}],
                "prompt": ["STRING", {}],
                "width": ["INT", {}],
                "height": ["INT", {}],
                "length": ["INT", {}],
                "ref_image_size": ["COMBO", {"options": ["match", "max"]}],
            },
            "optional": {
                "ref_images": _h3_autogrow("IMAGE", "ref_image_"),
                "ref_audios": _h3_autogrow("AUDIO", "ref_audio_"),
                "ref_videos": _h3_autogrow("IMAGE", "ref_video_"),
                "ref_video_audios": _h3_autogrow("AUDIO", "ref_video_audio_"),
            },
        },
        "output": ["CONDITIONING", "LATENT"],
    }
    return info


def test_h3_r2v_autogrow_validates_dotted_slots_only():
    """Bare group keys are a COMFY_AUTOGROW_V3 mismatch; dotted slots pass."""
    import copy

    from master_agent.comfy.validator import validate_workflow

    wf, _meta = load_and_patch_workflow(
        "h3_r2v",
        prompt="slow push into a neon alley",
        seed=1,
        duration_s=5.0,
        image_name="hero.png",
        audio_name="line.wav",
    )
    info = _object_info_for_graph(wf)
    report = validate_workflow(wf, info, file_label="h3_r2v", object_info_source="fixture")
    assert report.ok, report.errors

    bare = copy.deepcopy(wf)
    for node in bare.values():
        if isinstance(node, dict) and node.get("class_type") == "MiniMaxH3ReferenceToVideo":
            node["inputs"]["ref_images"] = node["inputs"]["ref_images.ref_image_0"]
            node["inputs"]["ref_audios"] = node["inputs"]["ref_audios.ref_audio_0"]
    bad = validate_workflow(bare, info, file_label="h3_r2v-bare", object_info_source="fixture")
    assert any("COMFY_AUTOGROW_V3" in err.message for err in bad.errors)
    assert any(err.input_name == "ref_images" for err in bad.errors)
    assert any(err.input_name == "ref_audios" for err in bad.errors)

    combo = copy.deepcopy(wf)
    for node in combo.values():
        if isinstance(node, dict) and node.get("class_type") == "MiniMaxH3ReferenceToVideo":
            node["inputs"]["ref_image_size"] = "nope"
    combo_report = validate_workflow(
        combo, info, file_label="h3_r2v-combo", object_info_source="fixture"
    )
    assert any("not in combo choices" in err.message for err in combo_report.errors)


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


def test_h3_named_photo_voice_warns_and_stays_on_h3_r2v():
    from master_agent.orchestrator.talking import (
        H3_R2V_AUDIO_LABEL,
        h3_r2v_audio_warning,
        preview_media_variant,
    )

    brief = "hailuo, she says the line"
    assert preview_media_variant(brief, has_image=True, has_audio=True) == "h3_r2v"
    assert (
        h3_r2v_audio_warning(brief, has_image=True, has_audio=True) == H3_R2V_AUDIO_LABEL
    )
    assert (
        h3_r2v_audio_warning(
            "she says the line", variant="h3_r2v", has_image=True, has_audio=True
        )
        == H3_R2V_AUDIO_LABEL
    )
    assert (
        h3_r2v_audio_warning(
            "minimax ref2va close-up", variant="ref2va", has_image=True, has_audio=True
        )
        == H3_R2V_AUDIO_LABEL
    )
    assert preview_media_variant(
        "she says the line", has_image=True, has_audio=True
    ) == "ltx25_a2v"
    assert h3_r2v_audio_warning("she says the line", has_image=True, has_audio=True) is None
    assert (
        h3_r2v_audio_warning(
            brief, variant="ltx25_a2v", has_image=True, has_audio=True
        )
        is None
    )
    assert h3_r2v_audio_warning(brief, has_image=True, has_audio=True, has_video=True) is None


def test_h3_r2v_voice_reference_label_is_on_picker_surfaces():
    from master_agent.comfy.catalog import load_catalog
    from master_agent.orchestrator.talking import H3_R2V_AUDIO_LABEL

    entry = next(item for item in load_catalog() if item.id == "h3_r2v")
    assert H3_R2V_AUDIO_LABEL in entry.description
    root = Path(__file__).resolve().parents[1]
    repo = root.parent
    manifest = (root / "workflows" / "manifests.yaml").read_text(encoding="utf-8")
    assert H3_R2V_AUDIO_LABEL in manifest
    for rel in (
        root / "README.md",
        root / "workflows" / "minimax-h3" / "README.md",
        root / "docs" / "QUICKSTART.md",
        root / "skills" / "video-buddy" / "TOOLS.md",
        root / "master_agent" / "orchestrator" / "prompts" / "director.md",
        root / "master_agent" / "web" / "static" / "index.html",
        root / "AGENTS.md",
        repo / "README.md",
    ):
        text = rel.read_text(encoding="utf-8")
        assert H3_R2V_AUDIO_LABEL in text, rel
