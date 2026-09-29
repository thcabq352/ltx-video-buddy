"""LTX 2.3 in/outpaint layout, patch, and offline validation. No GPU.

Run: python -m pytest tests/test_ltx23_inoutpaint.py -q
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from master_agent.comfy.inoutpaint import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MASK_PLACEHOLDER,
    OFFICIAL_NEGATIVE,
    VARIANT,
    default_inpaint_mask_png,
    ensure_inoutpaint_mask,
    fit_source_video,
    fit_working_size,
    finalize_inoutpaint_graph,
    outpaint_layout,
    outpaint_mask_png,
    paint_mask,
    parse_aspect,
    prepare_queue_inputs,
    record_comfy_provenance,
    stash_local_video,
    trim_video_command,
)
from master_agent.comfy.validator import validate_workflow
from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.config import OBJECT_INFO_CACHE, WORKFLOW_FILES, WORKFLOWS_DIR
from master_agent.orchestrator.director import choose_variant, rule_based_variant
from master_agent.provenance import CLIP_PROVENANCE_SCHEMA, missing_required, sidecar_path
from master_agent.setup import snapshot


def _png_size(png: bytes) -> tuple[int, int]:
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    length = struct.unpack(">I", png[8:12])[0]
    assert png[12:16] == b"IHDR"
    width, height = struct.unpack(">II", png[16 : 16 + 8])
    assert length >= 8
    return width, height


def _rgb_at(rgb: bytes, width: int, x: int, y: int) -> tuple[int, int, int]:
    i = (y * width + x) * 3
    return rgb[i], rgb[i + 1], rgb[i + 2]


def test_parse_aspect_and_16_9_to_9_16_layout():
    assert abs(parse_aspect("9:16") - 9 / 16) < 1e-9
    assert abs(parse_aspect("9/16") - 9 / 16) < 1e-9
    plan = outpaint_layout(1280, 720, aspect="9:16")
    assert plan["canvas_w"] >= 1280
    assert plan["canvas_h"] >= 720
    pad = plan["pad"]
    for key in ("left", "top", "right", "bottom"):
        assert pad[key] % 8 == 0
    assert pad["left"] + 1280 + pad["right"] == plan["canvas_w"]
    assert pad["top"] + 720 + pad["bottom"] == plan["canvas_h"]
    # 16:9 into 9:16 grows the vertical borders, not the sides.
    assert pad["top"] > 0 and pad["bottom"] > 0
    assert plan["width"] % 64 == 0 and plan["height"] % 64 == 0
    assert max(plan["width"], plan["height"]) <= 768
    assert plan["stage1_w"] * 2 == plan["width"]
    assert plan["stage1_h"] * 2 == plan["height"]
    assert _png_size(plan["mask_png"]) == (plan["canvas_w"], plan["canvas_h"])
    rgb = paint_mask(
        plan["canvas_w"],
        plan["canvas_h"],
        fill=255,
        rect=(pad["left"], pad["top"], 1280, 720),
        rect_value=0,
    )
    assert _rgb_at(rgb, plan["canvas_w"], pad["left"] + 10, pad["top"] + 10) == (0, 0, 0)
    assert _rgb_at(rgb, plan["canvas_w"], 0, 0) == (255, 255, 255)


def test_explicit_target_canvas_contains_source():
    plan = outpaint_layout(640, 360, target_w=800, target_h=800)
    assert plan["canvas_w"] >= 640 and plan["canvas_h"] >= 360
    assert plan["pad"]["left"] % 8 == 0
    stage = fit_working_size(plan["canvas_w"], plan["canvas_h"])
    assert stage == (plan["width"], plan["height"])


def test_default_inpaint_mask_is_white_center():
    png = default_inpaint_mask_png(64, 64)
    assert _png_size(png) == (64, 64)
    assert png.startswith(b"\x89PNG")
    # zlib round-trip is covered by the writer; spot-check the raw painter.
    rgb = paint_mask(64, 64, fill=0, rect=(16, 16, 32, 32), rect_value=255)
    assert _rgb_at(rgb, 64, 32, 32) == (255, 255, 255)
    assert _rgb_at(rgb, 64, 0, 0) == (0, 0, 0)
    assert b"IDAT" in png and b"IEND" in png


def test_outpaint_mask_png_matches_canvas():
    png = outpaint_mask_png(96, 48, 16, 8, 64, 32)
    assert _png_size(png) == (96, 48)


def test_patch_writes_user_prompt_and_official_negative():
    wf, meta = load_and_patch_workflow(
        VARIANT,
        prompt="UNIQUE_SANDY_BEACH",
        seed=7,
        duration_s=5.0,
    )
    positive = wf["20"]["inputs"]["text"]
    negative = wf["21"]["inputs"]["text"]
    assert positive == "UNIQUE_SANDY_BEACH"
    assert "racing suit" not in positive
    assert negative == OFFICIAL_NEGATIVE
    assert "blurry, low quality" not in negative
    assert wf["40"]["inputs"]["length"] == 25
    assert wf["40"]["inputs"]["width"] == DEFAULT_WIDTH // 2
    assert wf["40"]["inputs"]["height"] == DEFAULT_HEIGHT // 2
    assert meta["width"] == DEFAULT_WIDTH
    assert meta["height"] == DEFAULT_HEIGHT
    assert meta["frames"] == 25
    assert wf["14"]["inputs"]["lora_name"] == "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    assert wf["14"]["inputs"]["strength_model"] == 0.5
    assert wf["15"]["inputs"]["lora_name"] == "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
    assert wf["15"]["inputs"]["strength_model"] == 1.0
    assert wf["10"]["inputs"]["ckpt_name"] == "ltx-2.3-22b-dev-fp8.safetensors"
    assert wf["36"]["inputs"]["spatial_radius"] == 15
    assert wf["62"]["inputs"]["spatial_radius"] == 30
    assert wf["75"]["inputs"]["mask_low_res_dilation"] == 6
    assert wf["75"]["inputs"]["trim_to_shortest"] is True
    assert wf["33"]["inputs"]["image"] == MASK_PLACEHOLDER
    assert wf["30"]["inputs"]["file"] == "robot_unitree.mp4"
    assert "attention_mask" not in wf["43"]["inputs"]
    assert wf["43"]["inputs"]["use_tiled_encode"] is False
    assert wf["74"]["inputs"]["overlap"] == 6


def test_outpaint_patch_sets_pads_and_skips_dilation():
    plan = outpaint_layout(1280, 720, aspect="9:16")
    wf, meta = load_and_patch_workflow(
        VARIANT,
        prompt="continue the street",
        seed=3,
        width=plan["width"],
        height=plan["height"],
        frames=25,
        inoutpaint={
            "mode": "outpaint",
            "pad": plan["pad"],
            "mask_png": plan["mask_png"],
        },
    )
    pad = wf["32"]["inputs"]
    assert pad["left"] == plan["pad"]["left"]
    assert pad["top"] == plan["pad"]["top"]
    assert pad["right"] == plan["pad"]["right"]
    assert pad["bottom"] == plan["pad"]["bottom"]
    assert wf["36"]["inputs"]["spatial_radius"] == 0
    assert wf["62"]["inputs"]["spatial_radius"] == 0
    assert wf["75"]["inputs"]["mask_low_res_dilation"] == 2
    assert wf["35"]["inputs"]["input"] == ["32", 1]
    assert wf["61"]["inputs"]["input"] == ["32", 1]
    assert wf["32"]["inputs"]["feathering"] == 0
    assert "inoutpaint_png_b64" in (wf["33"].get("_meta") or {})
    assert meta["width"] == plan["width"]
    assert wf["20"]["inputs"]["text"] == "continue the street"

    uploaded: list[str] = []

    def _upload(path: Path) -> str:
        assert path.name == "ltx23_inoutpaint_mask.png"
        assert path.read_bytes().startswith(b"\x89PNG")
        uploaded.append("ltx23_mask_uploaded.png")
        return uploaded[-1]

    name = ensure_inoutpaint_mask(wf, _upload)
    assert name == "ltx23_mask_uploaded.png"
    assert wf["33"]["inputs"]["image"] == name
    assert "inoutpaint_png_b64" not in (wf["33"].get("_meta") or {})


def _outpaint_graph(frames: int):
    plan = outpaint_layout(768, 448, aspect="9:16")
    return load_and_patch_workflow(
        VARIANT,
        prompt="continue the scene naturally past the frame edges",
        seed=42,
        width=plan["width"],
        height=plan["height"],
        frames=frames,
        inoutpaint={
            "mode": "outpaint",
            "pad": plan["pad"],
            "mask_png": plan["mask_png"],
        },
    )


def test_outpaint_mask_matches_requested_frames():
    """The pad mask is one frame. Outpaint must repeat it before the blend."""
    wf, _meta = _outpaint_graph(25)
    assert wf["40"]["inputs"]["length"] == 25
    assert wf["75"]["inputs"]["trim_to_shortest"] is False
    assert wf["35"]["inputs"]["input"] == ["32", 1]
    assert wf["61"]["inputs"]["input"] == ["32", 1]

    dups = {
        str((node.get("_meta") or {}).get("title")): (nid, node)
        for nid, node in wf.items()
        if isinstance(node, dict) and node.get("class_type") == "VHS_DuplicateMasks"
    }
    assert set(dups) == {
        "Repeat Outpaint Mask Stage 1",
        "Repeat Outpaint Mask Stage 2",
    }
    stage1_id, stage1 = dups["Repeat Outpaint Mask Stage 1"]
    stage2_id, stage2 = dups["Repeat Outpaint Mask Stage 2"]
    assert stage1["inputs"]["mask"] == ["35", 0]
    assert stage2["inputs"]["mask"] == ["61", 0]
    assert stage1["inputs"]["multiply_by"] == 25
    assert stage2["inputs"]["multiply_by"] == 25
    assert wf["36"]["inputs"]["mask"] == [stage1_id, 0]
    assert wf["62"]["inputs"]["mask"] == [stage2_id, 0]
    assert wf["75"]["inputs"]["mask"] == ["62", 0]

    short, _meta = _outpaint_graph(17)
    assert short["40"]["inputs"]["length"] == 17
    assert short["75"]["inputs"]["trim_to_shortest"] is False
    for node in short.values():
        if isinstance(node, dict) and node.get("class_type") == "VHS_DuplicateMasks":
            assert node["inputs"]["multiply_by"] == 17

    painted, _meta = load_and_patch_workflow(VARIANT, prompt="a potted plant", seed=1, frames=25)
    assert painted["75"]["inputs"]["trim_to_shortest"] is True
    assert painted["36"]["inputs"]["mask"] == ["35", 0]
    assert painted["62"]["inputs"]["mask"] == ["61", 0]
    assert not any(
        isinstance(node, dict) and node.get("class_type") == "VHS_DuplicateMasks"
        for node in painted.values()
    )

    info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
    report = validate_workflow(wf, info, file_label="outpaint-mask-length")
    assert report.ok, [str(item) for item in report.errors]


def test_user_mask_name_is_not_replaced():
    wf, _meta = load_and_patch_workflow(
        VARIANT,
        prompt="fill the sign",
        seed=1,
        frames=25,
        mask_name="my_mask.png",
    )
    assert wf["33"]["inputs"]["image"] == "my_mask.png"

    def _upload(_path: Path) -> str:
        raise AssertionError("user mask should not be regenerated")

    assert ensure_inoutpaint_mask(wf, _upload) is None


def test_template_and_patched_graph_validate_offline():
    info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
    raw = json.loads((WORKFLOWS_DIR / WORKFLOW_FILES[VARIANT]).read_text(encoding="utf-8"))
    save = raw["78"]["inputs"]
    assert save["format"] == "auto"
    assert save["format.codec"] == "auto"
    assert save["codec"] == "auto"
    report = validate_workflow(raw, info, file_label="ltx23_inoutpaint_api.json")
    assert report.ok, [str(item) for item in report.errors]
    wf, _meta = load_and_patch_workflow(VARIANT, prompt="a wet street", seed=1, frames=17)
    assert wf["78"]["inputs"]["format.codec"] == "auto"
    assert wf["78"]["inputs"]["codec"] == "auto"
    patched = validate_workflow(wf, info, file_label="patched")
    assert patched.ok, [str(item) for item in patched.errors]
    # The IC-LoRA is newer than the cached combo. Inventory makes that a warning.
    assert any("in-outpainting" in str(item) for item in patched.warnings)


def test_director_routes_outpaint_and_leaves_wan_inpaint():
    assert rule_based_variant("please inpaint this plate") == "wan_fun_inpaint"
    assert rule_based_variant("ltx outpaint the frame") == VARIANT
    assert rule_based_variant("extend the canvas to vertical") == VARIANT
    assert choose_variant("outpaint this clip to 9:16", has_video=True) == (VARIANT, "input")
    assert choose_variant("please inpaint this plate", has_video=True) == ("lipsync", "input")
    assert choose_variant("movie builder shot-by-shot", has_video=True) == ("lipsync", "input")


def test_doctor_lists_inoutpaint_lora():
    names = {row["name"] for row in snapshot()}
    assert "ltx23-inoutpaint-lora" in names


def test_comfy_run_provenance_uses_existing_schema(tmp_path: Path):
    wf, _meta = load_and_patch_workflow(
        VARIANT,
        prompt="UNIQUE_SANDY_BEACH",
        seed=11,
        frames=25,
        video_name="robot_unitree.mp4",
    )
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real mp4")
    payload = record_comfy_provenance(wf, clip, variant=VARIANT)
    assert payload is not None
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA
    assert missing_required(payload) == []
    assert payload["prompts"]["positive"] == "UNIQUE_SANDY_BEACH"
    assert payload["engine"]["variant"] == VARIANT
    assert payload["params"]["fps"] == 24
    assert "robot_unitree.mp4" in payload["params"]["refs"]
    sidecar = clip.with_name("clip.buddy.json")
    assert sidecar.is_file()
    assert record_comfy_provenance(wf, clip, variant="lipsync") is None


def test_finalize_halves_stage_size():
    wf = {
        "34": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {"width": 1, "height": 1},
            "_meta": {"title": "Resize Frames Stage 1"},
        },
        "60": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {"width": 1, "height": 1},
            "_meta": {"title": "Resize Frames Stage 2"},
        },
        "35": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {"width": 1, "height": 1},
            "_meta": {"title": "Resize Mask Stage 1"},
        },
        "40": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 9, "height": 9, "length": 25}},
    }
    finalize_inoutpaint_graph(wf, width=768, height=448, mode="inpaint")
    stage2 = wf["60"]["inputs"]
    assert stage2["resize_type.width"] == 768
    assert stage2["resize_type.height"] == 448
    assert stage2["resize_type.crop"] == "disabled"
    assert "width" not in stage2 and "height" not in stage2
    assert wf["34"]["inputs"]["resize_type.width"] == 384
    assert wf["34"]["inputs"]["resize_type.height"] == 224
    assert wf["35"]["inputs"]["resize_type"] == "match size"
    assert wf["35"]["inputs"]["resize_type.match"] == ["34", 0]
    assert wf["35"]["inputs"]["resize_type.crop"] == "center"
    assert "width" not in wf["35"]["inputs"]
    assert wf["40"]["inputs"]["width"] == 384
    assert wf["40"]["inputs"]["height"] == 224


def _resize_nodes(wf: dict) -> list[dict]:
    return [
        node
        for node in wf.values()
        if isinstance(node, dict) and node.get("class_type") == "ResizeImageMaskNode"
    ]


def test_prefixed_resize_inputs_on_template_and_patch():
    raw = json.loads((WORKFLOWS_DIR / WORKFLOW_FILES[VARIANT]).read_text(encoding="utf-8"))
    for node in _resize_nodes(raw):
        inputs = node["inputs"]
        assert "width" not in inputs and "height" not in inputs
        kind = inputs["resize_type"]
        if kind == "scale dimensions":
            assert isinstance(inputs["resize_type.width"], int)
            assert isinstance(inputs["resize_type.height"], int)
            assert inputs["resize_type.crop"] == "disabled"
        else:
            assert kind == "match size"
            assert inputs["resize_type.crop"] == "center"
            assert isinstance(inputs["resize_type.match"], list)
    wf, _meta = load_and_patch_workflow(VARIANT, prompt="a potted plant", seed=1, frames=25)
    frames = wf["60"]["inputs"]
    assert frames["resize_type.width"] == DEFAULT_WIDTH
    assert frames["resize_type.height"] == DEFAULT_HEIGHT
    assert frames["resize_type.crop"] == "disabled"
    assert wf["61"]["inputs"]["resize_type"] == "match size"
    assert wf["61"]["inputs"]["resize_type.match"] == ["60", 0]
    info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
    bare = {
        "1": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["2", 0],
                "resize_type": "scale dimensions",
                "scale_method": "area",
                "width": 768,
                "height": 448,
            },
        },
        "2": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    }
    # LoadImage may not be required for the missing-prefix errors.
    report = validate_workflow(bare, info, file_label="plain-resize")
    missing = {err.input_name for err in report.errors if "resize_type." in err.input_name}
    assert missing == {"resize_type.width", "resize_type.height", "resize_type.crop"}


def test_mask_repeat_matches_requested_frames():
    wf, _meta = load_and_patch_workflow(VARIANT, prompt="a potted plant", seed=1, frames=25)
    assert wf["80"]["inputs"]["amount"] == 25
    assert wf["80"]["inputs"]["image"] == ["79", 0]
    assert wf["81"]["inputs"]["image"] == ["80", 0]
    assert wf["35"]["inputs"]["input"] == ["81", 0]
    assert wf["61"]["inputs"]["input"] == ["81", 0]
    assert wf["75"]["inputs"]["trim_to_shortest"] is True
    short, _meta = load_and_patch_workflow(VARIANT, prompt="a potted plant", seed=1, frames=17)
    assert short["40"]["inputs"]["length"] == 17
    assert short["80"]["inputs"]["amount"] == 17


def test_prepare_trims_source_longer_than_latent(tmp_path: Path, monkeypatch):
    src = tmp_path / "smoke_src_48f.mp4"
    src.write_bytes(b"long-source")
    wf, _meta = load_and_patch_workflow(
        VARIANT,
        prompt="a potted plant",
        seed=1,
        frames=25,
        video_name="smoke_src_48f.mp4",
        mask_name="kept.png",
    )
    stash_local_video(wf, src)
    trimmed = tmp_path / "trimmed.mp4"

    def _probe(path: Path) -> int:
        if Path(path) == src:
            return 48
        return 25

    def _trim(src_path: Path, frames: int, dest: Path) -> None:
        assert Path(src_path) == src
        assert frames == 25
        dest.write_bytes(b"trimmed-25")
        trimmed.write_bytes(b"trimmed-25")

    monkeypatch.setattr("master_agent.comfy.inoutpaint.probe_video_frame_count", _probe)
    monkeypatch.setattr("master_agent.comfy.inoutpaint.trim_video_to_frames", _trim)
    uploaded: list[bytes] = []

    def _upload(path: Path) -> str:
        uploaded.append(path.read_bytes())
        return "uploaded.mp4"

    prepare_queue_inputs(wf, _upload)
    assert uploaded == [b"trimmed-25"]
    assert wf["30"]["inputs"]["file"] == "uploaded.mp4"
    assert "local_path" not in (wf["30"].get("_meta") or {})

    same = tmp_path / "already25.mp4"
    same.write_bytes(b"already")
    wf2, _ = load_and_patch_workflow(
        VARIANT, prompt="a potted plant", seed=1, frames=25, mask_name="kept.png"
    )
    stash_local_video(wf2, same)

    def _probe_same(path: Path) -> int:
        return 25

    def _trim_forbidden(*_args, **_kwargs):
        raise AssertionError("a clip that already matches the latent must not be re-encoded")

    monkeypatch.setattr("master_agent.comfy.inoutpaint.probe_video_frame_count", _probe_same)
    monkeypatch.setattr("master_agent.comfy.inoutpaint.trim_video_to_frames", _trim_forbidden)
    seen: list[bytes] = []

    def _upload_same(path: Path) -> str:
        seen.append(path.read_bytes())
        return "same.mp4"

    prepare_queue_inputs(wf2, _upload_same)
    assert seen == [b"already"]
    cmd = trim_video_command(src, 25, trimmed)
    assert cmd[cmd.index("-frames:v") + 1] == "25"
    assert "0:v:0" in cmd and "0:a?" in cmd


def test_prepare_does_not_trim_a_graph_without_inpaint_preprocess(tmp_path: Path, monkeypatch):
    src = tmp_path / "talk.mp4"
    src.write_bytes(b"talk")
    wf = {
        "1": {
            "class_type": "LoadVideo",
            "inputs": {"file": "talk.mp4"},
            "_meta": {},
        },
        "2": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"length": 25, "width": 64, "height": 64},
        },
    }
    stash_local_video(wf, src)

    def _boom(*_args, **_kwargs):
        raise AssertionError("a graph without the inpaint plate must not be trimmed")

    monkeypatch.setattr("master_agent.comfy.inoutpaint.fit_source_video", _boom)
    seen: list[bytes] = []

    def _upload(path: Path) -> str:
        seen.append(path.read_bytes())
        return "talk.mp4"

    prepare_queue_inputs(wf, _upload)
    assert seen == [b"talk"]


def test_fit_source_video_rejects_a_bad_trim(tmp_path: Path, monkeypatch):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")

    def _probe(path: Path) -> int:
        return 48 if Path(path) == src else 10

    def _trim(_src: Path, _frames: int, dest: Path) -> None:
        dest.write_bytes(b"bad")

    monkeypatch.setattr("master_agent.comfy.inoutpaint.probe_video_frame_count", _probe)
    monkeypatch.setattr("master_agent.comfy.inoutpaint.trim_video_to_frames", _trim)
    try:
        fit_source_video(src, 25)
    except Exception as exc:
        assert "25" in str(exc)
    else:
        raise AssertionError("a trim that does not land on the latent length must fail")


def test_provenance_duration_follows_the_output_file(tmp_path: Path, monkeypatch):
    wf, _meta = load_and_patch_workflow(
        VARIANT,
        prompt="a potted plant",
        seed=11,
        frames=25,
        video_name="robot_unitree.mp4",
    )
    clip = tmp_path / "one_frame.mp4"
    clip.write_bytes(b"not a real mp4")

    def _probe(_path):
        return {"exists": True, "frames": 1, "duration_s": 1 / 24, "size_bytes": 10}

    monkeypatch.setattr("master_agent.judge.probe.probe_video", _probe)
    payload = record_comfy_provenance(wf, clip, variant=VARIANT)
    assert payload is not None
    assert payload["params"]["frames"] == 1
    assert abs(payload["params"]["duration"] - (1 / 24)) < 1e-9
    assert payload["params"]["duration"] != 25 / 24
    sidecar = json.loads(sidecar_path(clip).read_text(encoding="utf-8"))
    assert sidecar["params"]["frames"] == 1
    assert abs(sidecar["params"]["duration"] - (1 / 24)) < 1e-9
