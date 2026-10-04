"""Blaze vertical remakes: graph prep and CLI flags. No GPU.

Run: python -m pytest tests/test_blaze_remake.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.comfy.blaze_remake import (
    MAX_FRAMES,
    SAFE_FRAMES,
    SAFE_HEIGHT,
    SAFE_WIDTH,
    SOURCE_FRAMES,
    prepare_blaze_remake,
    provenance_for_clip,
    upload_remake_stills,
)
from master_agent.comfy.workflow_patcher import (
    _find_nodes_by_class,
    image_feeds_sampler_latent,
    load_and_patch_workflow,
)
from master_agent.config import OBJECT_INFO_CACHE
from master_agent.provenance import CLIP_PROVENANCE_SCHEMA, missing_required, read_clip_provenance
from master_agent.__main__ import cmd_comfy


def _names(workflow, class_type: str, field: str) -> list:
    return [
        (node.get("inputs") or {}).get(field)
        for _nid, node in _find_nodes_by_class(workflow, class_type)
    ]


def _primitive(workflow, title_part: str) -> str:
    for _nid, node in _find_nodes_by_class(workflow, "PrimitiveStringMultiline"):
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        if title_part in title:
            return str((node.get("inputs") or {}).get("value") or "")
    raise AssertionError(f"no primitive {title_part}")


def test_cached_object_info_has_ingredients_guide_not_the_msr_node():
    info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
    assert "LTXAddVideoICLoRAGuide" in info
    assert "LTXICLoRALoaderModelOnly" in info
    assert "ComfyUILTX25MSRMultiReferenceGuide" not in info


def test_safe_default_locks_both_guides_and_provenance():
    prepared = prepare_blaze_remake(
        "concert",
        scott="inputs/blaze/scott.jpg",
        blaze="inputs/blaze/blaze.jpg",
    )
    wf = prepared.workflow
    assert prepared.recipe_id == "blaze-concert"
    assert prepared.warning == ""
    assert (prepared.width, prepared.height, prepared.frames) == (
        SAFE_WIDTH,
        SAFE_HEIGHT,
        SAFE_FRAMES,
    )
    assert prepared.frames == 97
    assert prepared.frames != SOURCE_FRAMES
    images = _names(wf, "LoadImage", "image")
    assert images.count("scott.jpg") == 1
    assert images.count("blaze.jpg") == 1
    assert image_feeds_sampler_latent(wf, "scott.jpg")
    assert image_feeds_sampler_latent(wf, "blaze.jpg")
    guides = _find_nodes_by_class(wf, "LTXAddVideoICLoRAGuide")
    assert len(guides) == 2
    assert all((node.get("inputs") or {}).get("strength") == 1 for _nid, node in guides)
    loras = _names(wf, "LTXICLoRALoaderModelOnly", "lora_name")
    assert any("ic-lora-ingredients" in str(name) for name in loras)
    latent = _find_nodes_by_class(wf, "EmptyLTXVLatentVideo")[0][1]["inputs"]
    assert (latent["width"], latent["height"], latent["length"]) == (448, 800, 97)
    assert "shorter_size" not in json.dumps(
        [
            node.get("inputs")
            for _nid, node in _find_nodes_by_class(wf, "ResizeImageMaskNode")
        ]
    )
    for _nid, node in _find_nodes_by_class(wf, "ResizeImageMaskNode"):
        inputs = node["inputs"]
        assert inputs["resize_type"] == "scale dimensions"
        assert inputs["resize_type.width"] == 448
        assert inputs["resize_type.height"] == 800
    repeats = _find_nodes_by_class(wf, "RepeatImageBatch")
    assert repeats
    assert all(isinstance((node["inputs"] or {})["amount"], list) for _nid, node in repeats)
    durations = [
        (node.get("inputs") or {}).get("value")
        for _nid, node in _find_nodes_by_class(wf, "PrimitiveFloat")
        if "duration in seconds" in str((node.get("_meta") or {}).get("title") or "").lower()
    ]
    assert durations == [97 / 24]
    audio = _find_nodes_by_class(wf, "LTXVEmptyLatentAudio")
    assert audio[0][1]["inputs"]["frames_number"] == 97
    positive = _primitive(wf, "positive")
    negative = _primitive(wf, "negative")
    assert "Scott" in positive and "Blaze" in positive
    assert "backwards" in positive and "diamond" in positive
    assert "mhetliicca" in negative
    assert "sticker" in negative
    assert "morph" in negative
    prefixes = _names(wf, "SaveVideo", "filename_prefix")
    assert prefixes == ["blaze-concert"]
    assert all("intro" not in str(item) for item in prefixes)
    payload = prepared.provenance
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA
    assert missing_required(payload) == []
    assert payload["engine"]["variant"] == "ltx25_msr"
    assert payload["params"]["refs"] == ["scott.jpg", "blaze.jpg"]
    assert payload["params"]["size"] == [448, 800]
    assert payload["hash"] is None
    assert not str(payload["output_path"]).startswith("/")


def test_pier_prompt_forbids_captions_and_stickers():
    prepared = prepare_blaze_remake(
        "clearwater-pier",
        scott="scott.jpg",
        blaze="blaze-alt-1.jpg",
    )
    assert prepared.recipe_id == "blaze-pier"
    positive = _primitive(prepared.workflow, "positive")
    negative = _primitive(prepared.workflow, "negative")
    assert "Clearwater" in positive
    assert "Blaze" in positive
    assert "caption" in negative or "subtitles" in negative
    assert "sticker" in negative
    assert "identity drift" in negative
    assert prepared.provenance["params"]["refs"] == ["scott.jpg", "blaze-alt-1.jpg"]


def test_prompt_override_and_source_canvas_warning():
    prepared = prepare_blaze_remake(
        "blaze-concert",
        scott="scott.jpg",
        blaze="blaze.jpg",
        prompt="CUSTOM LOCK keep the Scott still",
        negative="NO TEXT NO STICKER",
        width=720,
        height=1280,
        frames=241,
        seed=7,
    )
    assert (prepared.width, prepared.height, prepared.frames) == (704, 1280, 241)
    assert "704x1280" in prepared.warning
    assert "unmeasured" in prepared.warning
    assert _primitive(prepared.workflow, "positive") == "CUSTOM LOCK keep the Scott still"
    assert _primitive(prepared.workflow, "negative") == "NO TEXT NO STICKER"
    latent = _find_nodes_by_class(prepared.workflow, "EmptyLTXVLatentVideo")[0][1]["inputs"]
    assert latent["length"] == 241
    assert latent["width"] == 704
    assert latent["height"] == 1280
    assert prepared.provenance["params"]["seed"] == 7
    assert prepared.provenance["params"]["size"] == [704, 1280]


def test_length_eight_snaps_to_nine_and_thirty_seconds_is_refused():
    prepared = prepare_blaze_remake(
        "pier",
        scott="scott.jpg",
        blaze="blaze.jpg",
        frames=8,
    )
    latent = _find_nodes_by_class(prepared.workflow, "EmptyLTXVLatentVideo")[0][1]["inputs"]
    assert latent["length"] == 9
    assert latent["length"] != 8
    with pytest.raises(ValueError, match="241"):
        prepare_blaze_remake(
            "blaze-pier",
            scott="scott.jpg",
            blaze="blaze.jpg",
            frames=MAX_FRAMES + 8,
        )
    assert SOURCE_FRAMES == MAX_FRAMES


def test_same_still_and_half_canvas_are_refused():
    with pytest.raises(ValueError, match="different"):
        prepare_blaze_remake("concert", scott="scott.jpg", blaze="scott.jpg")
    with pytest.raises(ValueError, match="both width and height"):
        prepare_blaze_remake("concert", scott="a.jpg", blaze="b.jpg", width=720)


def test_default_msr_patch_still_clamps_vertical_source():
    _wf, meta = load_and_patch_workflow(
        "ltx25_msr",
        prompt="stadium",
        width=720,
        height=1280,
        frames=97,
        seed=1,
    )
    assert meta["width"] == 704
    assert meta["height"] == 512


def test_upload_requires_local_files(tmp_path: Path):
    prepared = prepare_blaze_remake(
        "concert",
        scott=str(tmp_path / "scott.jpg"),
        blaze=str(tmp_path / "blaze.jpg"),
    )
    with pytest.raises(FileNotFoundError, match="Scott"):
        upload_remake_stills(
            prepared.workflow,
            lambda path: path.name,
            str(tmp_path / "scott.jpg"),
            str(tmp_path / "blaze.jpg"),
        )


def test_provenance_hashes_only_a_real_file(tmp_path: Path):
    prepared = prepare_blaze_remake("pier", scott="scott.jpg", blaze="blaze.jpg")
    missing = provenance_for_clip(prepared, tmp_path / "blaze-pier.mp4")
    assert missing["hash"] is None
    assert missing["schema"] == CLIP_PROVENANCE_SCHEMA
    clip = tmp_path / "blaze-pier.mp4"
    clip.write_bytes(b"clip-bytes")
    found = provenance_for_clip(prepared, clip)
    assert found["hash"]
    assert found["output_path"] == "outputs/blaze-pier.mp4"
    assert not str(found["output_path"]).startswith("/")


def _ns(**kwargs) -> Namespace:
    values = {
        "comfy_command": "run",
        "mode": "generate",
        "recipe": "blaze-concert",
        "scott": "inputs/blaze/scott.jpg",
        "blaze": "inputs/blaze/blaze.jpg",
        "prompt": "",
        "negative_prompt": None,
        "width": None,
        "height": None,
        "frames": None,
        "seed": None,
        "variant": "base",
        "out": None,
        "prepare": True,
        "set": [],
        "workflow_json": None,
        "template": None,
    }
    values.update(kwargs)
    return Namespace(**values)


def test_cli_prepare_does_not_queue(monkeypatch, tmp_path: Path, capsys):
    def boom(*_a, **_k):
        raise AssertionError("execute_prepared must not run in --prepare")

    monkeypatch.setattr("master_agent.comfy.cli_run.execute_prepared", boom)

    class FakeClient:
        def load_object_info(self, prefer_live=True):
            return {}, "cache"

        def upload_image(self, path, image_type="input", overwrite=True):
            raise AssertionError("prepare must not upload")

    monkeypatch.setattr("master_agent.__main__.ComfyClient", FakeClient)
    monkeypatch.setattr("master_agent.comfy.cli_run.lint_or_raise", lambda *_a, **_k: None)
    out = tmp_path / "concert.json"
    rc = cmd_comfy(_ns(out=str(out), prompt="OVERRIDE FACE LOCK"))
    assert rc == 0
    text = capsys.readouterr().out
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line == "{")
    body = json.loads("\n".join(lines[start:]))
    assert body["ok"] is True
    assert body["recipe"] == "blaze-concert"
    assert body["frames"] == SAFE_FRAMES
    assert body["height"] == SAFE_HEIGHT
    assert body["provenance"]["schema"] == CLIP_PROVENANCE_SCHEMA
    assert body["provenance"]["prompts"]["positive"] == "OVERRIDE FACE LOCK"
    assert missing_required(body["provenance"]) == []
    sidecar = read_clip_provenance(tmp_path / "concert.mp4")
    assert sidecar is not None
    assert sidecar["schema"] == CLIP_PROVENANCE_SCHEMA
    assert sidecar["params"]["refs"] == ["scott.jpg", "blaze.jpg"]
    assert sidecar["hash"] is None
    graph = json.loads(out.read_text(encoding="utf-8"))
    assert image_feeds_sampler_latent(graph, "scott.jpg")
    assert image_feeds_sampler_latent(graph, "blaze.jpg")


def test_cli_requires_stills_and_rejects_attach_recipe_on_run(capsys):
    rc = cmd_comfy(_ns(scott=None, prepare=True))
    assert rc == 1
    assert "requires --scott and --blaze" in capsys.readouterr().out
    rc = cmd_comfy(_ns(recipe="previs.json", prepare=True))
    assert rc == 1
    assert "blaze-concert" in capsys.readouterr().out


def test_cli_queue_uploads_and_writes_hashed_sidecar(monkeypatch, tmp_path: Path):
    scott = tmp_path / "scott.jpg"
    blaze = tmp_path / "blaze.jpg"
    scott.write_bytes(b"scott-still")
    blaze.write_bytes(b"blaze-still")
    seen: dict = {}

    class FakeClient:
        def upload_image(self, path, image_type="input", overwrite=True):
            return "input/" + Path(path).name

    def fake_exec(workflow, **kwargs):
        images = _names(workflow, "LoadImage", "image")
        seen["images"] = images
        clip = tmp_path / "out.mp4"
        clip.write_bytes(b"rendered-bytes")
        return {
            "status": "done",
            "prompt_id": "abc",
            "video_path": str(clip),
            "outputs": [str(clip)],
            "nodes": len(workflow),
        }

    monkeypatch.setattr("master_agent.__main__.ComfyClient", FakeClient)
    monkeypatch.setattr("master_agent.comfy.cli_run.execute_prepared", fake_exec)
    rc = cmd_comfy(
        _ns(
            prepare=False,
            recipe="blaze-pier",
            scott=str(scott),
            blaze=str(blaze),
        )
    )
    assert rc == 0
    assert seen["images"] == ["input/scott.jpg", "input/blaze.jpg"]
    sidecar = read_clip_provenance(tmp_path / "out.mp4")
    assert sidecar is not None
    assert sidecar["schema"] == CLIP_PROVENANCE_SCHEMA
    assert sidecar["hash"]
    assert sidecar["params"]["refs"] == ["scott.jpg", "blaze.jpg"]
    assert sidecar["engine"]["workflow_id"] == "ltx25_msr"
    assert missing_required(sidecar) == []
