"""Sulphur LTX 2.3 studio pack: graphs, path resolve, catalog. No GPU.

Run: python -m pytest tests/test_sulphur_workflows.py -q
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from master_agent.comfy.catalog import (
    clear_catalog_cache,
    default_variant_ids,
    is_known_variant,
    list_catalog_items,
    resolve_variant,
    resolve_workflow_path,
)
from master_agent.comfy.sulphur import (
    SULPHUR_FILES,
    is_sulphur_variant,
    resolve_sulphur_lora,
)
from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.config import WORKFLOW_FILES, WORKFLOWS_DIR
from master_agent.models.vram_policy import workflow_row
from master_agent.orchestrator.director import rule_based_variant

# Tower bytes. A change here means the committed graph is no longer the source.
_SOURCE_SHA256 = {
    "sulphur/ltx23_i2v_base.json": "571c572d7d377a4c4c102331731ac93862e60cb4b53fd121d554e75adeeafb08",
    "sulphur/ltx23_i2v_distilled.json": "0af6043519b25037edc46978bbf429dffaf23d46ac3496731fc795ab97c5e0ef",
    "sulphur/ltx23_t2v_base.json": "da499a27938b30d8c5814ac66d30f91d66fbe75687533fb0d625ae0f34c8856e",
    "sulphur/ltx23_t2v_distilled.json": "c17a69296352ca626223f75388b981184cef2a3a9ef67d70c71a716709aa7152",
}
_EROS_SHA256 = "868c3d110462b6d3be24f28030d410b7ff4ed7ef57b4f77fffdfd2b91442c545"


@pytest.fixture(autouse=True)
def _clear_catalog():
    clear_catalog_cache()
    yield
    clear_catalog_cache()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ui_active_loras(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in data.get("nodes") or []:
        if node.get("type") != "LoraLoaderModelOnly":
            continue
        if node.get("mode") in (2, 4):
            continue
        widgets = node.get("widgets_values") or []
        if widgets and isinstance(widgets[0], str):
            names.append(widgets[0])
    return names


def _api_loras(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in data.values():
        if not isinstance(node, dict) or node.get("class_type") != "LoraLoaderModelOnly":
            continue
        lora_name = (node.get("inputs") or {}).get("lora_name")
        if isinstance(lora_name, str):
            names.append(lora_name)
    return names


def test_tower_source_graphs_are_unchanged_and_nonempty():
    for rel, digest in _SOURCE_SHA256.items():
        path = WORKFLOWS_DIR / rel
        assert path.is_file(), rel
        assert path.stat().st_size > 0
        assert _sha256(path) == digest


def test_eros_workflow_is_unchanged():
    path = WORKFLOWS_DIR / "eros_t2v_i2v.json"
    assert WORKFLOW_FILES["eros"] == "eros_t2v_i2v.json"
    assert _sha256(path) == _EROS_SHA256


def test_runtime_files_are_api_and_keep_active_lora_tokens():
    distilled = WORKFLOWS_DIR / SULPHUR_FILES["ltx23_i2v_distilled"]
    distilled_data = json.loads(distilled.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, dict) and "class_type" in node for node in distilled_data.values()
    )
    assert "nodes" not in distilled_data

    pairs = {
        "ltx23_i2v_base": "sulphur/ltx23_i2v_base.json",
        "ltx23_t2v_base": "sulphur/ltx23_t2v_base.json",
        "ltx23_t2v_distilled": "sulphur/ltx23_t2v_distilled.json",
    }
    for vid, ui_rel in pairs.items():
        api_path = WORKFLOWS_DIR / SULPHUR_FILES[vid]
        assert api_path.is_file()
        assert api_path.stat().st_size > 0
        assert _api_loras(api_path) == _ui_active_loras(WORKFLOWS_DIR / ui_rel)

    base_api = json.loads(
        (WORKFLOWS_DIR / SULPHUR_FILES["ltx23_i2v_base"]).read_text(encoding="utf-8")
    )
    assert base_api["68"]["inputs"]["resolution"] == 1024
    assert base_api["68"]["inputs"]["method"] == "NEAREST"
    assert base_api["70"]["inputs"]["scale_by"] == 0.5


def test_catalog_lists_sulphur_api_graphs():
    ids = default_variant_ids()
    items = {item["id"]: item for item in list_catalog_items()}
    for vid, rel in SULPHUR_FILES.items():
        assert vid in ids
        assert is_known_variant(vid)
        assert resolve_variant(vid).path == rel
        assert resolve_workflow_path(vid).is_file()
        assert items[vid]["kind"] == "variant"
        assert WORKFLOW_FILES[vid] == rel
    # UI sources stay discoverable as files and are not the queue variant.
    assert items["sulphur/ltx23_i2v_base.json"]["kind"] == "file"
    assert "sulphur/ltx23_i2v_base.json" not in WORKFLOW_FILES.values()


def test_keyword_routing_selects_sulphur_without_stealing_eros():
    assert rule_based_variant("sulphur studio clip") == "ltx23_i2v_distilled"
    assert rule_based_variant("sulphur i2v distilled") == "ltx23_i2v_distilled"
    assert rule_based_variant("sulphur i2v base") == "ltx23_i2v_base"
    assert rule_based_variant("sulphur t2v") == "ltx23_t2v_base"
    assert rule_based_variant("sulphur t2v distilled") == "ltx23_t2v_distilled"
    assert rule_based_variant("10eros teaser") == "eros"
    assert rule_based_variant("rain on a window") == "base"


def test_vram_rows_are_heavy_and_not_the_16gb_default():
    for vid in SULPHUR_FILES:
        row = workflow_row(vid)
        assert row.vram_class == "heavy"
        assert row.safer_alternate == "base"
        assert row.family == "ltx23"


def test_resolve_sulphur_lora_prefers_subdir_then_loras_root(tmp_path: Path):
    sub = tmp_path / "loras" / "sulphur"
    sub.mkdir(parents=True)
    token = "tower_local.safetensors"
    placed = sub / token
    placed.write_bytes(b"not-a-real-weight")
    found = resolve_sulphur_lora(token, [tmp_path])
    assert found == placed

    flat = tmp_path / "loras" / token
    flat.write_bytes(b"flat")
    # Subdir still wins when both exist.
    assert resolve_sulphur_lora(token, [tmp_path]) == placed
    placed.unlink()
    assert resolve_sulphur_lora(token, [tmp_path]) == flat
    assert resolve_sulphur_lora("", [tmp_path]) is None


def test_patch_keeps_lora_tokens_and_linked_i2v_size():
    prompt = "slow push across a quiet street"
    negative = "watermark"
    wf, meta = load_and_patch_workflow(
        "ltx23_i2v_distilled",
        prompt=prompt,
        negative_prompt=negative,
        seed=7,
        width=768,
        height=512,
        duration_s=3.0,
        first_image="still.png",
    )
    assert meta["variant"] == "ltx23_i2v_distilled"
    assert is_sulphur_variant("ltx23_i2v_distilled")
    loras = _api_loras_from_dict(wf)
    original = _api_loras(WORKFLOWS_DIR / SULPHUR_FILES["ltx23_i2v_distilled"])
    assert loras == original
    latent = wf["21"]["inputs"]
    assert isinstance(latent["width"], list)
    assert isinstance(latent["height"], list)
    assert isinstance(latent["length"], list)
    assert wf["29"]["inputs"]["value"] == prompt
    assert wf["41"]["inputs"]["text"] == negative
    assert wf["41"]["inputs"]["text"] != prompt
    assert wf["67"]["inputs"]["image"] == "still.png"

    t2v, _meta = load_and_patch_workflow(
        "ltx23_t2v_base",
        prompt=prompt,
        negative_prompt=negative,
        seed=7,
        width=768,
        height=512,
        duration_s=3.0,
    )
    assert t2v["40"]["inputs"]["value"] == 768
    assert t2v["25"]["inputs"]["value"] == 512
    assert t2v["27"]["inputs"]["value"] == _meta["frames"]
    assert isinstance(t2v["21"]["inputs"]["width"], list)


def _api_loras_from_dict(workflow: dict) -> list[str]:
    names: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict) or node.get("class_type") != "LoraLoaderModelOnly":
            continue
        lora_name = (node.get("inputs") or {}).get("lora_name")
        if isinstance(lora_name, str):
            names.append(lora_name)
    return names


def test_pack_does_not_add_weight_blobs_or_download_targets():
    root = WORKFLOWS_DIR / "sulphur"
    banned = {".safetensors", ".gguf", ".ckpt", ".pt", ".bin", ".pth"}
    for path in root.iterdir():
        assert path.suffix.lower() not in banned

    from master_agent.models.weights import WEIGHT_FILES

    blob = " ".join(
        f"{weight.filename} {weight.repo_id} {weight.repo_filename}"
        for weight in WEIGHT_FILES.values()
    )
    assert "sulphur_final" not in blob
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "civitai" not in readme.lower()
    assert "huggingface.co" not in readme.lower()
    assert "sulphur_final" not in readme
