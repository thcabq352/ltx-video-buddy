"""GGUF / Comfy loader names must match object_info combo strings.

Tower bug: UnetLoaderGGUF rejected bare
``minimax_h3_fl2va_pruned-Q4_K.gguf`` because the live list has
``gguf\\minimax_h3_fl2va_pruned-Q4_K.gguf`` (Windows) or ``gguf/…`` (posix).

Run: python -m pytest tests/test_loader_names.py tests/test_h3_catalog.py -q
"""

from __future__ import annotations

from pathlib import Path

from master_agent.comfy.loader_names import (
    match_combo_name,
    name_from_inventory,
    name_from_local_path,
    normalize_loader_name,
    normalize_loader_widgets,
)
from master_agent.comfy.validator import validate_workflow
from master_agent.comfy.workflow_patcher import _find_nodes_by_class, load_and_patch_workflow
from master_agent.models.inventory import Inventory, ModelEntry

H3_BARE = "minimax_h3_fl2va_pruned-Q4_K.gguf"
H3_WIN = r"gguf\minimax_h3_fl2va_pruned-Q4_K.gguf"
H3_POSIX = "gguf/minimax_h3_fl2va_pruned-Q4_K.gguf"


def _entry(rel_path: str, *, name: str | None = None) -> ModelEntry:
    slash = rel_path.replace("\\", "/")
    return ModelEntry(
        name=name or slash.rsplit("/", 1)[-1],
        rel_path=slash,
        folder=slash.split("/", 1)[0],
        role="diffusion",
        size_bytes=1,
        mtime="t",
        root="comfyui",
    )


def _inventory(*rels: str) -> Inventory:
    return Inventory(
        generated_at="t",
        models_dir="m",
        comfyui_models_dir="c",
        entries=[_entry(rel) for rel in rels],
    )


def _unet_object_info(choices: list[str]) -> dict:
    return {
        "UnetLoaderGGUF": {
            "input": {"required": {"unet_name": [choices, {}]}},
            "output": ["MODEL"],
        }
    }


def test_match_windows_combo_from_bare_basename():
    assert match_combo_name(H3_BARE, [H3_WIN, "other.gguf"]) == H3_WIN


def test_match_posix_combo_from_bare_basename():
    assert match_combo_name(H3_BARE, [H3_POSIX, "other.gguf"]) == H3_POSIX


def test_already_prefixed_windows_is_not_double_prefixed():
    assert match_combo_name(H3_WIN, [H3_WIN, "gguf\\\\gguf\\\\x.gguf"]) == H3_WIN
    assert normalize_loader_name(H3_WIN, choices=[H3_WIN]) == H3_WIN
    assert normalize_loader_name(H3_WIN, inventory=_inventory(f"diffusion_models/{H3_POSIX}")) == H3_WIN


def test_already_prefixed_posix_is_not_double_prefixed():
    assert match_combo_name(H3_POSIX, [H3_POSIX]) == H3_POSIX
    assert normalize_loader_name(H3_POSIX, choices=[H3_POSIX]) == H3_POSIX
    assert normalize_loader_name(H3_POSIX, inventory=_inventory(f"diffusion_models/{H3_POSIX}")) == H3_POSIX


def test_slash_style_rewritten_to_exact_list_entry():
    assert match_combo_name(H3_POSIX, [H3_WIN]) == H3_WIN
    assert match_combo_name(H3_WIN, [H3_POSIX]) == H3_POSIX


def test_exact_list_hit_wins_over_prefixed_sibling():
    assert match_combo_name(H3_BARE, [H3_BARE, H3_WIN]) == H3_BARE


def test_inventory_adds_folder_prefix_once():
    inv = _inventory("diffusion_models/gguf/" + H3_BARE)
    assert name_from_inventory(H3_BARE, inv) == H3_POSIX
    assert normalize_loader_name(H3_BARE, inventory=inv) == H3_POSIX
    assert name_from_inventory(H3_WIN, inv) == H3_WIN


def test_name_from_local_path_strips_role_folder():
    path = Path("models") / "diffusion_models" / "gguf" / H3_BARE
    assert name_from_local_path(path) == H3_POSIX
    flat = Path("models") / "diffusion_models" / H3_BARE
    assert name_from_local_path(flat) == H3_BARE


def test_normalize_loader_widgets_rewrites_h3_unet():
    wf = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": H3_BARE},
        }
    }
    changed = normalize_loader_widgets(wf, object_info=_unet_object_info([H3_WIN]))
    assert wf["1"]["inputs"]["unet_name"] == H3_WIN
    assert changed == [("1", "unet_name", H3_BARE, H3_WIN)]


def test_validator_rewrites_bare_gguf_to_windows_combo():
    wf = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": H3_BARE},
        }
    }
    report = validate_workflow(
        wf,
        _unet_object_info([H3_WIN]),
        file_label="h3",
        inventory=_inventory("diffusion_models/gguf/" + H3_BARE),
    )
    assert wf["1"]["inputs"]["unet_name"] == H3_WIN
    assert report.ok
    assert any("normalized" in str(w).lower() for w in report.warnings)


def test_validator_rewrites_bare_gguf_to_posix_combo():
    wf = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": H3_BARE},
        }
    }
    report = validate_workflow(
        wf,
        _unet_object_info([H3_POSIX]),
        file_label="h3",
        inventory=_inventory("diffusion_models/gguf/" + H3_BARE),
    )
    assert wf["1"]["inputs"]["unet_name"] == H3_POSIX
    assert report.ok


def test_h3_t2v_patcher_emits_list_value(tmp_path, monkeypatch):
    root = tmp_path / "models"
    gguf = root / "diffusion_models" / "gguf" / H3_BARE
    te = root / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    gguf.parent.mkdir(parents=True)
    te.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    te.write_bytes(b"te")
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [root])
    info = _unet_object_info([H3_WIN, r"gguf\other.gguf"])
    wf, _meta = load_and_patch_workflow(
        "h3_t2v",
        prompt="hero shot",
        seed=1,
        duration_s=5.0,
        object_info=info,
    )
    ggufs = _find_nodes_by_class(wf, "UnetLoaderGGUF")
    assert ggufs
    assert ggufs[0][1]["inputs"]["unet_name"] == H3_WIN
    assert H3_WIN in info["UnetLoaderGGUF"]["input"]["required"]["unet_name"][0]


def test_h3_t2v_path_prefix_without_object_info(tmp_path, monkeypatch):
    root = tmp_path / "models"
    gguf = root / "diffusion_models" / "gguf" / H3_BARE
    te = root / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    gguf.parent.mkdir(parents=True)
    te.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    te.write_bytes(b"te")
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [root])
    wf, _meta = load_and_patch_workflow(
        "h3_t2v",
        prompt="hero shot",
        seed=1,
        duration_s=5.0,
    )
    ggufs = _find_nodes_by_class(wf, "UnetLoaderGGUF")
    assert ggufs
    name = ggufs[0][1]["inputs"]["unet_name"]
    assert name in {H3_POSIX, H3_WIN}
    assert not name.startswith("gguf/gguf")
    assert not name.startswith(r"gguf\gguf")
