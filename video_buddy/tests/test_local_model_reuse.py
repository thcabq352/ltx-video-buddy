"""Local-first model reuse: do not re-download files already on disk.

Repro: Buddy used to treat a weight as missing unless it sat at the exact
``MODELS_DIR/<folder>/<official-name>`` destination. Files in Comfy
``models/``, ``EXTRA_MODELS_DIRS``, the Hugging Face hub cache, or Windows
folder-prefixed names (``wan\\file.safetensors``) triggered a Hub fetch.

Run: python -m pytest tests/test_local_model_reuse.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from master_agent.config import resolve_model_path
from master_agent.models.download import download_hub_file
from master_agent.models.inventory import check_bundles, scan_inventory
from master_agent.models.weights import (
    WEIGHT_FILES,
    download_files,
    find_weight_file,
    format_ask,
    model_search_roots,
    resolve_weight,
    scan_bundle,
)


WAN_FP8 = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
WAN_PREFIXED = rf"wan\{WAN_FP8}"


def _write(path: Path, blob: bytes = b"local-weight") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def test_folder_prefixed_name_finds_basename(tmp_path: Path):
    dest = _write(tmp_path / "diffusion_models" / "wan" / WAN_FP8)
    found = find_weight_file(WAN_PREFIXED, [tmp_path])
    assert found == dest


def test_folder_prefixed_name_finds_flat_basename(tmp_path: Path):
    dest = _write(tmp_path / "diffusion_models" / WAN_FP8)
    found = find_weight_file(WAN_PREFIXED, [tmp_path])
    assert found == dest


def test_resolve_model_path_uses_extra_models_dirs(tmp_path: Path, monkeypatch):
    elsewhere = tmp_path / "other-volume" / "ComfyUI" / "models"
    dest = _write(elsewhere / "diffusion_models" / WAN_FP8)
    monkeypatch.setenv("EXTRA_MODELS_DIRS", str(elsewhere))
    monkeypatch.setattr("master_agent.config.MODELS_DIR", tmp_path / "missing-models")
    monkeypatch.setattr("master_agent.config.COMFYUI_ROOT", tmp_path / "missing-comfy")
    found = resolve_model_path(WAN_PREFIXED)
    assert found == dest


def test_hf_cache_any_repo_is_a_search_root(tmp_path: Path, monkeypatch):
    """Flux (and other non-LTX) hub snapshots must count as already-present."""
    hub = tmp_path / "hf-home" / "hub"
    snap = hub / "models--Comfy-Org--flux1-dev" / "snapshots" / "abcd"
    dest = _write(snap / "flux1-dev-fp8.safetensors")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    roots = model_search_roots()
    assert any(dest.is_relative_to(root) or root == snap or snap in root.parents or root == dest.parent for root in roots) or any(
        "flux1-dev" in str(r) or str(hub) in str(r) for r in roots
    )
    found = find_weight_file("flux1-dev-fp8.safetensors")
    assert found == dest


def test_download_hub_file_skips_comfy_copy(tmp_path: Path, monkeypatch):
    comfy = tmp_path / "ComfyUI" / "models"
    existing = _write(comfy / "diffusion_models" / "flux1-dev-fp8.safetensors", b"already-here")
    dest = tmp_path / "buddy-models" / "diffusion_models" / "flux1-dev-fp8.safetensors"
    logs: list[str] = []

    def boom(*_a, **_k):
        raise AssertionError("hf_hub_download must not run when a local file exists")

    monkeypatch.setattr("master_agent.config.MODELS_DIR", tmp_path / "buddy-models")
    monkeypatch.setattr("master_agent.config.COMFYUI_ROOT", tmp_path / "ComfyUI")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    out = download_hub_file(
        repo_id="Comfy-Org/flux1-dev",
        repo_filename="flux1-dev-fp8.safetensors",
        dest=dest,
        progress=logs.append,
    )
    assert out == existing
    assert dest.exists() is False
    joined = "\n".join(logs)
    assert "local file found" in joined.lower()
    assert "not re-download" in joined.lower()
    assert str(existing) in joined


def test_download_hub_file_skips_hf_cache(tmp_path: Path, monkeypatch):
    hub = tmp_path / "hf-home" / "hub"
    cached = _write(
        hub
        / "models--Comfy-Org--flux1-dev"
        / "snapshots"
        / "ffff"
        / "flux1-dev-fp8.safetensors",
        b"cached-blob",
    )
    dest = tmp_path / "buddy-models" / "diffusion_models" / "flux1-dev-fp8.safetensors"
    logs: list[str] = []

    def boom(*_a, **_k):
        raise AssertionError("hf_hub_download must not run when HF cache has the file")

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr("master_agent.config.MODELS_DIR", tmp_path / "buddy-models")
    monkeypatch.setattr("master_agent.config.COMFYUI_ROOT", tmp_path / "no-comfy")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    out = download_hub_file(
        repo_id="Comfy-Org/flux1-dev",
        repo_filename="flux1-dev-fp8.safetensors",
        dest=dest,
        progress=logs.append,
    )
    assert out == cached
    assert "local file found" in "\n".join(logs).lower()


def test_download_hub_file_refuses_when_roots_missing(tmp_path: Path, monkeypatch):
    dest = tmp_path / "nowhere" / "models" / "vae" / "missing.safetensors"
    logs: list[str] = []

    def boom(*_a, **_k):
        raise AssertionError("must not start a Hub download when paths are misconfigured")

    monkeypatch.setattr("master_agent.config.MODELS_DIR", tmp_path / "missing-models")
    monkeypatch.setattr("master_agent.config.COMFYUI_ROOT", tmp_path / "missing-comfy")
    monkeypatch.setenv("EXTRA_MODELS_DIRS", str(tmp_path / "also-missing"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    from master_agent.models.download import LocalModelNotFound

    with pytest.raises(LocalModelNotFound) as exc:
        download_hub_file(
            repo_id="Comfy-Org/flux1-dev",
            repo_filename="flux1-dev-fp8.safetensors",
            dest=dest,
            progress=logs.append,
        )
    msg = str(exc.value)
    assert "missing-models" in msg
    assert "missing-comfy" in msg or "also-missing" in msg
    assert "flux1-dev-fp8.safetensors" in msg
    assert not dest.exists()


def test_download_files_skips_when_accepted_alias_exists(tmp_path: Path, monkeypatch):
    alias = _write(
        tmp_path
        / "comfy"
        / "models"
        / "diffusion_models"
        / "gguf"
        / "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf",
        b"gguf-local",
    )
    dest_root = tmp_path / "buddy-models"
    logs: list[str] = []

    def boom(*_a, **_k):
        raise AssertionError("download_hub_file must not run for a present alias")

    monkeypatch.setattr("master_agent.models.download.download_hub_file", boom)
    monkeypatch.setattr("master_agent.config.MODELS_DIR", dest_root)
    monkeypatch.setattr("master_agent.config.COMFYUI_ROOT", tmp_path / "comfy")
    paths = download_files([WEIGHT_FILES["transformer"]], dest_root=dest_root, progress=logs.append, yes=True)
    assert paths == [alias]
    assert "local file found" in "\n".join(logs).lower()
    assert not list(dest_root.rglob("*.safetensors"))


def test_scan_bundle_lists_checked_paths_in_ask(tmp_path: Path):
    empty = tmp_path / "models"
    empty.mkdir()
    status = scan_bundle("ltx25_core", roots=[empty])
    ask = format_ask(status)
    assert str(empty) in ask
    assert "paths checked" in ask.lower()


def test_inventory_sees_extra_dir_and_prefixed_bundle(tmp_path: Path, monkeypatch):
    extra = tmp_path / "extra" / "models"
    _write(extra / "diffusion_models" / "wan" / WAN_FP8)
    _write(extra / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
    _write(extra / "vae" / "wan_2.1_vae.safetensors")
    _write(
        extra / "loras" / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
    )
    monkeypatch.setenv("EXTRA_MODELS_DIRS", str(extra))
    monkeypatch.setattr("master_agent.config.MODELS_DIR", tmp_path / "empty-models")
    monkeypatch.setattr("master_agent.config.COMFYUI_ROOT", tmp_path / "empty-comfy")
    (tmp_path / "empty-models").mkdir()
    inv = scan_inventory(
        models_dir=tmp_path / "empty-models",
        comfyui_root=tmp_path / "empty-comfy",
        write=False,
    )
    assert inv.resolve(WAN_PREFIXED) is not None
    assert inv.resolve(WAN_FP8) is not None
    bundles = check_bundles(inv.entries)
    assert bundles["wan22"]["runnable"] is True or WAN_FP8 in {
        e.name for e in inv.entries
    }
    # Folder-prefixed MODEL_FILES.wan22.checkpoint_high must count as present.
    assert bundles["wan22"]["present"].get("checkpoint_high")


def test_resolve_weight_finds_flux_in_hf_cache(tmp_path: Path, monkeypatch):
    hub = tmp_path / "hf-home" / "hub"
    dest = _write(
        hub
        / "models--city96--FLUX.1-dev-gguf"
        / "snapshots"
        / "abc"
        / "flux1-dev-Q4_K_S.gguf",
        b"gguf",
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    found = resolve_weight(WEIGHT_FILES["flux"])
    assert found == dest
    status = scan_bundle("flux")
    assert status.ok is True
    assert "flux" in status.found_paths
