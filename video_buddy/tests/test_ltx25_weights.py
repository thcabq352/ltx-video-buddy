"""Scan-first / ask-to-download. Never auto-downloads.

Run: python -m pytest tests/test_ltx25_weights.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from master_agent.models.weights import (
    BUNDLES,
    STUB_ALIASES,
    TRANSFORMER_PREFERENCE,
    WEIGHT_FILES,
    MissingWeightsError,
    describe_transformer_pick,
    download_files,
    find_weight_file,
    format_ask,
    require_weights,
    resolve_weight,
    scan_bundle,
    transformer_preference_order,
)


def test_official_filenames_and_stub_aliases():
    assert WEIGHT_FILES["transformer"].repo_filename == (
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    )
    assert WEIGHT_FILES["text_encoder"].repo_filename == (
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    )
    assert WEIGHT_FILES["duration_head"].repo_filename == (
        "model_patches/ltx-2.5-duration-head-bf16.safetensors"
    )
    assert WEIGHT_FILES["spatial_upscaler"].repo_filename == (
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    )
    assert STUB_ALIASES["ltx-2.5-22b-distilled.safetensors"] == WEIGHT_FILES["transformer"].filename
    assert STUB_ALIASES["ltx-2.5-ic-lora.safetensors"] == WEIGHT_FILES["ic_lora"].filename
    assert WEIGHT_FILES["transformer"].gated is True
    assert WEIGHT_FILES["transformer"].mandatory is True
    assert WEIGHT_FILES["duration_head"].mandatory is False
    assert WEIGHT_FILES["spatial_upscaler"].mandatory is True
    assert WEIGHT_FILES["distilled_lora"].mandatory is False
    for key in (
        "transformer",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "duration_head",
        "spatial_upscaler",
    ):
        assert key in BUNDLES["ltx25_core"]


def test_scan_missing_on_empty_roots(tmp_path: Path):
    empty = tmp_path / "models"
    empty.mkdir()
    status = scan_bundle("ltx25_core", roots=[empty])
    assert status.ok is False
    names = {w.filename for w in status.missing_mandatory}
    assert WEIGHT_FILES["transformer"].filename in names
    assert WEIGHT_FILES["text_encoder"].filename in names
    ask = format_ask(status)
    assert "download-models --ltx25 --yes" in ask
    assert "doctor --fix-models" in ask
    assert "gated" in ask.lower()
    assert "Lightricks/LTX-2.5" in ask
    assert "models/diffusion_models/" in ask
    assert "42.0 GB" in ask
    assert "also accepted locally" in ask
    assert "Q4_K_M.gguf" in ask
    assert "duration-head-bf16.safetensors" in ask
    assert "latent-spatial-upscaler" in ask


def test_scan_present_is_silent_ok(tmp_path: Path):
    root = tmp_path / "models"
    for w in [WEIGHT_FILES[k] for k in BUNDLES["ltx25_core"]]:
        dest = root / w.dest_folder / w.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ok")
    status = scan_bundle("ltx25_core", roots=[root])
    assert status.ok is True
    assert status.missing_mandatory == []
    assert require_weights("ltx25_t2v_i2v", roots=[root]) is not None


def test_stub_filename_counts_as_present(tmp_path: Path):
    root = tmp_path / "models"
    dest = root / "checkpoints" / "ltx-2.5-22b-distilled.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"stub")
    found = find_weight_file(WEIGHT_FILES["transformer"].filename, [root])
    assert found == dest


def test_require_weights_asks_when_missing(tmp_path: Path):
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(MissingWeightsError) as exc:
        require_weights("ltx25_t2v_i2v_two_stage", roots=[empty])
    assert "spatial" in str(exc.value).lower() or "upscaler" in str(exc.value).lower()
    assert exc.value.missing


def test_require_weights_skips_legacy_variants():
    assert require_weights("base") is None
    assert require_weights("wan22") is None


def test_download_refuses_without_yes(tmp_path: Path):
    missing = [WEIGHT_FILES["audio_vae"]]
    with pytest.raises(MissingWeightsError) as exc:
        download_files(missing, dest_root=tmp_path, yes=False)
    assert "consent" in str(exc.value).lower()
    assert not list(tmp_path.rglob("*.safetensors"))


def test_zero_byte_file_counts_as_missing(tmp_path: Path):
    root = tmp_path / "models"
    dest = root / "diffusion_models" / WEIGHT_FILES["transformer"].filename
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"")
    assert dest.stat().st_size == 0
    assert resolve_weight(WEIGHT_FILES["transformer"], [root]) is None
    status = scan_bundle("ltx25_core", roots=[root])
    assert WEIGHT_FILES["transformer"] in status.missing_mandatory


def test_gguf_and_heretic_te_satisfy_core(tmp_path: Path):
    root = tmp_path / "models"
    files = {
        "diffusion_models/gguf/ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf": b"gguf",
        "text_encoders/gemma4-12b-heretic-ltx25-int8convrot.safetensors": b"te",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors": b"vae",
        "vae/ltx-2.5-audio-vae-bf16.safetensors": b"avae",
        "model_patches/ltx-2.5-duration-head-bf16.safetensors": b"head",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": b"up",
    }
    for rel, blob in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    status = scan_bundle("ltx25_core", roots=[root])
    assert status.ok is True
    found = resolve_weight(WEIGHT_FILES["transformer"], [root])
    assert found is not None
    assert found.name.endswith(".gguf")


def test_transformer_prefers_gguf_over_nvfp4_and_int8(tmp_path: Path):
    root = tmp_path / "models"
    gguf = root / "diffusion_models" / "gguf" / TRANSFORMER_PREFERENCE[0]
    nvfp4 = root / "diffusion_models" / TRANSFORMER_PREFERENCE[1]
    official = root / "diffusion_models" / WEIGHT_FILES["transformer"].filename
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    nvfp4.write_bytes(b"nvfp4")
    official.write_bytes(b"bf16")
    found = resolve_weight(WEIGHT_FILES["transformer"], [root])
    assert found == gguf
    assert "GGUF Q4" in describe_transformer_pick(found)
    assert transformer_preference_order(vram_gb=16)[0].endswith(".gguf")
    assert transformer_preference_order(vram_gb=16)[1].endswith("nvfp4.safetensors")


def test_transformer_prefers_nvfp4_over_int8_when_no_gguf(tmp_path: Path):
    root = tmp_path / "models"
    nvfp4 = root / "diffusion_models" / TRANSFORMER_PREFERENCE[1]
    int8 = root / "diffusion_models" / TRANSFORMER_PREFERENCE[2]
    nvfp4.parent.mkdir(parents=True)
    nvfp4.write_bytes(b"nvfp4")
    int8.write_bytes(b"int8")
    found = resolve_weight(WEIGHT_FILES["transformer"], [root])
    assert found == nvfp4
    assert "NVFP4" in describe_transformer_pick(found)


def test_transformer_prefers_gguf_over_int8(tmp_path: Path):
    root = tmp_path / "models"
    gguf = root / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf"
    official = root / "diffusion_models" / WEIGHT_FILES["transformer"].filename
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    official.write_bytes(b"int8")
    found = resolve_weight(WEIGHT_FILES["transformer"], [root])
    assert found == gguf


def test_extra_models_dirs_env_is_searched(tmp_path: Path, monkeypatch):
    elsewhere = tmp_path / "other-volume" / "weights"
    dest = elsewhere / "diffusion_models" / WEIGHT_FILES["transformer"].filename
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"across-drives")
    monkeypatch.setenv("EXTRA_MODELS_DIRS", str(elsewhere))
    from master_agent.config import extra_models_dirs

    assert elsewhere in extra_models_dirs() or Path(str(elsewhere)) in extra_models_dirs()
    found = resolve_weight(WEIGHT_FILES["transformer"], extra_models_dirs())
    assert found == dest


def test_extra_model_paths_yaml_base_is_searched(tmp_path: Path, monkeypatch):
    volume = tmp_path / "comfy-on-other-disk"
    dest = volume / "models" / "vae" / WEIGHT_FILES["video_vae"].filename
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"yaml-base")
    yaml_path = tmp_path / "extra_model_paths.yaml"
    yaml_path.write_text(
        f"other:\n  base_path: {volume.as_posix()}\n  vae: models/vae\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("master_agent.config.COMFYUI_ROOT", tmp_path)
    from master_agent.models.weights import _extra_model_paths_yaml_roots

    roots = _extra_model_paths_yaml_roots()
    found = resolve_weight(WEIGHT_FILES["video_vae"], roots)
    assert found == dest


def test_class16_install_only_asks_for_zero_byte_duration_head(tmp_path: Path):
    """Reference 16GB-class layout: int8 + nvfp4 + gguf + heretic TE. No official bf16."""
    root = tmp_path / "models"
    files = {
        "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors": b"int8",
        "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors": b"nvfp4",
        "diffusion_models/gguf/ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf": b"gguf",
        "text_encoders/gemma4-12b-heretic-ltx25-int8convrot.safetensors": b"te",
        "vae/ltx-2.5-video-vae-bf16.safetensors": b"vae",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors": b"vae2",
        "vae/ltx-2.5-audio-vae-bf16.safetensors": b"avae",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": b"up",
        "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors": b"lora",
        "loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors": b"iclora",
    }
    for rel, blob in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    empty = root / "model_patches" / "ltx-2.5-duration-head-bf16.safetensors"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")

    core = scan_bundle("ltx25_core", roots=[root])
    assert core.missing_mandatory == []
    assert core.ok is True
    assert "duration_head" in {w.key for w in core.missing_optional}
    assert "transformer" in core.found_paths
    assert core.found_paths["transformer"].endswith(".gguf")
    assert "text_encoder" in core.found_paths
    assert "heretic" in core.found_paths["text_encoder"]
    assert WEIGHT_FILES["transformer"].filename not in {Path(p).name for p in core.found_paths.values()}

    ic = scan_bundle("ltx25_iclora", roots=[root])
    assert "ic_lora" not in {w.key for w in ic.missing_mandatory}
    assert Path(ic.found_paths["ic_lora"]).name.endswith("pixel-spatial-upscaler-x2-1.0.safetensors")


def test_a2v_require_weights_without_duration_head(tmp_path: Path):
    root = tmp_path / "models"
    files = {
        "diffusion_models/gguf/ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf": b"gguf",
        "text_encoders/gemma4-12b-heretic-ltx25-int8convrot.safetensors": b"te",
        "vae/ltx-2.5-video-vae-bf16.safetensors": b"vae",
        "vae/ltx-2.5-audio-vae-bf16.safetensors": b"avae",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": b"up",
    }
    for rel, blob in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    status = require_weights("ltx25_a2v", roots=[root])
    assert status is not None
    assert status.ok is True
    assert "duration_head" not in {w.key for w in status.missing_mandatory}


def test_hf_hub_snapshot_counts_as_present(tmp_path: Path):
    snap = (
        tmp_path
        / "hub"
        / "models--Lightricks--LTX-2.5"
        / "snapshots"
        / "abcd"
        / "diffusion_models"
    )
    snap.mkdir(parents=True)
    dest = snap / WEIGHT_FILES["transformer"].filename
    dest.write_bytes(b"from-hub")
    found = resolve_weight(WEIGHT_FILES["transformer"], [tmp_path / "hub" / "models--Lightricks--LTX-2.5" / "snapshots"])
    assert found == dest
