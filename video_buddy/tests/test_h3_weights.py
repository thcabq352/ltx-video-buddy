"""Scan-first / ask-to-download for MiniMax H3. Never auto-downloads.

Run: python -m pytest tests/test_h3_weights.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from master_agent.models.weights import (
    BUNDLES,
    H3_FL2VA_PREFERENCE,
    WEIGHT_FILES,
    MissingWeightsError,
    describe_h3_transformer_pick,
    download_files,
    find_weight_file,
    format_ask,
    h3_transformer_preference_order,
    require_weights,
    resolve_weight,
    scan_bundle,
)


def test_h3_official_filenames():
    assert WEIGHT_FILES["h3_fl2va"].filename == "minimax_h3_fl2va_pruned-Q4_K.gguf"
    assert WEIGHT_FILES["h3_ref2va"].filename == "minimax_h3_ref2va_pruned-Q4_K.gguf"
    assert WEIGHT_FILES["h3_text_encoder"].filename == (
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    )
    assert WEIGHT_FILES["h3_video_vae"].filename == "minimax_h3_video_vae_fp16.safetensors"
    assert WEIGHT_FILES["h3_audio_vae"].filename == "minimax_h3_audio_vae_fp32.safetensors"
    assert WEIGHT_FILES["h3_fl2va"].repo_id == "unsloth/MiniMax-H3-GGUF"
    assert WEIGHT_FILES["h3_text_encoder"].repo_id == "Comfy-Org/MiniMax-H3"
    assert WEIGHT_FILES["h3_fl2va"].mandatory is True
    assert WEIGHT_FILES["h3_fl2v_turbo"].mandatory is False
    for key in ("h3_fl2va", "h3_text_encoder", "h3_video_vae", "h3_audio_vae"):
        assert key in BUNDLES["h3_fl2va"]
    assert "h3_ref2va" in BUNDLES["h3_ref2va"]
    assert "h3_fl2va" not in BUNDLES["ltx25_all"]
    assert "transformer" in BUNDLES["ltx25_all"]
    assert "h3_fl2va" in BUNDLES["h3_all"]


def test_scan_missing_on_empty_roots(tmp_path: Path):
    empty = tmp_path / "models"
    empty.mkdir()
    status = scan_bundle("h3_fl2va", roots=[empty])
    assert status.ok is False
    names = {w.filename for w in status.missing_mandatory}
    assert WEIGHT_FILES["h3_fl2va"].filename in names
    assert WEIGHT_FILES["h3_text_encoder"].filename in names
    ask = format_ask(status)
    assert "download-models --h3 --yes" in ask
    assert "MiniMax H3 Community License" in ask
    assert "unsloth/MiniMax-H3-GGUF" in ask
    assert "Q4_K.gguf" in ask
    assert "also accepted locally" in ask


def test_scan_present_is_silent_ok(tmp_path: Path):
    root = tmp_path / "models"
    for w in [WEIGHT_FILES[k] for k in BUNDLES["h3_fl2va"]]:
        dest = root / w.dest_folder / w.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ok")
    status = scan_bundle("h3_fl2va", roots=[root])
    assert status.ok is True
    assert status.missing_mandatory == []
    assert require_weights("h3_t2v", roots=[root]) is not None


def test_require_weights_asks_when_missing(tmp_path: Path):
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(MissingWeightsError) as exc:
        require_weights("h3_r2v", roots=[empty])
    assert "ref2va" in str(exc.value).lower() or "Q4_K" in str(exc.value)
    assert exc.value.missing


def test_require_weights_skips_legacy_variants():
    assert require_weights("base") is None
    assert require_weights("wan22") is None


def test_download_refuses_without_yes(tmp_path: Path):
    missing = [WEIGHT_FILES["h3_audio_vae"]]
    with pytest.raises(MissingWeightsError) as exc:
        download_files(missing, dest_root=tmp_path, yes=False)
    assert "consent" in str(exc.value).lower()
    assert not list(tmp_path.rglob("*.safetensors"))


def test_zero_byte_file_counts_as_missing(tmp_path: Path):
    root = tmp_path / "models"
    dest = root / "diffusion_models" / WEIGHT_FILES["h3_fl2va"].filename
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"")
    assert dest.stat().st_size == 0
    assert resolve_weight(WEIGHT_FILES["h3_fl2va"], [root]) is None
    status = scan_bundle("h3_fl2va", roots=[root])
    assert WEIGHT_FILES["h3_fl2va"] in status.missing_mandatory


def test_gguf_and_comfy_te_satisfy_core(tmp_path: Path):
    root = tmp_path / "models"
    files = {
        "diffusion_models/gguf/minimax_h3_fl2va_pruned-Q4_K.gguf": b"gguf",
        "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors": b"te",
        "vae/minimax_h3_video_vae_fp16.safetensors": b"vae",
        "vae/minimax_h3_audio_vae_fp32.safetensors": b"avae",
    }
    for rel, blob in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    status = scan_bundle("h3_fl2va", roots=[root])
    assert status.ok is True
    found = resolve_weight(WEIGHT_FILES["h3_fl2va"], [root])
    assert found is not None
    assert found.name.endswith(".gguf")


def test_transformer_prefers_q4k_gguf_on_vram_16(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VRAM_GB", "16")
    root = tmp_path / "models"
    gguf = root / "diffusion_models" / "gguf" / H3_FL2VA_PREFERENCE[0]
    nvfp4 = root / "diffusion_models" / "minimax_h3_fl2va_pruned_nvfp4.safetensors"
    official = root / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    nvfp4.write_bytes(b"nvfp4")
    official.write_bytes(b"int8")
    found = resolve_weight(WEIGHT_FILES["h3_fl2va"], [root])
    assert found == gguf
    assert "GGUF Q4" in describe_h3_transformer_pick(found)
    order = h3_transformer_preference_order("h3_fl2va", vram_gb=16)
    assert order[0].endswith(".gguf")
    assert any("nvfp4" in n for n in order[1:3])


def test_transformer_prefers_nvfp4_over_int8_when_no_gguf(tmp_path: Path):
    root = tmp_path / "models"
    nvfp4 = root / "diffusion_models" / "minimax_h3_fl2va_pruned_nvfp4.safetensors"
    int8 = root / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    nvfp4.parent.mkdir(parents=True)
    nvfp4.write_bytes(b"nvfp4")
    int8.write_bytes(b"int8")
    found = resolve_weight(WEIGHT_FILES["h3_fl2va"], [root])
    assert found == nvfp4
    assert "NVFP4" in describe_h3_transformer_pick(found)


def test_int8_counts_when_no_gguf_or_nvfp4(tmp_path: Path):
    root = tmp_path / "models"
    int8 = root / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    int8.parent.mkdir(parents=True)
    int8.write_bytes(b"int8")
    found = resolve_weight(WEIGHT_FILES["h3_fl2va"], [root])
    assert found == int8


def test_gguf_te_is_last_resort(tmp_path: Path):
    root = tmp_path / "models"
    gguf_te = root / "text_encoders" / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    comfy_te = root / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    gguf_te.parent.mkdir(parents=True)
    gguf_te.write_bytes(b"gguf-te")
    comfy_te.write_bytes(b"comfy-te")
    found = resolve_weight(WEIGHT_FILES["h3_text_encoder"], [root])
    assert found == comfy_te


def test_ltx25_scan_unchanged_by_h3_keys(tmp_path: Path):
    empty = tmp_path / "models"
    empty.mkdir()
    status = scan_bundle("ltx25_core", roots=[empty])
    keys = {w.key for w in status.missing_mandatory}
    assert "h3_fl2va" not in keys
    assert "transformer" in keys
