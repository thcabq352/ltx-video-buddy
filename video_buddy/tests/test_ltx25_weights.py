"""Scan-first / ask-to-download. Never auto-downloads.

Run: python -m pytest tests/test_ltx25_weights.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from master_agent.models.weights import (
    BUNDLES,
    STUB_ALIASES,
    WEIGHT_FILES,
    MissingWeightsError,
    download_files,
    find_weight_file,
    format_ask,
    require_weights,
    scan_bundle,
)


def test_official_filenames_and_stub_aliases():
    assert WEIGHT_FILES["transformer"].filename.endswith(
        "distilled-transformer-comfy-int8-convrot.safetensors"
    )
    assert STUB_ALIASES["ltx-2.5-22b-distilled.safetensors"] == WEIGHT_FILES["transformer"].filename
    assert STUB_ALIASES["ltx-2.5-ic-lora.safetensors"] == WEIGHT_FILES["ic_lora"].filename
    assert WEIGHT_FILES["transformer"].gated is True
    assert WEIGHT_FILES["transformer"].mandatory is True
    assert WEIGHT_FILES["distilled_lora"].mandatory is False


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
    assert "21.5 GB" in ask


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
