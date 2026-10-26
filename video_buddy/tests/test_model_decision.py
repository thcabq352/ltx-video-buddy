"""Route first, then a reasoned GGUF. No GPU.

Run: python -m pytest tests/test_model_decision.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from master_agent.models.model_decision import (
    ModelSelectionError,
    decide_generation,
    require_model,
    select_model,
)

Q4 = "LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf"
Q3 = "ltx-2.3-22b-distilled-Q3_K_S.gguf"
DEV = "LTX-2.3-dev-Q4_K_S.gguf"
EROS = "10Eros_v1.5-Q4_K_M.gguf"
SULPHUR = "sulphur_dev-Q3_K_S.gguf"
KLEIN = "flux-2-klein-4b-fp8.safetensors"
H3 = "minimax_h3_fl2va_pruned-Q4_K.gguf"
LTX25 = "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf"


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weight")
    return path


def _pair(text: str) -> None:
    for phrase in (
        "Sulphur vs base",
        "Eros vs 10Eros",
        "Q4 vs Q3",
        "dev vs distilled",
        "MiniMax H3 vs LTX",
        "Paid hosted MiniMax H3 is not auto-selected",
        "H3_OPT_IN",
    ):
        assert phrase in text, phrase


def test_intent_picks_base_eros_and_directors(tmp_path: Path):
    _write(tmp_path / "unet" / Q4)
    _write(tmp_path / "unet" / Q3)
    _write(tmp_path / "unet" / DEV)
    _write(tmp_path / "diffusion_models" / EROS)
    _write(tmp_path / "unet" / SULPHUR)
    base = select_model("base", [tmp_path], vram_gb=16)
    eros = select_model("eros", [tmp_path], vram_gb=16)
    directors = select_model("directors", [tmp_path], vram_gb=16)
    assert base.chosen == Q4
    assert eros.chosen == EROS
    assert directors.chosen == DEV
    for decision in (base, eros, directors):
        _pair(decision.text)
        assert decision.error is None


def test_vram_downgrade_rejects_q4(tmp_path: Path):
    _write(tmp_path / "unet" / Q4)
    _write(tmp_path / "unet" / Q3)
    decision = select_model("base", [tmp_path], vram_gb=8)
    assert decision.chosen == Q3
    assert "exceeds VRAM" in decision.text
    assert "Q4 vs Q3" in decision.text


def test_override_wins_and_warns_when_over_vram(tmp_path: Path):
    _write(tmp_path / "unet" / DEV)
    _write(tmp_path / "unet" / Q3)
    decision = select_model("base", [tmp_path], vram_gb=8, override=DEV)
    assert decision.chosen == DEV
    assert decision.warning
    assert "wins" in decision.warning
    assert "8.0" in decision.warning


def test_nothing_on_disk_names_what_would_fit(tmp_path: Path):
    decision = select_model("base", [tmp_path], vram_gb=16)
    assert decision.chosen is None
    assert decision.error
    assert Q4 in decision.error
    assert "would fit" in decision.error
    with pytest.raises(ModelSelectionError, match="would fit"):
        require_model(decision)


def test_flux_i2v_wins_only_when_installed_and_it_fits(tmp_path: Path):
    missing = tmp_path / "no-flux.json"
    absent = decide_generation(
        "base",
        roots=[tmp_path],
        vram_gb=16,
        flux_workflow=missing,
        allowed={"base", "flux_ltx_i2v"},
    )
    assert absent.route == "ltx_t2v"
    assert absent.apply_route is False
    assert "flux_ltx_i2v_api.json" in absent.text
    assert "PR #92" in absent.text
    assert "cursor/gguf-vram-flux-ltx" in absent.text
    assert "T2V was chosen" in absent.text
    _pair(absent.text)

    graph = _write(tmp_path / "flux_ltx_i2v_api.json")
    _write(tmp_path / "diffusion_models" / KLEIN)
    _write(tmp_path / "unet" / Q4)
    chosen = decide_generation(
        "base",
        roots=[tmp_path],
        vram_gb=16,
        flux_workflow=graph,
        allowed={"base", "flux_ltx_i2v"},
    )
    assert chosen.route == "flux_i2v"
    assert chosen.apply_route is True
    assert chosen.variant == "flux_ltx_i2v"
    assert chosen.flux_model == KLEIN
    assert chosen.chosen == Q4
    assert "T2V was not chosen" in chosen.text
    assert "Input image: no" in chosen.text
    assert "Flux I2V installed: yes" in chosen.text

    tight = decide_generation(
        "base",
        roots=[tmp_path],
        vram_gb=10,
        flux_workflow=graph,
        allowed={"base", "flux_ltx_i2v"},
    )
    assert tight.route == "ltx_t2v"
    assert tight.apply_route is False
    assert "exceeds" in tight.flux_detail


def test_user_image_stays_on_ltx_i2v(tmp_path: Path):
    graph = _write(tmp_path / "flux_ltx_i2v_api.json")
    _write(tmp_path / "diffusion_models" / KLEIN)
    _write(tmp_path / "unet" / Q4)
    decision = decide_generation(
        "base",
        roots=[tmp_path],
        vram_gb=16,
        has_image=True,
        flux_workflow=graph,
        allowed={"base", "flux_ltx_i2v"},
    )
    assert decision.route == "ltx_i2v"
    assert decision.apply_route is False
    assert "Input image: yes" in decision.text
    assert "not a new Flux still" in decision.text
    assert "T2V was not chosen" in decision.text


def test_text_to_video_phrase_locks_t2v(tmp_path: Path):
    graph = _write(tmp_path / "flux_ltx_i2v_api.json")
    _write(tmp_path / "diffusion_models" / KLEIN)
    _write(tmp_path / "unet" / Q4)
    decision = decide_generation(
        "base",
        request="text-to-video only, no flux",
        roots=[tmp_path],
        vram_gb=16,
        flux_workflow=graph,
        allowed={"base", "flux_ltx_i2v"},
    )
    assert decision.route == "ltx_t2v"
    assert decision.apply_route is False
    assert "text-to-video" in decision.text or "T2V was chosen" in decision.text


def test_hosted_h3_is_not_auto_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "paid-key")
    monkeypatch.delenv("H3_OPT_IN", raising=False)
    decision = decide_generation(
        "base",
        roots=[tmp_path],
        vram_gb=16,
        flux_workflow=tmp_path / "missing.json",
    )
    assert decision.apply_route is False
    assert decision.h3_mode == "hosted"
    assert "Paid hosted MiniMax H3 is not auto-selected" in decision.text
    assert "cost" in decision.text
    assert "H3_OPT_IN" in decision.text


def test_hosted_opt_in_without_local_weights_does_not_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MINIMAX_API_KEY", "paid-key")
    monkeypatch.setenv("H3_OPT_IN", "1")
    decision = decide_generation(
        "base",
        request="make it talk",
        roots=[tmp_path],
        vram_gb=16,
        flux_workflow=tmp_path / "missing.json",
    )
    assert decision.apply_route is False
    assert decision.route != "h3"
    assert "no hosted client" in decision.text
    assert "cost" in decision.text


def test_local_h3_wins_talking_only_with_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write(tmp_path / "unet" / H3)
    _write(tmp_path / "unet" / Q4)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("H3_OPT_IN", raising=False)
    quiet = decide_generation(
        "base",
        request="rain on a window",
        roots=[tmp_path],
        vram_gb=16,
        flux_workflow=tmp_path / "missing.json",
    )
    assert quiet.route != "h3"
    assert "not a talking" in quiet.text or "was not chosen" in quiet.text

    monkeypatch.setenv("H3_OPT_IN", "1")
    talking = decide_generation(
        "base",
        request="lip-sync this line",
        roots=[tmp_path],
        vram_gb=16,
        flux_workflow=tmp_path / "missing.json",
    )
    assert talking.route == "h3"
    assert talking.apply_route is True
    assert talking.variant == "h3_t2v"
    assert "talking" in talking.text
    assert "lip-sync" in talking.text
    assert talking.chosen == H3


def test_named_h3_with_local_weights_beats_flux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("H3_OPT_IN", raising=False)
    graph = _write(tmp_path / "flux_ltx_i2v_api.json")
    _write(tmp_path / "diffusion_models" / KLEIN)
    _write(tmp_path / "unet" / LTX25)
    _write(tmp_path / "unet" / H3)
    decision = decide_generation(
        "base",
        request="minimax h3 talking portrait",
        roots=[tmp_path],
        vram_gb=16,
        flux_workflow=graph,
        allowed={"base", "flux_ltx_i2v", "h3_t2v"},
    )
    assert decision.route == "h3"
    assert decision.apply_route is True
    assert "MiniMax H3 wins" in decision.text
