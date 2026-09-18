"""Brain/Hands split — CapabilityContract + Hands fit. No GPU.

Wire-compatible with sibling your-video-buddy Rust
``buddy.capability.contract/v1`` (PR #9 / docs/BRAIN_HANDS.md).

Run: python -m pytest tests/test_brain_hands.py -q
"""

from __future__ import annotations

import json

import pytest

from master_agent.capability import (
    CAPABILITY_CONTRACT_SCHEMA,
    CAPACITY_KEYS,
    CapabilityContract,
    contract_from_variant,
)
from master_agent.hands import (
    CHAIN_CLIP_S,
    FitResult,
    HandsSnapshot,
    SnapshotHands,
    plan_last_frame_chain,
)
from unittest.mock import patch

from master_agent.orchestrator.director import (
    _director_payload,
    choose_variant,
    rank_story_candidates,
)


FORBIDDEN_CAPACITY = (
    "vram",
    "vram_gb",
    "expected_vram",
    "expected_vram_gb",
    "slot",
    "slots",
    "weight_path",
    "weights_path",
    "weight_paths",
    "models_dir",
    "free_vram",
    "free_vram_gb",
    "peak_vram",
    "vram_free",
    "vram_total",
    "vram_class",
)


def test_capacity_key_catalog_covers_forbidden():
    for key in FORBIDDEN_CAPACITY:
        assert key in CAPACITY_KEYS


def test_contract_serde_has_no_capacity_keys():
    contract = CapabilityContract(
        model="ltx-2.5",
        family="ltx25",
        variant="ltx25_t2v_i2v",
        resolution=(768, 512),
        duration_s=8.0,
        story_duration_s=20.0,
        audio=True,
        control_layers=("openpose", "depth"),
    )
    payload = contract.to_dict()
    assert payload["schema"] == CAPABILITY_CONTRACT_SCHEMA
    assert payload["schema"] == "buddy.capability.contract/v1"
    assert payload["model"] == "ltx-2.5"
    assert payload["family"] == "ltx25"
    assert payload["variant"] == "ltx25_t2v_i2v"
    assert payload["resolution"] == [768, 512]
    assert payload["duration_s"] == 8.0
    assert payload["story_duration_s"] == 20.0
    assert payload["audio"] is True
    assert payload["control_layers"] == ["openpose", "depth"]
    blob = json.dumps(payload)
    lowered = blob.lower()
    for key in FORBIDDEN_CAPACITY:
        assert key not in payload, key
        assert f'"{key}"' not in lowered

    roundtrip = CapabilityContract.from_dict(
        {
            **payload,
            "vram_gb": 16,
            "slot": "dit",
            "weight_path": "/models/foo.safetensors",
        }
    )
    again = roundtrip.to_dict()
    for key in FORBIDDEN_CAPACITY:
        assert key not in again


def test_contract_from_variant_fills_story_fields():
    contract = contract_from_variant(
        "ltx25_t2v_i2v",
        duration_s=8.0,
        story_duration_s=20.0,
        width=768,
        height=512,
        audio=True,
        control_layers=["edges"],
    )
    assert contract.schema == CAPABILITY_CONTRACT_SCHEMA
    assert contract.family == "ltx25"
    assert contract.model == "ltx-2.5"
    assert contract.variant == "ltx25_t2v_i2v"
    assert "vram" not in contract.to_dict()


def test_last_frame_chain_20s_is_three_times_8s():
    chain = plan_last_frame_chain(20.0)
    assert CHAIN_CLIP_S == 8.0
    assert len(chain.clips) == 3
    assert [c.duration_s for c in chain.clips] == [8.0, 8.0, 8.0]
    assert chain.clips[0].use_last_frame is False
    assert chain.clips[1].use_last_frame is True
    assert chain.clips[2].use_last_frame is True
    payload = chain.to_dict()
    assert payload["clip_s"] == 8.0
    assert payload["count"] == 3


def test_last_frame_chain_short_story_is_single_clip():
    chain = plan_last_frame_chain(5.0)
    assert [c.duration_s for c in chain.clips] == [5.0]
    assert chain.clips[0].use_last_frame is False


def test_snapshot_hands_rejects_insufficient_vram():
    hands = SnapshotHands(
        HandsSnapshot(vram_free_gb=2.0, vram_total_gb=8.0)
    )
    contract = contract_from_variant(
        "vb_movie_builder", duration_s=8.0, story_duration_s=8.0
    )
    fit = hands.can_fulfill(contract)
    assert isinstance(fit, FitResult)
    assert fit.ok is False
    assert fit.reason == "insufficient_vram"


def test_snapshot_hands_rejects_missing_weights():
    hands = SnapshotHands(HandsSnapshot(present_weights=frozenset()))
    contract = contract_from_variant(
        "ltx25_t2v_i2v", duration_s=8.0, story_duration_s=8.0
    )
    fit = hands.can_fulfill(contract)
    assert fit.ok is False
    assert fit.reason == "missing_weights"


def test_snapshot_hands_fits_when_vram_and_weights_ok():
    contract = contract_from_variant(
        "ltx25_t2v_i2v", duration_s=8.0, story_duration_s=8.0
    )
    from master_agent.models.vram_policy import workflow_row

    pack = workflow_row("ltx25_t2v_i2v").default_pack
    hands = SnapshotHands(
        HandsSnapshot(vram_free_gb=16.0, vram_total_gb=16.0, present_weights=frozenset({pack}))
    )
    fit = hands.can_fulfill(contract)
    assert fit.ok is True
    assert fit.reason is None


def test_rank_stories_before_hands_and_exclude_vram_from_brain():
    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False):
        ranked = rank_story_candidates("movie builder shot-by-shot feature")
    assert ranked[0][0] == "vb_movie_builder"
    assert any(v == "base" for v, _src in ranked)
    payload = _director_payload("movie builder shot-by-shot", fallback="vb_movie_builder")
    assert "vram_policy" not in payload
    blob = json.dumps(payload).lower()
    assert "vram" not in blob
    assert "safer" not in blob


def test_mock_hands_reject_falls_back_to_next_story():
    class RejectMovieHands:
        def can_fulfill(self, contract: CapabilityContract) -> FitResult:
            if contract.variant == "vb_movie_builder":
                return FitResult.reject("insufficient_vram", "mock reject")
            return FitResult.fit()

        def plan_last_frame_chain(self, story_duration_s: float):
            return plan_last_frame_chain(story_duration_s)

    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False):
        variant, source = choose_variant(
            "movie builder shot-by-shot feature",
            hands=RejectMovieHands(),
        )
    assert variant != "vb_movie_builder"
    assert variant == "base"
    assert source in {"rules", "fallback"}


def test_forced_variant_skips_hands_fallback():
    class RejectAll:
        def can_fulfill(self, contract: CapabilityContract) -> FitResult:
            return FitResult.reject("missing_weights", "no")

        def plan_last_frame_chain(self, story_duration_s: float):
            return plan_last_frame_chain(story_duration_s)

    assert choose_variant("anything", force="eros", hands=RejectAll()) == (
        "eros",
        "forced",
    )


def test_pipeline_uses_hands_chain_not_vram_ladder():
    from master_agent.orchestrator.pipeline import plan_story_segments

    segs = plan_story_segments(20.0, kind="run")
    assert segs == [8.0, 8.0, 8.0]


def test_music_video_does_not_use_last_frame_chain():
    from master_agent.orchestrator.pipeline import plan_story_segments

    segs = plan_story_segments(20.0, kind="music_video", quality="balanced")
    assert segs != [8.0, 8.0, 8.0]
    assert pytest.approx(sum(segs), abs=0.05) == 20.0
