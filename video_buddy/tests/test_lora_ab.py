"""LoRA A/B lock: encoder + seed family. No GPU.

Run: python -m pytest tests/test_lora_ab.py -q
"""

from __future__ import annotations

import pytest

from master_agent.config import MODEL_FILES
from master_agent.lora.ab_lock import (
    AbLockError,
    apply_ab_lock_to_workflow,
    lock_lora_ab,
    preserve_checkpoint_style,
    validate_ab_pair,
)


def test_default_text_encoder_is_not_heretic():
    te = MODEL_FILES["base"]["text_encoder"]
    assert "heretic" not in te.lower()
    assert "gemma_3_12B" in te


def test_lock_only_changes_lora_name_and_strength():
    base = {
        "text_encoder": "gemma_3_12B_it_fp8_scaled.safetensors",
        "seed": 20260911,
        "checkpoint": "wan\\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "lora": "alpha.safetensors",
        "strength": 0.8,
    }
    challenger = lock_lora_ab(base, lora_name="beta.safetensors", strength=0.55)
    assert challenger["text_encoder"] == base["text_encoder"]
    assert challenger["seed"] == base["seed"]
    assert challenger["lora"] == "beta.safetensors"
    assert challenger["strength"] == 0.55
    assert challenger["checkpoint"] == "wan\\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    assert "\\" in challenger["checkpoint"]
    assert "/" not in challenger["checkpoint"]
    assert validate_ab_pair(base, challenger) == []


def test_heretic_swap_is_rejected():
    base = {
        "text_encoder": "gemma_3_12B_it_fp8_scaled.safetensors",
        "seed": 11,
    }
    with pytest.raises(AbLockError):
        lock_lora_ab(
            {**base, "text_encoder": "qwen3-vl-heretic"},
            lora_name="beta.safetensors",
        )
    bad = {
        "text_encoder": "qwen3-vl-heretic",
        "seed": 11,
        "lora": "beta.safetensors",
    }
    assert any("Heretic" in err for err in validate_ab_pair(base, bad))


def test_seed_family_and_encoder_must_stay():
    base = {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors", "seed": 100}
    drifted = {"text_encoder": base["text_encoder"], "seed": 999, "lora": "b.safetensors"}
    assert any("seed family" in err for err in validate_ab_pair(base, drifted))
    swapped = {"text_encoder": "other.safetensors", "seed": 100, "lora": "b.safetensors"}
    assert any("text encoder" in err for err in validate_ab_pair(base, swapped))


def test_windows_checkpoint_backslash_style():
    raw = r"wan\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    assert preserve_checkpoint_style(raw) == raw
    assert "\\" in preserve_checkpoint_style(raw)


def test_apply_ab_lock_to_workflow_leaves_lora_writable():
    wf = {
        "1": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {"text_encoder": "gemma_3_12B_it_fp8_scaled.safetensors"},
        },
        "2": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
        "3": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": "a.safetensors", "strength_model": 1.0, "model": ["9", 0]},
        },
    }
    apply_ab_lock_to_workflow(
        wf,
        text_encoder="gemma_3_12B_it_fp8_scaled.safetensors",
        seed=42,
        lora_name="b.safetensors",
        strength=0.4,
    )
    assert wf["1"]["inputs"]["text_encoder"] == "gemma_3_12B_it_fp8_scaled.safetensors"
    assert wf["2"]["inputs"]["noise_seed"] == 42
    assert wf["3"]["inputs"]["lora_name"] == "b.safetensors"
    assert wf["3"]["inputs"]["strength_model"] == 0.4
    with pytest.raises(AbLockError):
        apply_ab_lock_to_workflow(wf, text_encoder="qwen3-vl-heretic", seed=42)
