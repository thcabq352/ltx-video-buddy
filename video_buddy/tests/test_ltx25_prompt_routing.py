"""LTX 2.5 graphs: the prompt reaches the Positive encoder, never the Negative one."""

from __future__ import annotations

import pytest

from master_agent.comfy.workflow_patcher import load_and_patch_workflow

POS = "POSITIVE-PROMPT-MARKER a fox runs through snow"
NEG = "NEGATIVE-PROMPT-MARKER blurry"

PRIMITIVES = {
    "ltx25_t2v_i2v": ("5508", "5509"),
    "ltx25_t2v_i2v_two_stage": ("5508", "5509"),
    "ltx25_flf2v": ("5508", "5509"),
    "ltx25_msr": ("5508", "5509"),
    "ltx25_v2v_ic_lora": ("5508", "5509"),
    "ltx25_a2v": ("5508", "5509"),
    "ltx25_t2a": ("6", "7"),
}


def _encoder_text(wf: dict, title_word: str) -> list:
    return [
        node["inputs"].get("text")
        for node in wf.values()
        if isinstance(node, dict)
        and node.get("class_type") == "CLIPTextEncode"
        and title_word in str((node.get("_meta") or {}).get("title") or "")
    ]


def _resolve(wf: dict, value):
    seen = 0
    while isinstance(value, list) and len(value) == 2 and str(value[0]) in wf and seen < 10:
        node = wf[str(value[0])]
        inputs = node.get("inputs") or {}
        if node.get("class_type") == "ComfySwitchNode":
            value = inputs.get("on_true") if inputs.get("switch") else inputs.get("on_false")
        elif "value" in inputs:
            value = inputs["value"]
        else:
            break
        seen += 1
    return value


@pytest.mark.parametrize("variant", sorted(PRIMITIVES))
def test_prompt_lands_on_positive_encoder(variant):
    wf, _meta = load_and_patch_workflow(
        variant,
        prompt=POS,
        negative_prompt=NEG,
        duration_s=2.0,
        image_name="face.png",
        audio_name="line.wav",
        video_name="clip.mp4",
    )
    pos_id, neg_id = PRIMITIVES[variant]
    assert wf[pos_id]["inputs"]["value"] == POS
    assert wf[neg_id]["inputs"]["value"] == NEG

    negatives = _encoder_text(wf, "Negative")
    positives = _encoder_text(wf, "Positive")
    assert negatives and positives
    for raw in negatives:
        assert POS not in str(_resolve(wf, raw))
    for raw in positives:
        assert _resolve(wf, raw) == POS
