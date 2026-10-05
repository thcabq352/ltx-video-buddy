"""User prompt must land on the guider's positive path.

CLIPTextEncode index 0 is the negative encoder on ltx25_t2v_i2v and lipsync.
The manifest has to name the positive primitive (or the positive encoder)
or the negative text becomes the conditioning CFG actually uses.

Run: python -m pytest tests/test_prompt_polarity.py -q
"""

from __future__ import annotations

from master_agent.comfy.catalog import LTX25_FILES
from master_agent.comfy.workflow_patcher import (
    _find_nodes_by_class,
    load_and_patch_workflow,
    load_workflow_template,
    variant_manifest,
)

_PROMPT = "UNIQUE_POSITIVE_SKULL_LOCK_8841"
_NEGATIVE = "UNIQUE_NEGATIVE_WATERMARK_8841"
_FOLLOW = {"positive", "text", "value", "prompt", "on_false", "on_true", "conditioning"}


def _is_link(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[1], int)
    )


def _positive_strings(workflow: dict) -> list[str]:
    """String widgets reached by following the guider positive input only."""
    found: list[str] = []
    guiders = _find_nodes_by_class(workflow, "CFGGuider")
    assert guiders, "graph has no CFGGuider"
    seen: set[str] = set()

    def walk(link) -> None:
        if not _is_link(link):
            return
        nid = str(link[0])
        if nid in seen:
            return
        seen.add(nid)
        inputs = (workflow.get(nid) or {}).get("inputs") or {}
        for key, value in inputs.items():
            if key not in _FOLLOW:
                continue
            if _is_link(value):
                walk(value)
            elif isinstance(value, str) and value.strip():
                found.append(value)

    for _nid, node in guiders:
        walk((node.get("inputs") or {}).get("positive"))
    return found


def test_guider_positive_is_the_user_prompt():
    for variant in ("ltx25_t2v_i2v", "lipsync"):
        wf, _meta = load_and_patch_workflow(
            variant,
            prompt=_PROMPT,
            negative_prompt=_NEGATIVE,
            seed=11,
            duration_s=2.0,
        )
        texts = _positive_strings(wf)
        assert any(_PROMPT in text for text in texts), (variant, texts)
        assert all(_NEGATIVE not in text for text in texts), (variant, texts)


def test_ltx25_t2v_keeps_clip_links_and_seeds_sampler_noise():
    wf, _meta = load_and_patch_workflow(
        "ltx25_t2v_i2v",
        prompt=_PROMPT,
        negative_prompt=_NEGATIVE,
        seed=11,
        duration_s=2.0,
    )
    assert _is_link(wf["5014:2483"]["inputs"]["text"])
    assert _is_link(wf["5014:2612"]["inputs"]["text"])
    assert wf["5508"]["inputs"]["value"] == _PROMPT
    assert wf["5509"]["inputs"]["value"] == _NEGATIVE
    assert wf["5516:4832"]["inputs"]["noise_seed"] == 11


def test_ltx25_seed_maps_name_real_random_noise():
    for vid in LTX25_FILES:
        spec = (variant_manifest(vid).get("fields") or {}).get("seed")
        if not spec:
            continue
        assert spec.get("class_type") != "KSampler", vid
        raw = load_workflow_template(vid)
        nids = spec.get("node_id")
        if nids is None:
            assert spec.get("class_type") == "RandomNoise", vid
            assert spec.get("input") == "noise_seed", vid
            continue
        if not isinstance(nids, list):
            nids = [nids]
        assert spec.get("input") == "noise_seed", vid
        for nid in nids:
            node = raw[str(nid)]
            assert node.get("class_type") == "RandomNoise", (vid, nid)
            assert "noise_seed" in (node.get("inputs") or {}), (vid, nid)
