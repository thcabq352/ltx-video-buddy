"""Match an ingested graph to a specialized family and reuse that path.

inoutpaint goes through ``finalize_inoutpaint_graph`` (PR #44:
``VHS_DuplicateMasks`` and ``trim_to_shortest=false`` on the 2.5 blend).
sulphur goes through ``patch_sulphur_graph``. lipsync keeps linked prompt
widgets on the source node and, at queue time, ``prepare_queue_inputs``.

Unmatched graphs stay on the generic patcher. ``--no-family-route`` forces
that generic path even when a family matches.
"""

from __future__ import annotations

from typing import Any

from master_agent.comfy.ingest.classify import class_types, iter_nodes

_LORA_CLASSES = frozenset({"LoraLoader", "LoraLoaderModelOnly", "LTXICLoRALoaderModelOnly"})


_VIDEO_GUIDES = frozenset({"LTXAddVideoICLoRAGuide", "LTXVSetAudioRefTokens"})
_SULPHUR_HINT_NODES = frozenset({"PathchSageAttentionKJ", "LTX2SamplingPreviewOverride"})


def _has_sulphur_lora(workflow: dict[str, Any]) -> bool:
    for _node_id, node in iter_nodes(workflow):
        if node.get("class_type") not in _LORA_CLASSES:
            continue
        name = str((node.get("inputs") or {}).get("lora_name") or "").lower()
        if "sulphur" in name:
            return True
    return False


def _save_prefix_mentions(workflow: dict[str, Any], word: str) -> bool:
    for _node_id, node in iter_nodes(workflow):
        prefix = str((node.get("inputs") or {}).get("filename_prefix") or "")
        if word in prefix.lower():
            return True
    return False


def match_family(workflow: dict[str, Any]) -> str | None:
    """Return ``inoutpaint``, ``lipsync``, ``sulphur``, or None.

    inoutpaint wins when ``LTXVInpaintPreprocess`` is present so that family
    is not mistaken for lipsync or sulphur.

    lipsync needs an audio encode plus a video guide: IC-LoRA guides alone
    also appear in LTX 2.5 MSR / V2V. sulphur needs a sulphur LoRA (or a
    sulphur-only node plus a ``sulphur`` save prefix): sage attention and
    the sampling preview override also appear in Movie Builder and other
    LTX 2.3 graphs.
    """
    classes = set(class_types(workflow))
    if "LTXVInpaintPreprocess" in classes:
        return "inoutpaint"
    if "LTXVAudioVAEEncode" in classes and classes & _VIDEO_GUIDES:
        return "lipsync"
    if _has_sulphur_lora(workflow):
        return "sulphur"
    if classes & _SULPHUR_HINT_NODES and _save_prefix_mentions(workflow, "sulphur"):
        return "sulphur"
    return None


def is_ltx25_inoutpaint(workflow: dict[str, Any]) -> bool:
    """2.5 in/outpaint is the graph PR #44 patches (mask repeat + trim flag)."""
    classes = set(class_types(workflow))
    if classes & {"VHS_DuplicateMasks", "LTXVImgToVideoInplace", "LTXVImgToVideoConditionOnly"}:
        return True
    for _node_id, node in iter_nodes(workflow):
        if node.get("class_type") != "SaveVideo":
            continue
        prefix = str((node.get("inputs") or {}).get("filename_prefix") or "")
        if "ltx25" in prefix.lower():
            return True
    return False


def family_warning(family: str | None, *, route: str) -> str:
    if route == "generic" and family:
        return (
            f"--no-family-route: matched {family} stays on the generic path"
        )
    if not family:
        return "unmatched graph stays on the generic path"
    return ""


def decide_route(
    family: str | None,
    *,
    no_family_route: bool,
    stored_route: str | None = None,
) -> str:
    """``specialized`` or ``generic``.

    An explicit ``no_family_route`` wins. Otherwise a stored ``generic`` from
    ingest (the flag, or an unmatched graph) sticks until ``learn`` rewrites
    it. A stored ``specialized`` routes when the graph still matches.
    """
    if no_family_route or stored_route == "generic":
        return "generic"
    if family and stored_route in {None, "specialized"}:
        return "specialized"
    return "generic"


def _latent_int(workflow: dict[str, Any], key: str) -> int | None:
    for _node_id, node in iter_nodes(workflow):
        if node.get("class_type") != "EmptyLTXVLatentVideo":
            continue
        raw = (node.get("inputs") or {}).get(key)
        if isinstance(raw, int) and raw > 0:
            return int(raw)
    return None


def apply_family_route(
    workflow: dict[str, Any],
    family: str | None,
    *,
    values: dict[str, Any],
) -> list[str]:
    """Mutate ``workflow`` with the existing specialized helper. No new patcher."""
    if family == "inoutpaint":
        from master_agent.comfy.inoutpaint import (
            DEFAULT_HEIGHT,
            DEFAULT_WIDTH,
            finalize_inoutpaint_graph,
        )

        width = int(values.get("width") or _latent_int(workflow, "width") or DEFAULT_WIDTH)
        height = int(values.get("height") or _latent_int(workflow, "height") or DEFAULT_HEIGHT)
        finalize_inoutpaint_graph(
            workflow,
            width=width,
            height=height,
            ltx25=is_ltx25_inoutpaint(workflow),
        )
        return ["family inoutpaint: finalize_inoutpaint_graph"]
    if family == "sulphur":
        from master_agent.comfy.sulphur import patch_sulphur_graph

        patch_sulphur_graph(
            workflow,
            prompt=values.get("prompt"),
            negative_prompt=values.get("negative_prompt"),
            width=values.get("width"),
            height=values.get("height"),
            frames=values.get("frames"),
        )
        return ["family sulphur: patch_sulphur_graph"]
    if family == "lipsync":
        return [
            "family lipsync: prompt stays on the source widget; "
            "prepare_queue_inputs runs at queue"
        ]
    return []
