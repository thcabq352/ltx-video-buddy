"""Last-frame keyframe for pause-reset lipdub. No new node packs.

The shipped ``ltx25_a2v`` graph has one ``LoadImage`` into both
``LTXVImgToVideoInplace`` stages. ``LTXVAddGuide`` is a core
ComfyUI-LTXVideo node (also in Comfy core ``nodes_lt``). It refuses a
combined audio-video latent, so the guide is inserted on the video latent
after ``LTXVImgToVideoInplace`` and before ``LTXVConcatAVLatent``.
``LTXVCropGuides`` then drops the appended guide frames after
``LTXVSeparateAVLatent``, before the upscaler and the decode.

``frame_idx=-1`` is the last pixel frame (negative indexes count from the
end; a one-frame still is a legal guide). If either node is missing, or the
graph has no place to wire them, the caller falls back to re-anchoring the
pause on the source still and crossfading inside the silence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

GUIDE_CLASS = "LTXVAddGuide"
CROP_CLASS = "LTXVCropGuides"
GUIDE_CLASSES = (GUIDE_CLASS, CROP_CLASS)

# Shared with doctor and dry-run so the report is one sentence, not two dialects.
GUIDE_MISSING_MESSAGE = (
    "LTXVAddGuide and LTXVCropGuides are not in this Comfy's object_info. "
    "They are standard ComfyUI-LTXVideo / core LTX nodes, not a new download. "
    "pause-reset cannot pin the source still as a last-frame keyframe, so each "
    "pause is re-anchored on the source still with a short crossfade kept inside "
    "the silence and never across speech."
)
GUIDE_INFEASIBLE_MESSAGE = (
    "LTXVAddGuide is registered, but this a2v graph has no video-latent slot "
    "where a last-frame keyframe can be added before the audio concat and cropped "
    "off after the split. pause-reset falls back to re-anchoring each pause on the "
    "source still with a short crossfade inside the silence, never across speech."
)
GUIDE_READY_MESSAGE = (
    "pause-reset: LTXVAddGuide and LTXVCropGuides are registered. "
    "Each silence piece keeps the previous frame as frame 0 and pins the source "
    "still on the last frame (LTXVAddGuide frame_idx -1, then LTXVCropGuides). "
    "No new nodes are downloaded."
)


@dataclass
class GuidePatch:
    status: str
    message: str
    guide_ids: list[str] = field(default_factory=list)
    crop_ids: list[str] = field(default_factory=list)
    image_node: Optional[str] = None

    @property
    def applied(self) -> bool:
        return self.status == "applied"


def guide_nodes_present(object_info: Optional[dict]) -> bool:
    if not isinstance(object_info, dict):
        return False
    return all(name in object_info for name in GUIDE_CLASSES)


def missing_guide_classes(object_info: Optional[dict]) -> list[str]:
    info = object_info if isinstance(object_info, dict) else {}
    return [name for name in GUIDE_CLASSES if name not in info]


def _is_link(value: Any, node_id: str, out_index: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and str(value[0]) == str(node_id)
        and int(value[1]) == int(out_index)
    )


def _links_to(workflow: dict, node_id: str, out_index: int) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if _is_link(value, node_id, out_index):
                hits.append((str(nid), str(key)))
    return hits


def _node(workflow: dict, node_id: str) -> Optional[dict]:
    node = workflow.get(str(node_id))
    return node if isinstance(node, dict) else None


def _copy_link(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    return value


def apply_last_frame_guide(
    workflow: dict,
    *,
    image_name: str,
    object_info: Optional[dict],
    frame_idx: int = -1,
    strength: float = 1.0,
) -> tuple[dict, GuidePatch]:
    """Return a workflow with a source-still keyframe on the last frame.

    The input workflow is not mutated. When the guide nodes are absent or the
    graph cannot take the keyframe, the original workflow is returned with a
    status the caller can show in doctor / preflight.
    """
    if not guide_nodes_present(object_info):
        missing = ", ".join(missing_guide_classes(object_info)) or ", ".join(GUIDE_CLASSES)
        return workflow, GuidePatch(
            status="missing",
            message=GUIDE_MISSING_MESSAGE + f" Missing: {missing}.",
        )
    try:
        return _wire_guide(
            workflow,
            image_name=str(image_name or "still.png"),
            frame_idx=int(frame_idx),
            strength=float(strength),
        )
    except Exception as exc:
        return workflow, GuidePatch(
            status="infeasible",
            message=f"{GUIDE_INFEASIBLE_MESSAGE} ({exc})",
        )


def _wire_guide(
    workflow: dict,
    *,
    image_name: str,
    frame_idx: int,
    strength: float,
) -> tuple[dict, GuidePatch]:
    wf = copy.deepcopy(workflow)
    inplaces = [
        nid
        for nid, node in wf.items()
        if isinstance(node, dict) and node.get("class_type") == "LTXVImgToVideoInplace"
    ]
    if not inplaces:
        return workflow, GuidePatch(status="infeasible", message=GUIDE_INFEASIBLE_MESSAGE)

    image_id = "vb_end_still"
    wf[image_id] = {
        "inputs": {"image": image_name},
        "class_type": "LoadImage",
        "_meta": {"title": "Pause-reset source still"},
    }
    guide_ids: list[str] = []
    crop_ids: list[str] = []
    for index, inplace_id in enumerate(inplaces):
        wired = _wire_stage(
            wf,
            inplace_id=str(inplace_id),
            image_id=image_id,
            index=index,
            frame_idx=frame_idx,
            strength=strength,
        )
        if wired is None:
            continue
        guide_ids.append(wired[0])
        crop_ids.append(wired[1])
    if not guide_ids:
        return workflow, GuidePatch(status="infeasible", message=GUIDE_INFEASIBLE_MESSAGE)
    return wf, GuidePatch(
        status="applied",
        message=GUIDE_READY_MESSAGE,
        guide_ids=guide_ids,
        crop_ids=crop_ids,
        image_node=image_id,
    )


def _wire_stage(
    wf: dict,
    *,
    inplace_id: str,
    image_id: str,
    index: int,
    frame_idx: int,
    strength: float,
) -> Optional[tuple[str, str]]:
    inplace = _node(wf, inplace_id)
    if inplace is None:
        return None
    vae = _copy_link((inplace.get("inputs") or {}).get("vae"))
    if not isinstance(vae, list):
        return None
    concat_id = None
    for nid, key in _links_to(wf, inplace_id, 0):
        node = _node(wf, nid)
        if node and node.get("class_type") == "LTXVConcatAVLatent" and key == "video_latent":
            concat_id = nid
            break
    if concat_id is None:
        return None
    sampler_id = None
    for nid, key in _links_to(wf, concat_id, 0):
        node = _node(wf, nid)
        if node and node.get("class_type") == "SamplerCustomAdvanced" and key == "latent_image":
            sampler_id = nid
            break
    if sampler_id is None:
        return None
    sampler = _node(wf, sampler_id)
    guider_link = (sampler.get("inputs") or {}).get("guider") if sampler else None
    if not isinstance(guider_link, list) or not guider_link:
        return None
    guider_id = str(guider_link[0])
    guider = _node(wf, guider_id)
    if guider is None or guider.get("class_type") != "CFGGuider":
        return None
    positive = _copy_link((guider.get("inputs") or {}).get("positive"))
    negative = _copy_link((guider.get("inputs") or {}).get("negative"))
    if positive is None or negative is None:
        return None
    separate_id = None
    for nid, key in _links_to(wf, sampler_id, 0):
        node = _node(wf, nid)
        if node and node.get("class_type") == "LTXVSeparateAVLatent" and key == "av_latent":
            separate_id = nid
            break
    if separate_id is None:
        return None
    consumers = _links_to(wf, separate_id, 0)
    if not consumers:
        return None

    guide_id = f"vb_add_guide_{index}"
    crop_id = f"vb_crop_guides_{index}"
    wf[guide_id] = {
        "inputs": {
            "positive": positive,
            "negative": negative,
            "vae": vae,
            "latent": [inplace_id, 0],
            "image": [image_id, 0],
            "frame_idx": int(frame_idx),
            "strength": float(strength),
        },
        "class_type": GUIDE_CLASS,
        "_meta": {"title": "Pause-reset last-frame guide"},
    }
    wf[crop_id] = {
        "inputs": {
            "positive": [guide_id, 0],
            "negative": [guide_id, 1],
            "latent": [separate_id, 0],
        },
        "class_type": CROP_CLASS,
        "_meta": {"title": "Pause-reset crop guides"},
    }
    concat = _node(wf, concat_id)
    concat["inputs"]["video_latent"] = [guide_id, 2]
    guider["inputs"]["positive"] = [guide_id, 0]
    guider["inputs"]["negative"] = [guide_id, 1]
    for nid, key in consumers:
        if nid in (guide_id, crop_id):
            continue
        node = _node(wf, nid)
        if node is None:
            continue
        node["inputs"][key] = [crop_id, 2]
    return guide_id, crop_id


def probe_pause_reset_guide(
    object_info: Optional[dict],
    *,
    variant: str = "ltx25_a2v",
) -> tuple[bool, str]:
    """True when a last-frame keyframe can be patched onto the shipped a2v graph."""
    if not guide_nodes_present(object_info):
        missing = ", ".join(missing_guide_classes(object_info)) or ", ".join(GUIDE_CLASSES)
        return False, GUIDE_MISSING_MESSAGE + f" Missing: {missing}."
    from master_agent.comfy.workflow_patcher import load_workflow_template

    workflow = load_workflow_template(variant)
    _patched, patch = apply_last_frame_guide(
        workflow,
        image_name="still.png",
        object_info=object_info,
    )
    if not patch.applied:
        return False, patch.message or GUIDE_INFEASIBLE_MESSAGE
    return True, GUIDE_READY_MESSAGE
