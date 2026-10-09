"""Role heuristics for unbound widgets. No LLM. No network.

Linked widgets are not tunable. A CLIPTextEncode whose ``text`` is a link is
skipped in favor of the Primitive (or other source) widget, same lesson as
lipsync and LTX 2.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Stand-in for Comfy ``/object_info`` when the tower is offline. A class in
# neither this set nor a provided object_info fails closed. This list is not
# an installer and does not download packs.
OFFLINE_NODE_CATALOG = frozenset(
    {
        "CheckpointLoaderSimple",
        "CLIPLoader",
        "CLIPTextEncode",
        "EmptyLTXVLatentVideo",
        "EmptyLatentImage",
        "EmptySD3LatentImage",
        "CreateVideo",
        "GuiderParameters",
        "KSampler",
        "KSamplerAdvanced",
        "KSamplerSelect",
        "LoadAudio",
        "LoadImage",
        "LoadVideo",
        "LoraLoader",
        "LoraLoaderModelOnly",
        "LTXAVTextEncoderLoader",
        "LTXVAudioVAEDecode",
        "LTXVAudioVAEEncode",
        "LTXVAudioVAELoader",
        "LTXVConcatAVLatent",
        "LTXVConditioning",
        "LTXVEmptyLatentAudio",
        "LTXVDilateVideoMask",
        "LTXVImgToVideoConditionOnly",
        "LTXVImgToVideoInplace",
        "LTXVInpaintPreprocess",
        "LTXVLaplacianPyramidBlend",
        "LTXVScheduler",
        "LTXVSetAudioRefTokens",
        "LTXAddVideoICLoRAGuide",
        "LTX2SamplingPreviewOverride",
        "ImagePadForOutpaint",
        "PathchSageAttentionKJ",
        "RepeatImageBatch",
        "LTXVSeparateAVLatent",
        "LTXVTiledVAEDecode",
        "MultimodalGuider",
        "Note",
        "PreviewImage",
        "PrimitiveBoolean",
        "PrimitiveFloat",
        "PrimitiveInt",
        "PrimitiveString",
        "PrimitiveStringMultiline",
        "RandomNoise",
        "SamplerCustomAdvanced",
        "SaveImage",
        "SaveVideo",
        "UNETLoader",
        "VAEDecode",
        "VAEDecodeTiled",
        "VAELoader",
        "VHS_DuplicateMasks",
        "VHS_VideoCombine",
    }
)

_PACKS = {
    "EmptyLTXVLatentVideo": "ComfyUI-LTXVideo",
    "GuiderParameters": "ComfyUI-LTXVideo",
    "LTXAVTextEncoderLoader": "ComfyUI-LTXVideo",
    "LTXVAudioVAEDecode": "ComfyUI-LTXVideo",
    "LTXVAudioVAEEncode": "ComfyUI-LTXVideo",
    "LTXVAudioVAELoader": "ComfyUI-LTXVideo",
    "LTXVConcatAVLatent": "ComfyUI-LTXVideo",
    "LTXVConditioning": "ComfyUI-LTXVideo",
    "LTXVEmptyLatentAudio": "ComfyUI-LTXVideo",
    "LTXVInpaintPreprocess": "ComfyUI-LTXVideo",
    "LTXVScheduler": "ComfyUI-LTXVideo",
    "LTXVSeparateAVLatent": "ComfyUI-LTXVideo",
    "LTXVTiledVAEDecode": "ComfyUI-LTXVideo",
    "MultimodalGuider": "ComfyUI-LTXVideo",
    "LTXAddVideoICLoRAGuide": "ComfyUI-LTXVideo",
    "LTX2SamplingPreviewOverride": "ComfyUI-LTXVideo",
    "VHS_VideoCombine": "VideoHelperSuite",
    "VHS_DuplicateMasks": "VideoHelperSuite",
    "VHS_LoadVideo": "VideoHelperSuite",
    "PathchSageAttentionKJ": "ComfyUI-KJNodes",
    "ResizeImageMaskNode": "ComfyUI-KJNodes",
    "ComfyMathExpression": "ComfyMath",
}

_LATENT_HIGH = frozenset(
    {"EmptyLTXVLatentVideo", "EmptyLatentImage", "EmptySD3LatentImage"}
)
_SEED_FALLBACK = frozenset({"KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"})
_SAVE_CLASSES = ("SaveVideo", "SaveImage", "VHS_VideoCombine")
_PRIMITIVE_MARKERS = ("Primitive",)


@dataclass
class RoleGuess:
    role: str
    node_id: str
    input: str
    class_type: str
    index: int
    confidence: str
    detail: str


def iter_nodes(workflow: dict[str, Any]):
    for node_id, node in workflow.items():
        if isinstance(node, dict) and isinstance(node.get("class_type"), str):
            yield str(node_id), node


def is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def class_types(workflow: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for _nid, node in iter_nodes(workflow):
        name = node.get("class_type") or ""
        if name and name not in seen:
            seen.append(name)
    return seen


def missing_class_types(
    workflow: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> list[str]:
    """Class types absent from ``object_info`` or, offline, the bundled catalog."""
    if object_info is None:
        present = OFFLINE_NODE_CATALOG
    else:
        present = set(object_info)
    missing = [name for name in class_types(workflow) if name not in present]
    return sorted(missing)


def pack_from_python_module(module: str) -> str | None:
    """Best-effort pack folder from an object_info ``python_module`` string.

    ``custom_nodes.ComfyUI-LTXVideo.nodes`` → ``ComfyUI-LTXVideo``.
    Core ``nodes`` / ``comfy`` modules are not packs.
    """
    text = str(module or "").strip()
    if not text:
        return None
    parts = [part for part in text.split(".") if part]
    if "custom_nodes" in parts:
        index = parts.index("custom_nodes")
        if index + 1 < len(parts):
            return parts[index + 1]
        return None
    if parts[0] in {"nodes", "comfy", "comfy_extras", "execution", "folder_paths"}:
        return None
    return None


def pack_from_known_map(class_type: str) -> str | None:
    """Known class → pack. ``VHS_*`` is Video Helper Suite."""
    name = str(class_type or "")
    if name in _PACKS:
        return _PACKS[name]
    if name.startswith("VHS_"):
        return "VideoHelperSuite"
    return None


def pack_for_class(
    class_type: str,
    object_info: dict[str, Any] | None = None,
) -> tuple[str | None, str]:
    """Return ``(pack, source)`` where source is python_module, known_map, or unknown.

    object_info wins when that class entry has ``python_module``. Otherwise
    the known map. A missing class is usually absent from object_info, so the
    known map is what names it.
    """
    info = (object_info or {}).get(class_type)
    if isinstance(info, dict):
        module = info.get("python_module")
        if isinstance(module, str) and module.strip():
            pack = pack_from_python_module(module)
            if pack:
                return pack, "python_module"
    known = pack_from_known_map(class_type)
    if known:
        return known, "known_map"
    return None, "unknown"


def missing_node_packs(
    workflow: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """One row per missing class, with a best-effort pack name."""
    rows: list[dict[str, str]] = []
    for name in missing_class_types(workflow, object_info):
        pack, source = pack_for_class(name, object_info)
        row = {"class_type": name, "source": source}
        if pack:
            row["pack"] = pack
        rows.append(row)
    return rows


def requires_packs(
    workflow: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> list[str]:
    packs: list[str] = []
    for name in class_types(workflow):
        pack, _source = pack_for_class(name, object_info)
        if pack and pack not in packs:
            packs.append(pack)
    return packs


def _title(node: dict[str, Any]) -> str:
    meta = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
    return str((meta or {}).get("title") or "")


def _polarity(title: str) -> str | None:
    text = title.lower()
    if "negative" in text:
        return "negative"
    if "positive" in text or "prompt" in text:
        return "positive"
    return None


def _class_index(workflow: dict[str, Any], node_id: str, class_type: str) -> int:
    index = 0
    for nid, node in iter_nodes(workflow):
        if node.get("class_type") != class_type:
            continue
        if nid == str(node_id):
            return index
        index += 1
    return 0


def _is_primitive(node: dict[str, Any]) -> bool:
    name = str(node.get("class_type") or "")
    return any(marker in name for marker in _PRIMITIVE_MARKERS)


def _follow_source_widget(
    workflow: dict[str, Any], link: list[Any]
) -> tuple[str, dict[str, Any], str] | None:
    source_id = str(link[0])
    source = workflow.get(source_id)
    if not isinstance(source, dict):
        return None
    inputs = source.get("inputs") or {}
    for key in ("value", "text", "string"):
        if key not in inputs or is_link(inputs.get(key)):
            continue
        if _is_primitive(source) or isinstance(inputs.get(key), str):
            return source_id, source, key
    return None


def _text_candidates(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for node_id, node in iter_nodes(workflow):
        if node.get("class_type") != "CLIPTextEncode":
            continue
        inputs = node.get("inputs") or {}
        raw = inputs.get("text")
        title = _title(node)
        if is_link(raw):
            followed = _follow_source_widget(workflow, raw)
            if followed is None:
                continue
            src_id, src, key = followed
            title = f"{title} {_title(src)}".strip()
            target_id, target, input_name = src_id, src, key
        elif isinstance(raw, str):
            target_id, target, input_name = node_id, node, "text"
        else:
            continue
        found.append(
            {
                "node_id": target_id,
                "input": input_name,
                "class_type": str(target.get("class_type") or ""),
                "title": title,
                "polarity": _polarity(title),
            }
        )
    return found


def _guess_from_candidate(
    workflow: dict[str, Any],
    role: str,
    candidate: dict[str, Any],
    *,
    confidence: str,
    detail: str,
) -> RoleGuess:
    return RoleGuess(
        role=role,
        node_id=str(candidate["node_id"]),
        input=str(candidate["input"]),
        class_type=str(candidate["class_type"]),
        index=_class_index(workflow, candidate["node_id"], candidate["class_type"]),
        confidence=confidence,
        detail=detail,
    )


def _guess_widget(
    workflow: dict[str, Any],
    role: str,
    node_id: str,
    node: dict[str, Any],
    input_name: str,
    *,
    confidence: str,
    detail: str,
) -> RoleGuess:
    class_type = str(node.get("class_type") or "")
    return RoleGuess(
        role=role,
        node_id=str(node_id),
        input=input_name,
        class_type=class_type,
        index=_class_index(workflow, node_id, class_type),
        confidence=confidence,
        detail=detail,
    )


def _infer_prompts(workflow: dict[str, Any]) -> list[RoleGuess]:
    candidates = _text_candidates(workflow)
    if not candidates:
        return []
    guesses: list[RoleGuess] = []
    used: set[tuple[str, str]] = set()
    positives = [item for item in candidates if item["polarity"] == "positive"]
    negatives = [item for item in candidates if item["polarity"] == "negative"]
    if positives:
        prompt = positives[0]
        confidence = "high"
        detail = "positive/prompt title; linked CLIP widgets stay linked"
    else:
        pool = [item for item in candidates if item["polarity"] != "negative"] or candidates
        prompt = pool[0]
        confidence = "high" if len(candidates) == 1 else "low"
        detail = "no positive title; best guess is the first unbound text widget"
    guesses.append(
        _guess_from_candidate(
            workflow, "prompt", prompt, confidence=confidence, detail=detail
        )
    )
    used.add((prompt["node_id"], prompt["input"]))
    negative = next(
        (item for item in negatives if (item["node_id"], item["input"]) not in used),
        None,
    )
    if negative is not None:
        guesses.append(
            _guess_from_candidate(
                workflow,
                "negative_prompt",
                negative,
                confidence="high",
                detail="negative title",
            )
        )
    else:
        rest = [item for item in candidates if (item["node_id"], item["input"]) not in used]
        if rest:
            guesses.append(
                _guess_from_candidate(
                    workflow,
                    "negative_prompt",
                    rest[0],
                    confidence="low",
                    detail="no negative title; best guess is the next unbound text widget",
                )
            )
    return guesses


def _first_unbound(
    workflow: dict[str, Any], class_types: frozenset[str] | tuple[str, ...], key: str
) -> tuple[str, dict[str, Any]] | None:
    matches: list[tuple[str, dict[str, Any]]] = []
    for node_id, node in iter_nodes(workflow):
        if node.get("class_type") not in class_types:
            continue
        inputs = node.get("inputs") or {}
        if key in inputs and not is_link(inputs.get(key)):
            matches.append((node_id, node))
    if not matches:
        return None
    return matches[0]


def _infer_seed(workflow: dict[str, Any]) -> RoleGuess | None:
    noise = _first_unbound(workflow, frozenset({"RandomNoise"}), "noise_seed")
    if noise is not None:
        node_id, node = noise
        return _guess_widget(
            workflow,
            "seed",
            node_id,
            node,
            "noise_seed",
            confidence="high",
            detail="RandomNoise.noise_seed",
        )
    sampler = _first_unbound(workflow, _SEED_FALLBACK, "seed")
    if sampler is None:
        return None
    node_id, node = sampler
    count = sum(
        1
        for _nid, node in iter_nodes(workflow)
        if node.get("class_type") in _SEED_FALLBACK
        and not is_link((node.get("inputs") or {}).get("seed"))
    )
    return _guess_widget(
        workflow,
        "seed",
        node_id,
        node,
        "seed",
        confidence="high" if count == 1 else "low",
        detail="KSampler seed; no RandomNoise widget",
    )


def _latent_nodes(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    found: list[tuple[str, dict[str, Any], str]] = []
    for node_id, node in iter_nodes(workflow):
        name = str(node.get("class_type") or "")
        if name in _LATENT_HIGH:
            found.append((node_id, node, "high"))
        elif name.startswith("Empty") and "Latent" in name:
            found.append((node_id, node, "low"))
    return found


def _infer_size(workflow: dict[str, Any]) -> list[RoleGuess]:
    nodes = _latent_nodes(workflow)
    if not nodes:
        return []
    nodes.sort(key=lambda item: 0 if item[2] == "high" else 1)
    node_id, node, confidence = nodes[0]
    if len(nodes) > 1 and confidence == "high":
        confidence = "low"
    inputs = node.get("inputs") or {}
    guesses: list[RoleGuess] = []
    for role, key in (("width", "width"), ("height", "height")):
        if key in inputs and not is_link(inputs.get(key)):
            guesses.append(
                _guess_widget(
                    workflow,
                    role,
                    node_id,
                    node,
                    key,
                    confidence=confidence,
                    detail=f"{node.get('class_type')}.{key}",
                )
            )
    frame_key = "length" if "length" in inputs else "frames" if "frames" in inputs else ""
    if frame_key and not is_link(inputs.get(frame_key)):
        guesses.append(
            _guess_widget(
                workflow,
                "frames",
                node_id,
                node,
                frame_key,
                confidence=confidence,
                detail=f"{node.get('class_type')}.{frame_key}",
            )
        )
    return guesses


def _infer_checkpoint(workflow: dict[str, Any]) -> RoleGuess | None:
    found = _first_unbound(workflow, frozenset({"CheckpointLoaderSimple"}), "ckpt_name")
    if found is None:
        return None
    node_id, node = found
    return _guess_widget(
        workflow,
        "checkpoint",
        node_id,
        node,
        "ckpt_name",
        confidence="high",
        detail="CheckpointLoaderSimple.ckpt_name",
    )


def _infer_vae(workflow: dict[str, Any]) -> RoleGuess | None:
    found = _first_unbound(workflow, frozenset({"VAELoader"}), "vae_name")
    if found is None:
        return None
    node_id, node = found
    return _guess_widget(
        workflow,
        "vae",
        node_id,
        node,
        "vae_name",
        confidence="high",
        detail="VAELoader.vae_name",
    )


def _infer_filename_prefix(workflow: dict[str, Any]) -> tuple[RoleGuess | None, str | None]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for class_type in _SAVE_CLASSES:
        for node_id, node in iter_nodes(workflow):
            if node.get("class_type") != class_type:
                continue
            inputs = node.get("inputs") or {}
            if "filename_prefix" in inputs and not is_link(inputs.get("filename_prefix")):
                matches.append((node_id, node))
    if not matches:
        save_ids = [
            node_id
            for node_id, node in iter_nodes(workflow)
            if str(node.get("class_type") or "").startswith("Save")
        ]
        return None, save_ids[-1] if save_ids else None
    # Prefer the last SaveVideo, else the last save node that has the widget.
    videos = [item for item in matches if item[1].get("class_type") == "SaveVideo"]
    node_id, node = (videos or matches)[-1]
    guess = _guess_widget(
        workflow,
        "filename_prefix",
        node_id,
        node,
        "filename_prefix",
        confidence="high",
        detail=f"{node.get('class_type')}.filename_prefix",
    )
    return guess, node_id


def infer_roles(workflow: dict[str, Any]) -> tuple[list[RoleGuess], str | None]:
    """Return role guesses and the output node id (last SaveVideo, else last Save*)."""
    guesses: list[RoleGuess] = []
    guesses.extend(_infer_prompts(workflow))
    seed = _infer_seed(workflow)
    if seed is not None:
        guesses.append(seed)
    guesses.extend(_infer_size(workflow))
    checkpoint = _infer_checkpoint(workflow)
    if checkpoint is not None:
        guesses.append(checkpoint)
    vae = _infer_vae(workflow)
    if vae is not None:
        guesses.append(vae)
    prefix, output_id = _infer_filename_prefix(workflow)
    if prefix is not None:
        guesses.append(prefix)
    return guesses, output_id
