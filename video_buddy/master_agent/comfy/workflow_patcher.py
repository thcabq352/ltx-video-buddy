"""Load workflow JSON templates and inject generation parameters."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Optional

import yaml

from master_agent.config import (
    DEFAULT_CFG,
    DEFAULT_STEPS,
    MODELS_DIR,
    MODEL_FILES,
    WORKFLOWS_DIR,
    clamp_resolution,
    frames_for_duration,
    get_variant_gen,
    resolve_model_path,
    snap_ltx_frames,
)

# LTX latent widgets that must stay 8n+1 and paired with audio frames_number
LTX_LENGTH_CLASSES = frozenset({"EmptyLTXVLatentVideo", "LTXVEmptyLatentVideo"})
LTX_AUDIO_CLASSES = frozenset({"LTXVEmptyLatentAudio"})

# Preferred all-in-one checkpoint names (first that exists on disk wins as fallback)
CHECKPOINT_FALLBACKS = [
    "LTX2.3_DISTILLED-1.1_BAKED_LTX_10Eros_v14_r768.safetensors",
    "ltx-2.3-22b-distilled-10-eros_fp8.safetensors",
    "ltx-2.3-22b-dev.safetensors",  # optional official full model if user installs later
    "ltx-2.3-22b-distilled-1.1.safetensors",
]

# Official GuiderParameters defaults (AUDIO then VIDEO chain)
GUIDER_AUDIO_DEFAULTS = {
    "modality": "AUDIO",
    "cfg": 7.0,
    "stg": 1.0,
    "perturb_attn": True,
    "rescale": 0.7,
    "modality_scale": 3.0,
    "skip_step": 0,
    "cross_attn": True,
}
GUIDER_VIDEO_DEFAULTS = {
    "modality": "VIDEO",
    "cfg": 3.0,
    "stg": 1.0,
    "perturb_attn": True,
    "rescale": 0.9,
    "modality_scale": 3.0,
    "skip_step": 0,
    "cross_attn": True,
}
TILED_VAE_DEFAULTS = {
    "horizontal_tiles": 2,
    "vertical_tiles": 2,
    "overlap": 2,
    "last_frame_fix": False,
    "working_device": "auto",
    "working_dtype": "auto",
}


def _load_manifests() -> dict[str, Any]:
    path = WORKFLOWS_DIR / "manifests.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_workflow_template(variant: str) -> dict[str, Any]:
    manifests = _load_manifests()
    meta = manifests.get(variant) or {}
    filename = meta.get("file") or f"{variant}.json"
    path = WORKFLOWS_DIR / filename
    if not path.is_file():
        for cand in (
            WORKFLOWS_DIR / f"{variant}.json",
            WORKFLOWS_DIR / "base_t2v_i2v.json",
        ):
            if cand.is_file():
                path = cand
                break
        else:
            raise FileNotFoundError(
                f"Workflow template not found for variant={variant}: {path}"
            )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "prompt" in data and isinstance(data["prompt"], dict):
        return data["prompt"]
    return data


def resolve_checkpoint_name(preferred: Optional[str] = None) -> str:
    """Pick a checkpoint that exists under models/ (preferred, then fallbacks)."""
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(CHECKPOINT_FALLBACKS)
    # Scan checkpoints folder
    ckpt_dir = MODELS_DIR / "checkpoints"
    if ckpt_dir.is_dir():
        for p in sorted(ckpt_dir.glob("*.safetensors")):
            candidates.append(p.name)

    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        if resolve_model_path(name) is not None:
            return name
    # Last resort: return preferred or first fallback even if missing (Comfy will error clearly)
    return preferred or CHECKPOINT_FALLBACKS[0]


def _find_nodes_by_class(workflow: dict[str, Any], class_type: str) -> list[tuple[str, dict]]:
    out = []
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            out.append((nid, node))
    return out


def _set_input(node: dict[str, Any], key: str, value: Any) -> bool:
    if "inputs" not in node or not isinstance(node["inputs"], dict):
        node["inputs"] = {}
    node["inputs"][key] = value
    return True


def _apply_named_fields(
    workflow: dict[str, Any],
    field_map: dict[str, Any],
    values: dict[str, Any],
) -> list[str]:
    applied: list[str] = []
    for logical, spec in field_map.items():
        if logical not in values or values[logical] is None:
            continue
        value = values[logical]
        if not isinstance(spec, dict):
            continue
        if "node_id" in spec:
            nids = spec["node_id"]
            if not isinstance(nids, list):
                nids = [nids]
            key = spec.get("input") or spec.get("key")
            if key:
                for nid in nids:
                    nid = str(nid)
                    if nid in workflow:
                        _set_input(workflow[nid], key, value)
                        applied.append(logical)
        elif "class_type" in spec:
            matches = _find_nodes_by_class(workflow, spec["class_type"])
            idx = int(spec.get("index", 0))
            if matches:
                nid, node = matches[min(idx, len(matches) - 1)]
                key = spec.get("input") or spec.get("key")
                if key:
                    _set_input(node, key, value)
                    applied.append(logical)
    return applied


def _sanitize_ltx_nodes(workflow: dict[str, Any]) -> None:
    """Ensure LTX custom nodes keep required widget fields after any patching."""
    guider_nodes = _find_nodes_by_class(workflow, "GuiderParameters")
    for i, (_nid, node) in enumerate(guider_nodes):
        defaults = GUIDER_AUDIO_DEFAULTS if i == 0 else GUIDER_VIDEO_DEFAULTS
        # Prefer modality already set
        mod = (node.get("inputs") or {}).get("modality")
        if mod == "AUDIO":
            defaults = GUIDER_AUDIO_DEFAULTS
        elif mod == "VIDEO":
            defaults = GUIDER_VIDEO_DEFAULTS
        inputs = node.setdefault("inputs", {})
        # Preserve linked inputs (e.g. parameters chain)
        links = {k: v for k, v in inputs.items() if isinstance(v, list)}
        merged = {**defaults, **{k: v for k, v in inputs.items() if not isinstance(v, list)}}
        # Re-apply defaults for any missing required keys
        for k, v in defaults.items():
            if k not in merged or merged[k] is None:
                merged[k] = v
        node["inputs"] = {**merged, **links}

    for _nid, node in _find_nodes_by_class(workflow, "MultimodalGuider"):
        inputs = node.setdefault("inputs", {})
        # Empty skip_blocks is valid; avoid force-injecting "28" which can break models
        if "skip_blocks" not in inputs or inputs["skip_blocks"] is None:
            inputs["skip_blocks"] = ""
        if isinstance(inputs.get("cfg"), (int, float, str)) and not isinstance(
            inputs.get("cfg"), list
        ):
            inputs.pop("cfg", None)

    for _nid, node in _find_nodes_by_class(workflow, "LTXVTiledVAEDecode"):
        inputs = node.setdefault("inputs", {})
        links = {k: v for k, v in inputs.items() if isinstance(v, list)}
        scalars = {k: v for k, v in inputs.items() if not isinstance(v, list)}
        merged = {**TILED_VAE_DEFAULTS, **scalars}
        for k, v in TILED_VAE_DEFAULTS.items():
            if k not in merged or merged[k] is None:
                merged[k] = v
        # Drop wrong legacy widget keys
        for bad in (
            "tile_size",
            "temporal_size",
            "temporal_overlap",
            "frame_rate_mode",
            "memory_mode",
            "widget_0",
            "widget_1",
            "widget_2",
            "widget_3",
            "widget_4",
            "widget_5",
        ):
            merged.pop(bad, None)
        node["inputs"] = {**merged, **links}

    for _nid, node in _find_nodes_by_class(workflow, "RandomNoise"):
        inputs = node.setdefault("inputs", {})
        seed = inputs.get("noise_seed", inputs.get("seed", 0))
        if isinstance(seed, list):
            node["inputs"] = {"noise_seed": seed}
        else:
            try:
                node["inputs"] = {"noise_seed": int(seed)}
            except (TypeError, ValueError):
                node["inputs"] = {"noise_seed": 0}


def _ensure_lora_node(workflow: dict[str, Any], lora_name: str) -> Optional[str]:
    """
    Insert a LoraLoaderModelOnly between the model loader and its consumers.
    No-op when a lora loader already exists or no model loader is found.
    Returns the new node id, or None.
    """
    if not lora_name:
        return None
    for class_type in (
        "LoraLoader",
        "LoraLoaderModelOnly",
        "LTXICLoRALoaderModelOnly",
    ):
        if _find_nodes_by_class(workflow, class_type):
            return None
    src_id: Optional[str] = None
    for class_type in ("UNETLoader", "CheckpointLoaderSimple", "DiffusionModelLoader"):
        matches = _find_nodes_by_class(workflow, class_type)
        if matches:
            src_id = matches[0][0]
            break
    if src_id is None:
        return None
    numeric_ids = [int(k) for k in workflow if str(k).isdigit()]
    new_id = str(max(numeric_ids) + 1) if numeric_ids else "1"
    workflow[new_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": [src_id, 0],
            "lora_name": lora_name,
            "strength_model": 1.0,
        },
        "_meta": {"title": "LoRA"},
    }
    # Rewire consumers of the loader's MODEL output (index 0) through the lora
    for nid, node in workflow.items():
        if nid == new_id:
            continue
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            if (
                isinstance(value, list)
                and len(value) == 2
                and str(value[0]) == src_id
                and value[1] == 0
            ):
                inputs[key] = [new_id, 0]
    return new_id


def _heuristic_patch(workflow: dict[str, Any], values: dict[str, Any]) -> None:
    """Best-effort patching by common ComfyUI / LTX class types."""
    prompt = values.get("prompt")
    negative = values.get("negative_prompt")
    seed = values.get("seed")
    width = values.get("width")
    height = values.get("height")
    frames = values.get("frames")
    steps = values.get("steps")
    ckpt = values.get("checkpoint")
    lora = values.get("lora")
    clip_l = values.get("clip_l")
    t5xxl = values.get("t5xxl")
    vae_name = values.get("vae_name")
    image_name = values.get("image_name")
    audio_name = values.get("audio_name")
    filename_prefix = values.get("filename_prefix")
    text_encoder = values.get("text_encoder")
    # quality-correction fields also live on values

    # CLIP / text encode
    for _nid, node in _find_nodes_by_class(workflow, "CLIPTextEncode"):
        title = (node.get("_meta") or {}).get("title", "").lower()
        if negative is not None and "neg" in title:
            _set_input(node, "text", negative)
        elif prompt is not None and "neg" not in title:
            _set_input(node, "text", prompt)

    # Empty latent / size nodes — keep video length and audio frames_number in sync
    ltx_frames = snap_ltx_frames(int(frames)) if frames is not None else None
    for class_type in (
        "EmptyLTXVLatentVideo",
        "EmptyHunyuanLatentVideo",
        "EmptyLatentImage",
        "LTXVEmptyLatentVideo",
    ):
        for _nid, node in _find_nodes_by_class(workflow, class_type):
            if width is not None:
                _set_input(node, "width", width)
            if height is not None:
                _set_input(node, "height", height)
            # EmptyLatentImage (Flux t2i) has no length/frames input
            if frames is not None and class_type != "EmptyLatentImage":
                write = ltx_frames if class_type in LTX_LENGTH_CLASSES else frames
                for k in ("length", "frames", "num_frames", "frame_count"):
                    if k in (node.get("inputs") or {}) or k == "length":
                        _set_input(node, k, write)
                        break

    if frames is not None:
        paired = ltx_frames if ltx_frames is not None else int(frames)
        for class_type in LTX_AUDIO_CLASSES:
            for _nid, node in _find_nodes_by_class(workflow, class_type):
                _set_input(node, "frames_number", int(paired))
                if values.get("fps"):
                    _set_input(node, "frame_rate", int(values.get("fps") or 24))
        for _nid, node in _find_nodes_by_class(workflow, "LTXVConditioning"):
            if values.get("fps"):
                _set_input(node, "frame_rate", int(values.get("fps") or 24))
        for _nid, node in _find_nodes_by_class(workflow, "CreateVideo"):
            if values.get("fps"):
                _set_input(node, "fps", float(values.get("fps") or 24))
        # Sampler selection (quality correction may request euler_ancestral_cfg_pp)
        sampler_name = values.get("sampler_name")
        for _nid, node in _find_nodes_by_class(workflow, "KSamplerSelect"):
            if sampler_name:
                _set_input(node, "sampler_name", str(sampler_name))
            else:
                name = str((node.get("inputs") or {}).get("sampler_name") or "")
                if "cfg_pp" in name or not name:
                    _set_input(node, "sampler_name", "euler")

    # Seeds only on noise / sampler seed fields (never spray cfg onto RandomNoise)
    for class_type in ("RandomNoise",):
        for _nid, node in _find_nodes_by_class(workflow, class_type):
            if seed is not None:
                _set_input(node, "noise_seed", seed)

    for class_type in ("KSampler", "KSamplerAdvanced"):
        for _nid, node in _find_nodes_by_class(workflow, class_type):
            if seed is not None:
                inputs = node.get("inputs") or {}
                # KSampler uses 'seed', KSamplerAdvanced typically 'noise_seed'
                # (don't spray a bogus 'seed' key onto the latter)
                if "seed" in inputs or "noise_seed" not in inputs:
                    _set_input(node, "seed", seed)
                else:
                    _set_input(node, "noise_seed", seed)
            if steps is not None and "steps" in (node.get("inputs") or {}):
                _set_input(node, "steps", steps)
            if values.get("cfg") is not None and "cfg" in (node.get("inputs") or {}):
                _set_input(node, "cfg", values["cfg"])

    for _nid, node in _find_nodes_by_class(workflow, "LTXVScheduler"):
        if steps is not None:
            _set_input(node, "steps", steps)

    # STG / CFG on GuiderParameters (VIDEO modality gets primary STG)
    stg = values.get("stg_scale")
    stg_blocks = values.get("stg_blocks")
    cfg_val = values.get("cfg")
    if stg is not None or stg_blocks is not None or cfg_val is not None:
        for _nid, node in _find_nodes_by_class(workflow, "GuiderParameters"):
            inputs = node.get("inputs") or {}
            modality = str(inputs.get("modality") or "VIDEO").upper()
            if cfg_val is not None and modality == "VIDEO":
                _set_input(node, "cfg", float(cfg_val))
            if stg is not None and modality == "VIDEO":
                _set_input(node, "stg", float(stg))
            # skip_blocks live on MultimodalGuider as string; also encode STG blocks there
        if stg_blocks is not None:
            block_str = (
                ",".join(str(b) for b in stg_blocks)
                if isinstance(stg_blocks, (list, tuple))
                else str(stg_blocks)
            )
            for _nid, node in _find_nodes_by_class(workflow, "MultimodalGuider"):
                # Official tip: stg_blocks like [29] — map into skip_blocks string when empty
                existing = (node.get("inputs") or {}).get("skip_blocks")
                if existing in (None, "", "28") or stg is not None:
                    _set_input(node, "skip_blocks", block_str if float(stg or 0) > 0 else "")

    # All checkpoint-consuming loaders used by LTX graphs
    if ckpt:
        for class_type in (
            "CheckpointLoaderSimple",
            "LTXVAudioVAELoader",
            "LTXAVTextEncoderLoader",
            "LowVRAMCheckpointLoader",
            "LowVRAMAudioVAELoader",
            "UNETLoader",
            "DiffusionModelLoader",
        ):
            for _nid, node in _find_nodes_by_class(workflow, class_type):
                inputs = node.get("inputs") or {}
                if "ckpt_name" in inputs or class_type in (
                    "CheckpointLoaderSimple",
                    "LTXVAudioVAELoader",
                    "LTXAVTextEncoderLoader",
                    "LowVRAMCheckpointLoader",
                    "LowVRAMAudioVAELoader",
                ):
                    _set_input(node, "ckpt_name", ckpt)
                if "unet_name" in inputs:
                    _set_input(node, "unet_name", ckpt)
        if text_encoder:
            for _nid, node in _find_nodes_by_class(workflow, "LTXAVTextEncoderLoader"):
                _set_input(node, "text_encoder", text_encoder)

    if clip_l or t5xxl:
        for _nid, node in _find_nodes_by_class(workflow, "DualCLIPLoader"):
            if clip_l:
                _set_input(node, "clip_name1", clip_l)
            if t5xxl:
                _set_input(node, "clip_name2", t5xxl)

    if vae_name:
        for _nid, node in _find_nodes_by_class(workflow, "VAELoader"):
            _set_input(node, "vae_name", vae_name)

    if lora:
        for class_type in (
            "LoraLoader",
            "LoraLoaderModelOnly",
            "LTXICLoRALoaderModelOnly",
        ):
            for _nid, node in _find_nodes_by_class(workflow, class_type):
                for k in ("lora_name", "lora"):
                    if k in (node.get("inputs") or {}) or k == "lora_name":
                        _set_input(node, k, lora)
                        break

    if image_name:
        for _nid, node in _find_nodes_by_class(workflow, "LoadImage"):
            _set_input(node, "image", image_name)

    video_name = values.get("video_name")
    if video_name:
        for _nid, node in _find_nodes_by_class(workflow, "LoadVideo"):
            _set_input(node, "file", video_name)

    if audio_name:
        for class_type in ("LoadAudio", "VHS_LoadAudio", "AudioLoader"):
            for _nid, node in _find_nodes_by_class(workflow, class_type):
                for k in ("audio", "audio_file", "path", "file"):
                    _set_input(node, k, audio_name)

    if filename_prefix:
        for class_type in (
            "SaveVideo",
            "VHS_VideoCombine",
            "CreateVideo",
            "SaveAnimatedWEBP",
            "SaveImage",
        ):
            for _nid, node in _find_nodes_by_class(workflow, class_type):
                for k in ("filename_prefix", "filename", "save_prefix"):
                    if k in (node.get("inputs") or {}) or k == "filename_prefix":
                        _set_input(node, k, filename_prefix)
                        break


def load_and_patch_workflow(
    variant: str,
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    duration_s: float = 5.0,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    image_name: Optional[str] = None,
    audio_name: Optional[str] = None,
    video_name: Optional[str] = None,
    filename_prefix: str = "ltx_agent",
    global_prompt: Optional[str] = None,
    segment_prompts: Optional[list[str]] = None,
    stg_scale: Optional[float] = None,
    stg_blocks: Optional[list[int]] = None,
    sampler_name: Optional[str] = None,
    frames: Optional[int] = None,
    object_info: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Returns (workflow_api_dict, meta) where meta has resolved generation params.
    """
    width, height = clamp_resolution(width, height, quality="flux" if variant == "flux" else None)
    gen = get_variant_gen(variant)
    if frames is None:
        frames = frames_for_duration(duration_s, fps=gen["fps"], snap=gen["frame_snap"])
    else:
        frames = int(frames)
    if int(gen["frame_snap"]) == 8:
        frames = snap_ltx_frames(frames)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    steps = steps if steps is not None else DEFAULT_STEPS
    cfg = cfg if cfg is not None else DEFAULT_CFG

    models = MODEL_FILES.get(variant) or MODEL_FILES["base"]
    preferred = models.get("checkpoint") or models.get("diffusion")
    # Variants with no single all-in-one checkpoint (e.g. wan22's dual UNETs)
    # must not get a fallback LTX ckpt sprayed onto their loaders.
    checkpoint = resolve_checkpoint_name(preferred) if preferred else None
    lora = models.get("lora")
    text_encoder = models.get("text_encoder")
    clip_l = models.get("clip_l")
    t5xxl = models.get("t5xxl")
    vae_name = models.get("vae")

    # Only inject weight names whose file exists
    if lora and resolve_model_path(lora) is None:
        lora = None
    if clip_l and resolve_model_path(clip_l) is None:
        clip_l = None
    if t5xxl and resolve_model_path(t5xxl) is None:
        t5xxl = None
    if vae_name and resolve_model_path(vae_name) is None:
        vae_name = None

    workflow = copy.deepcopy(load_workflow_template(variant))
    manifests = _load_manifests()
    field_map = (manifests.get(variant) or {}).get("fields") or {}

    values = {
        "prompt": prompt,
        "negative_prompt": negative_prompt
        or "blurry, low quality, distorted face, watermark, text overlay",
        "seed": seed,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "cfg": cfg,
        "checkpoint": checkpoint,
        "checkpoint_high": models.get("checkpoint_high"),
        "checkpoint_low": models.get("checkpoint_low"),
        "lora": lora,
        "clip_l": clip_l,
        "t5xxl": t5xxl,
        "vae_name": vae_name,
        "text_encoder": text_encoder,
        "image_name": image_name,
        "audio_name": audio_name,
        "video_name": video_name,
        "filename_prefix": filename_prefix,
        "global_prompt": global_prompt or prompt,
        "segment_prompts": segment_prompts,
        "fps": gen["fps"],
        "stg_scale": stg_scale,
        "stg_blocks": stg_blocks,
        "sampler_name": sampler_name,
    }

    if variant == "directors" and segment_prompts:
        for i, seg in enumerate(segment_prompts):
            values[f"segment_{i}"] = seg

    _apply_named_fields(workflow, field_map, values)
    _heuristic_patch(workflow, values)
    if lora:
        _ensure_lora_node(workflow, lora)
    _sanitize_ltx_nodes(workflow)

    extras = (manifests.get(variant) or {}).get("extras") or {}
    for key, val in extras.items():
        if isinstance(val, dict) and "node_id" in val:
            nid = str(val["node_id"])
            if nid in workflow and "input" in val and key in values:
                _set_input(workflow[nid], val["input"], values[key])

    # Final sanitize after extras
    _sanitize_ltx_nodes(workflow)

    from master_agent.comfy.graph_ops import (
        LTX_TEACACHE_VARIANTS,
        ensure_teacache,
        looks_like_ltx_graph,
    )

    if variant in LTX_TEACACHE_VARIANTS or looks_like_ltx_graph(workflow):
        ensure_teacache(workflow, object_info)

    meta = {
        "variant": variant,
        "width": width,
        "height": height,
        "frames": frames,
        "duration_s": duration_s,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "stg_scale": stg_scale,
        "stg_blocks": stg_blocks,
        "sampler_name": sampler_name,
        "checkpoint": checkpoint,
        "checkpoint_preferred": preferred,
        "checkpoint_fallback": preferred != checkpoint,
        "lora": lora,
        "filename_prefix": filename_prefix,
    }
    return workflow, meta
