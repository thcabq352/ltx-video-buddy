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
    H3_DEFAULT_CFG,
    H3_DEFAULT_HEIGHT,
    H3_DEFAULT_STEPS,
    H3_DEFAULT_WIDTH,
    H3_MAX_DURATION_S,
    MODELS_DIR,
    MODEL_FILES,
    WORKFLOWS_DIR,
    clamp_h3_resolution,
    clamp_resolution,
    frames_for_duration,
    get_variant_gen,
    is_h3_variant,
    is_ltx25_variant,
    resolve_model_path,
    segment_max_s_for_variant,
    snap_h3_frames,
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


def _expected_template_path(variant: str) -> Path:
    """Best-known on-disk path for ``variant`` (may not exist)."""
    manifests = _load_manifests()
    meta = manifests.get(variant) or {}
    filename = meta.get("file")
    if isinstance(filename, str) and filename.strip():
        return WORKFLOWS_DIR / filename.replace("\\", "/")
    from master_agent.config import WORKFLOW_FILES

    seeded = WORKFLOW_FILES.get(variant)
    if seeded:
        return WORKFLOWS_DIR / seeded
    return WORKFLOWS_DIR / f"{variant}.json"


def load_workflow_template(variant: str) -> dict[str, Any]:
    path: Path | None = None
    try:
        from master_agent.comfy.catalog import resolve_workflow_path

        path = resolve_workflow_path(variant)
    except KeyError:
        path = None
    except FileNotFoundError as exc:
        expected = _expected_template_path(variant)
        raise FileNotFoundError(
            f"Workflow template not found for variant={variant!r}: {expected}"
        ) from exc
    if path is None or not path.is_file():
        expected = _expected_template_path(variant)
        raise FileNotFoundError(
            f"Workflow template not found for variant={variant!r}: {expected}"
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


_SAMPLER_CLASSES = frozenset(
    {"KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced", "LanPaint_KSampler"}
)
_SAMPLER_LATENT_KEYS = frozenset({"latent_image", "latent"})


def image_feeds_sampler_latent(workflow: dict[str, Any], filename: str) -> bool:
    """True when ``filename`` is on a node that links into a sampler latent.

    A dangling unused LoadImage does not count.
    """
    if not filename:
        return False
    owners: list[str] = []
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if value == filename:
                owners.append(str(nid))
    if not owners:
        return False
    children: dict[str, list[tuple[str, str]]] = {}
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        for key, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2:
                children.setdefault(str(value[0]), []).append((str(nid), str(key)))
    seen: set[str] = set()
    stack = list(owners)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for dst, key in children.get(cur, []):
            dst_node = workflow.get(dst)
            if (
                isinstance(dst_node, dict)
                and dst_node.get("class_type") in _SAMPLER_CLASSES
                and key in _SAMPLER_LATENT_KEYS
            ):
                return True
            stack.append(dst)
    return False


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

    for _nid, node in _find_nodes_by_class(workflow, "GemmaAPITextEncode"):
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs.get("enhance_prompt"), bool):
            inputs["enhance_prompt"] = False
        inputs.pop("ckpt_name", None)

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


def _remap_stub_filenames(workflow: dict[str, Any]) -> list[str]:
    """Replace research-graph stub weight names with official HF filenames."""
    from master_agent.models.weights import OPTIONAL_STUBS, STUB_ALIASES

    notes: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            if not isinstance(value, str):
                continue
            official = STUB_ALIASES.get(value)
            if official:
                inputs[key] = official
                notes.append(f"{value} → {official}")
            elif value in OPTIONAL_STUBS:
                notes.append(f"optional stub left: {value}")
    return notes


def _bypass_missing_optional_loras(workflow: dict[str, Any]) -> list[str]:
    """Drop style/camera placeholder LoRA nodes when the file is not on disk."""
    from master_agent.models.weights import OPTIONAL_STUBS, find_weight_file

    dropped: list[str] = []
    for nid, node in list(workflow.items()):
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type") or ""
        if ctype not in ("LoraLoader", "LoraLoaderModelOnly", "LTXVLoraLoader"):
            continue
        inputs = node.get("inputs") or {}
        name = inputs.get("lora_name") or inputs.get("lora")
        if not isinstance(name, str) or name not in OPTIONAL_STUBS:
            continue
        if find_weight_file(name) is not None:
            continue
        source = inputs.get("model")
        sid = str(nid)
        for other in workflow.values():
            if not isinstance(other, dict) or other is node:
                continue
            oin = other.get("inputs") or {}
            for key, value in list(oin.items()):
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and str(value[0]) == sid
                    and source is not None
                ):
                    oin[key] = source
        workflow.pop(sid, None)
        dropped.append(sid)
    return dropped


def _resolved_ltx25_names() -> tuple[str, str]:
    """16GB-class pick: GGUF Q4 → NVFP4 → int8/bf16. TE: heretic/int8 before official bf16."""
    from master_agent.models.weights import WEIGHT_FILES, resolve_weight

    transformer = WEIGHT_FILES["transformer"]
    text_encoder = WEIGHT_FILES["text_encoder"]
    from master_agent.comfy.loader_names import name_from_local_path

    local_tr = resolve_weight(transformer)
    local_te = resolve_weight(text_encoder)
    tr_name = name_from_local_path(local_tr) or transformer.filename
    te_name = name_from_local_path(local_te) or text_encoder.filename
    return tr_name, te_name


def _set_transformer_loader(node: dict[str, Any], filename: str) -> None:
    """Wire UNETLoader vs UnetLoaderGGUF from the file that is actually present."""
    if filename.lower().endswith(".gguf"):
        node["class_type"] = "UnetLoaderGGUF"
        node["inputs"] = {"unet_name": filename}
    else:
        node["class_type"] = "UNETLoader"
        node["inputs"] = {"unet_name": filename, "weight_dtype": "default"}
    node.setdefault("_meta", {})["title"] = "LTX 2.5 Distilled Transformer"


def _rewrite_ltx25_checkpoint_loader(workflow: dict[str, Any]) -> None:
    """Turn CheckpointLoaderSimple + stub/all-in-one name into split-pack loaders.

    Official LTX 2.5 weights are a transformer + Gemma 4 TE + VAEs, not a
    single ``.safetensors`` checkpoint. Research templates still use
    CheckpointLoaderSimple; rewrite so the official files (or a local
    GGUF / NVFP4 / int8 / bf16 stand-in) resolve.
    """
    from master_agent.models.weights import STUB_ALIASES, WEIGHT_FILES

    official_transformer = WEIGHT_FILES["transformer"].filename
    transformer_name, te_name = _resolved_ltx25_names()
    aliases = {official_transformer, *WEIGHT_FILES["transformer"].candidates, *STUB_ALIASES.keys()}

    ckpt_nodes = [
        (nid, node)
        for nid, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") == "CheckpointLoaderSimple"
    ]
    for src_id, node in ckpt_nodes:
        name = (node.get("inputs") or {}).get("ckpt_name")
        if not isinstance(name, str):
            continue
        if name not in aliases and "ltx-2.5" not in name.lower():
            continue
        _set_transformer_loader(node, transformer_name if name in aliases else name)

        if _find_nodes_by_class(workflow, "LTXAVTextEncoderLoader"):
            continue
        numeric_ids = [int(k) for k in workflow if str(k).isdigit()]
        te_id = str(max(numeric_ids) + 1) if numeric_ids else "90"
        workflow[te_id] = {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {"text_encoder": te_name},
            "_meta": {"title": "LTX 2.5 Gemma 4 TE"},
        }
        for other in workflow.values():
            if not isinstance(other, dict):
                continue
            inputs = other.get("inputs") or {}
            for key, value in list(inputs.items()):
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and str(value[0]) == str(src_id)
                    and value[1] == 1
                ):
                    inputs[key] = [te_id, 0]


def _resolved_h3_names(bundle: str) -> tuple[str, str, str, str]:
    """H3 16GB-class pick: GGUF Q4_K DiT + Comfy TE + official VAEs."""
    from master_agent.models.weights import WEIGHT_FILES, resolve_weight

    dit_key = "h3_ref2va" if bundle == "h3_ref2va" else "h3_fl2va"
    dit = WEIGHT_FILES[dit_key]
    te = WEIGHT_FILES["h3_text_encoder"]
    vvae = WEIGHT_FILES["h3_video_vae"]
    avae = WEIGHT_FILES["h3_audio_vae"]
    from master_agent.comfy.loader_names import name_from_local_path

    local_dit = resolve_weight(dit)
    local_te = resolve_weight(te)
    local_vvae = resolve_weight(vvae)
    local_avae = resolve_weight(avae)
    return (
        name_from_local_path(local_dit) or dit.filename,
        name_from_local_path(local_te) or te.filename,
        name_from_local_path(local_vvae) or vvae.filename,
        name_from_local_path(local_avae) or avae.filename,
    )


def _set_h3_dit_loader(node: dict[str, Any], filename: str) -> None:
    if filename.lower().endswith(".gguf"):
        node["class_type"] = "UnetLoaderGGUF"
        node["inputs"] = {"unet_name": filename}
    else:
        node["class_type"] = "UNETLoader"
        node["inputs"] = {"unet_name": filename, "weight_dtype": "default"}
    node.setdefault("_meta", {})["title"] = "MiniMax H3 Transformer"


def _set_h3_clip_loader(node: dict[str, Any], filename: str) -> None:
    if filename.lower().endswith(".gguf"):
        node["class_type"] = "CLIPLoaderGGUF"
        node["inputs"] = {"clip_name": filename, "type": "minimax"}
    else:
        node["class_type"] = "CLIPLoader"
        node["inputs"] = {"clip_name": filename, "type": "minimax"}
    node.setdefault("_meta", {})["title"] = "H3 Qwen3-VL TE"


def _apply_local_h3_weights(workflow: dict[str, Any], bundle: str) -> None:
    dit_name, te_name, video_vae, audio_vae = _resolved_h3_names(bundle)
    for class_type in ("UNETLoader", "UnetLoaderGGUF", "DiffusionModelLoader"):
        for _nid, node in _find_nodes_by_class(workflow, class_type):
            inputs = node.get("inputs") or {}
            current = inputs.get("unet_name") or inputs.get("ckpt_name") or ""
            if isinstance(current, str) and "minimax_h3" in current.lower():
                _set_h3_dit_loader(node, dit_name)
    for class_type in ("CLIPLoader", "CLIPLoaderGGUF"):
        for _nid, node in _find_nodes_by_class(workflow, class_type):
            inputs = node.get("inputs") or {}
            current = inputs.get("clip_name") or ""
            if isinstance(current, str) and "minimax_h3" in current.lower():
                _set_h3_clip_loader(node, te_name)
    for _nid, node in _find_nodes_by_class(workflow, "VAELoader"):
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        inputs = node.get("inputs") or {}
        current = str(inputs.get("vae_name") or "")
        if "audio" in title or "audio_vae" in current.lower():
            _set_input(node, "vae_name", audio_vae)
        elif "video" in title or "minimax_h3_video" in current.lower() or "video_vae" in current.lower():
            _set_input(node, "vae_name", video_vae)


def _set_family_unet_loader(node: dict[str, Any], filename: str, title: str) -> None:
    if filename.lower().endswith(".gguf"):
        node["class_type"] = "UnetLoaderGGUF"
        node["inputs"] = {"unet_name": filename}
    else:
        node["class_type"] = "UNETLoader"
        node["inputs"] = {"unet_name": filename, "weight_dtype": "default"}
    node.setdefault("_meta", {})["title"] = title


def _resolved_family_name(weight_key: str) -> str | None:
    from master_agent.models.weights import WEIGHT_FILES, resolve_weight

    weight = WEIGHT_FILES.get(weight_key)
    if weight is None:
        return None
    from master_agent.comfy.loader_names import name_from_local_path

    found = resolve_weight(weight)
    if found is None:
        return None
    return name_from_local_path(found) or found.name


def _apply_local_family_weights(workflow: dict[str, Any], variant: str) -> None:
    """16GB-class remaps for Wan / VACE / Krea / Flux / Qwen (GGUF vs UNET)."""
    from master_agent.models.vram_policy import family_for_slug

    family = family_for_slug(variant)
    loaders = ("UNETLoader", "UnetLoaderGGUF", "UnetLoader", "DiffusionModelLoader")
    gguf_loaders = ("UnetLoaderGGUF", "UnetLoaderGGUFAdvanced")

    def _rewrite(needles: tuple[str, ...], filename: str, title: str) -> None:
        for class_type in (*loaders, *gguf_loaders):
            for _nid, node in _find_nodes_by_class(workflow, class_type):
                inputs = node.get("inputs") or {}
                current = str(
                    inputs.get("unet_name")
                    or inputs.get("ckpt_name")
                    or inputs.get("gguf_name")
                    or ""
                ).lower()
                if any(token in current for token in needles):
                    _set_family_unet_loader(node, filename, title)
                    if "gguf_name" in inputs:
                        inputs = node.get("inputs") or {}
                        if "gguf_name" in inputs:
                            inputs["gguf_name"] = filename

    if family == "wan22":
        high = _resolved_family_name("wan22_high")
        low = _resolved_family_name("wan22_low")
        if high:
            _rewrite(("high_noise", "highnoise"), high, "Wan 2.2 High (16GB pick)")
        if low:
            _rewrite(("low_noise", "lownoise"), low, "Wan 2.2 Low (16GB pick)")
    elif family == "vace":
        name = _resolved_family_name("vace")
        if name:
            _rewrite(("vace", "skyreels"), name, "VACE Skyreels (16GB pick)")
    elif family == "krea2":
        name = _resolved_family_name("krea2")
        if name:
            _rewrite(("krea2", "krea-2"), name, "Krea-2 turbo (16GB pick)")
    elif family == "flux":
        name = _resolved_family_name("flux")
        if name:
            _rewrite(("flux1-dev", "flux.1-dev", "flux1_dev"), name, "Flux.1-dev (16GB pick)")
    elif family == "qwen_edit":
        name = _resolved_family_name("qwen_edit")
        if not name:
            return
        _rewrite(("qwen-image-edit", "qwen_image_edit"), name, "Qwen-Image-Edit (16GB pick)")
        for _nid, node in _find_nodes_by_class(workflow, "UnetLoaderGGUF"):
            inputs = node.get("inputs") or {}
            current = str(inputs.get("gguf_name") or inputs.get("unet_name") or "")
            if "qwen" in current.lower():
                if "gguf_name" in inputs:
                    inputs["gguf_name"] = name
                else:
                    _set_family_unet_loader(node, name, "Qwen-Image-Edit (16GB pick)")


def _apply_local_ltx25_weights(workflow: dict[str, Any]) -> None:
    """Prefer a local GGUF / NVFP4 / int8 / heretic TE when one is already present."""
    transformer_name, te_name = _resolved_ltx25_names()
    for class_type in ("UNETLoader", "UnetLoaderGGUF", "DiffusionModelLoader"):
        for _nid, node in _find_nodes_by_class(workflow, class_type):
            # Always pin 2.5 loaders to the local/official 2.5 transformer.
            # A 2.3 all-in-one must not remain on unet_name after patch.
            _set_transformer_loader(node, transformer_name)
    for _nid, node in _find_nodes_by_class(workflow, "LTXAVTextEncoderLoader"):
        _set_input(node, "text_encoder", te_name)
        inputs = node.get("inputs") or {}
        if "ckpt_name" in inputs and isinstance(inputs.get("ckpt_name"), str):
            if "ltx-2.5" in str(inputs.get("ckpt_name")).lower():
                _set_input(node, "ckpt_name", transformer_name)
    for _nid, node in _find_nodes_by_class(workflow, "CLIPLoader"):
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        inputs = node.get("inputs") or {}
        current = str(inputs.get("clip_name") or "")
        if "enhancer" in title or "e2b" in current.lower():
            continue
        if any(tok in current.lower() for tok in ("gemma", "ltx-2.5", "ltx25", "ltxv")):
            _set_input(node, "clip_name", te_name)


def _apply_multi_ref(workflow: dict[str, Any], refs: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    mapping = {
        "pic1": refs.get("pic1"),
        "pic2": refs.get("pic2"),
        "pic3": refs.get("pic3"),
        "pic4": refs.get("pic4"),
        "background": refs.get("background"),
    }
    strength = refs.get("strength")
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type") or ""
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        inputs = node.get("inputs") or {}
        if ctype in ("ComfyUILTX25MSRMultiReferenceGuide", "LoadImage"):
            for key, path in mapping.items():
                if not path:
                    continue
                if key in inputs and not isinstance(inputs[key], list):
                    inputs[key] = path
                    notes.append(f"msr {key}")
                elif "image" in inputs and key in title and not isinstance(inputs["image"], list):
                    inputs["image"] = path
                    notes.append(f"load {key}")
            if strength is not None and "strength" in inputs and not isinstance(inputs["strength"], list):
                inputs["strength"] = strength
    return notes


def _apply_typed_loras(workflow: dict[str, Any], slots: list[dict[str, Any]]) -> list[str]:
    """Apply research-agent LoRA slots (standard / ic_lora / msr / camera)."""
    standard = {"LoraLoader", "LoraLoaderModelOnly", "LTXVLoraLoader"}
    ic_classes = {"LTXVICLoRALoader", "ICLoRALoader", "V2VICLoRALoader"}
    msr_classes = {"ComfyUILTX25MSRICLoRALoader", "LTX MSR IC-LoRA Loader"}
    notes: list[str] = []
    for slot in slots:
        kind = str(slot.get("type") or "standard")
        name = slot.get("name")
        strength = slot.get("strength")
        if not name:
            continue
        applied = False
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type") or ""
            title = str((node.get("_meta") or {}).get("title") or "").lower()
            inputs = node.setdefault("inputs", {})
            if kind == "msr" and ctype in msr_classes:
                if "lora_name" in inputs and not isinstance(inputs["lora_name"], list):
                    inputs["lora_name"] = name
                if strength is not None and "strength_model" in inputs:
                    inputs["strength_model"] = strength
                applied = True
            elif kind == "ic_lora" and ctype in ic_classes:
                if "lora_name" in inputs and not isinstance(inputs["lora_name"], list):
                    inputs["lora_name"] = name
                elif "lora" in inputs and not isinstance(inputs["lora"], list):
                    inputs["lora"] = name
                if strength is not None:
                    if "strength_model" in inputs and not isinstance(inputs["strength_model"], list):
                        inputs["strength_model"] = strength
                    elif "strength" in inputs and not isinstance(inputs["strength"], list):
                        inputs["strength"] = strength
                applied = True
            elif kind in ("standard", "camera") and ctype in standard:
                if kind == "camera" and "camera" not in title and "motion" not in title:
                    continue
                if kind == "standard" and ("camera" in title or "ic-lora" in title or "msr" in title):
                    continue
                if "lora_name" in inputs and not isinstance(inputs["lora_name"], list):
                    inputs["lora_name"] = name
                if strength is not None:
                    if "strength_model" in inputs and not isinstance(inputs["strength_model"], list):
                        inputs["strength_model"] = strength
                    if "strength" in inputs and not isinstance(inputs["strength"], list):
                        inputs["strength"] = strength
                applied = True
        notes.append(f"lora {kind}:{name} applied={applied}")
    return notes


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

    # CLIP / text encode (research graphs also title nodes "Positive Prompt")
    for _nid, node in _find_nodes_by_class(workflow, "CLIPTextEncode"):
        title = (node.get("_meta") or {}).get("title", "").lower()
        if negative is not None and "neg" in title:
            _set_input(node, "text", negative)
        elif prompt is not None and "neg" not in title:
            _set_input(node, "text", prompt)
    if prompt is not None:
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            ctype = (node.get("class_type") or "").lower()
            title = str((node.get("_meta") or {}).get("title") or "").lower()
            if "neg" in title:
                continue
            if any(tok in ctype for tok in ("clip", "gemma", "textencode", "prompt", "primitivestring")) or "prompt" in title:
                inputs = node.get("inputs") or {}
                for key in ("text", "prompt", "string", "positive", "value"):
                    if key in inputs and not isinstance(inputs[key], list):
                        inputs[key] = prompt

    # Empty latent / size nodes — keep video length and audio frames_number in sync
    ltx_frames = snap_ltx_frames(int(frames)) if frames is not None else None
    for class_type in (
        "EmptyLTXVLatentVideo",
        "EmptyHunyuanLatentVideo",
        "EmptyLatentImage",
        "LTXVEmptyLatentVideo",
        "WanFunInpaintToVideo",
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
        for class_type in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"):
            for _nid, node in _find_nodes_by_class(workflow, class_type):
                if prompt is not None:
                    _set_input(node, "prompt", prompt)
                if width is not None:
                    _set_input(node, "width", width)
                if height is not None:
                    _set_input(node, "height", height)
                if frames is not None:
                    _set_input(node, "length", int(frames))
        for class_type in (
            "LTXVImgToVideo",
            "LTXVImgToVideoInplace",
            "LTXVImgToVideoConditionOnly",
            "LTXVConditioning",
        ):
            for _nid, node in _find_nodes_by_class(workflow, class_type):
                if values.get("fps") and class_type == "LTXVConditioning":
                    _set_input(node, "frame_rate", int(values.get("fps") or 24))
                if class_type == "LTXVImgToVideo":
                    if width is not None:
                        _set_input(node, "width", width)
                    if height is not None:
                        _set_input(node, "height", height)
                    write = ltx_frames if ltx_frames is not None else int(frames)
                    _set_input(node, "length", write)
                    if values.get("image_name") or values.get("first_image"):
                        _set_input(node, "image", values.get("first_image") or values.get("image_name"))
                    if values.get("last_image"):
                        _set_input(node, "last_frame", values["last_image"])
                if class_type in ("LTXVImgToVideoInplace", "LTXVImgToVideoConditionOnly"):
                    if values.get("image_name") or values.get("first_image"):
                        # Official 2.5 graphs: bypass=True means unused T2V.
                        # An image must actually condition the sampler latent.
                        _set_input(node, "bypass", False)
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

    for class_type in ("KSampler", "KSamplerAdvanced", "LanPaint_KSampler"):
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
        ckpt_l = str(ckpt).lower()
        spray_unet = "ltx-2.5" in ckpt_l or "ltx2.5" in ckpt_l
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
                    # Do not write a 2.3 all-in-one onto official 2.5 UNET loaders.
                    if class_type in ("UNETLoader", "DiffusionModelLoader") and not spray_unet:
                        continue
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
            title = str((node.get("_meta") or {}).get("title") or "").lower()
            current = str((node.get("inputs") or {}).get("vae_name") or "")
            if "audio" in title or "audio_vae" in current.lower():
                continue
            _set_input(node, "vae_name", vae_name)

    if lora:
        for class_type in (
            "LoraLoader",
            "LoraLoaderModelOnly",
            "LTXICLoRALoaderModelOnly",
            "LTXVICLoRALoader",
            "ComfyUILTX25MSRICLoRALoader",
            "LTXVLoraLoader",
        ):
            for _nid, node in _find_nodes_by_class(workflow, class_type):
                for k in ("lora_name", "lora"):
                    if k in (node.get("inputs") or {}) or k == "lora_name":
                        _set_input(node, k, lora)
                        break

    first_image = values.get("first_image") or image_name
    last_image = values.get("last_image")
    if first_image:
        assigned = False
        for _nid, node in _find_nodes_by_class(workflow, "LoadImage"):
            title = str((node.get("_meta") or {}).get("title") or "").lower()
            if "last" in title:
                continue
            _set_input(node, "image", first_image)
            assigned = True
            break
        if not assigned:
            for _nid, node in _find_nodes_by_class(workflow, "LoadImage"):
                _set_input(node, "image", first_image)
                break
        # Official LTX 2.5 T2V/I2V: PrimitiveBoolean drives I2V bypass via NOT.
        # true = use image (I2V); false = unused T2V.
        for _nid, node in _find_nodes_by_class(workflow, "PrimitiveBoolean"):
            _set_input(node, "value", True)
    if last_image:
        for _nid, node in _find_nodes_by_class(workflow, "LoadImage"):
            title = str((node.get("_meta") or {}).get("title") or "").lower()
            if "last" in title:
                _set_input(node, "image", last_image)
        for _nid, node in _find_nodes_by_class(workflow, "LTXVImgToVideo"):
            _set_input(node, "last_frame", last_image)

    mask_name = values.get("mask") or values.get("mask_name")
    if mask_name:
        for _nid, node in _find_nodes_by_class(workflow, "LoadImageMask"):
            _set_input(node, "image", mask_name)

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
            "SaveAudio",
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
    mask_name: Optional[str] = None,
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
    first_image: Optional[str] = None,
    last_image: Optional[str] = None,
    loras: Optional[list[dict[str, Any]]] = None,
    multi_ref: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Returns (workflow_api_dict, meta) where meta has resolved generation params.
    """
    h3 = is_h3_variant(variant)
    if h3:
        if width == 768 and height == 512:
            width, height = H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT
        width, height = clamp_h3_resolution(width, height)
    else:
        width, height = clamp_resolution(width, height, quality="flux" if variant == "flux" else None)
    gen = get_variant_gen(variant)
    if frames is None:
        if h3:
            cap_s = H3_MAX_DURATION_S
        elif is_ltx25_variant(variant):
            cap_s = segment_max_s_for_variant(variant)
        else:
            cap_s = None
        frames = frames_for_duration(
            duration_s,
            fps=gen["fps"],
            snap=gen["frame_snap"],
            max_s=cap_s,
            variant=variant,
        )
    else:
        frames = int(frames)
    if h3 or int(gen["frame_snap"]) == 17:
        frames = snap_h3_frames(frames)
    elif int(gen["frame_snap"]) == 8:
        frames = snap_ltx_frames(frames)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    if h3:
        steps = steps if steps is not None else H3_DEFAULT_STEPS
        cfg = H3_DEFAULT_CFG
    else:
        steps = steps if steps is not None else DEFAULT_STEPS
        cfg = cfg if cfg is not None else DEFAULT_CFG

    try:
        from master_agent.comfy.catalog import H3_ALIASES, RESEARCH_ALIASES

        resolved_id = H3_ALIASES.get(variant, RESEARCH_ALIASES.get(variant, variant))
    except Exception:
        resolved_id = variant
    models = MODEL_FILES.get(resolved_id) or MODEL_FILES.get(variant) or MODEL_FILES["base"]
    preferred = models.get("checkpoint") or models.get("diffusion")
    # Variants with no single all-in-one checkpoint (e.g. wan22's dual UNETs)
    # must not get a fallback LTX ckpt sprayed onto their loaders.
    # LTX 2.5 is a split pack: never resolve a 2.3 all-in-one onto UNETLoader.
    if is_ltx25_variant(variant):
        checkpoint = None
    else:
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
        "image_name": image_name or first_image,
        "image": image_name or first_image,
        "mask": mask_name,
        "mask_name": mask_name,
        "first_image": first_image or image_name,
        "last_image": last_image,
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
    _remap_stub_filenames(workflow)
    from master_agent.models.weights import bundle_for_variant, is_h3_bundle, is_ltx25_bundle

    bundle = bundle_for_variant(variant)
    if is_ltx25_bundle(bundle):
        _rewrite_ltx25_checkpoint_loader(workflow)
    _heuristic_patch(workflow, values)
    if is_ltx25_bundle(bundle):
        _apply_local_ltx25_weights(workflow)
    if is_h3_bundle(bundle):
        _apply_local_h3_weights(workflow, bundle or "h3_fl2va")
    _apply_local_family_weights(workflow, resolved_id or variant)
    if loras:
        _apply_typed_loras(workflow, loras)
    if multi_ref:
        _apply_multi_ref(workflow, multi_ref)
    if lora:
        _ensure_lora_node(workflow, lora)
    _bypass_missing_optional_loras(workflow)
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

    if object_info:
        from master_agent.comfy.loader_names import normalize_loader_widgets

        normalize_loader_widgets(workflow, object_info=object_info)

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
    try:
        from master_agent.models.vram_policy import prepare_warning, workflow_row

        row = workflow_row(resolved_id or variant)
        meta["vram_class"] = row.vram_class
        meta["vram_peak_gb"] = row.expected_vram_gb
        meta["default_pack"] = row.default_pack
        meta["safer_alternate"] = row.safer_alternate
        warn = prepare_warning(resolved_id or variant)
        if warn:
            meta["prepare_warning"] = warn
    except Exception:
        pass
    return workflow, meta
