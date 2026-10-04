"""Vertical Blaze Buddy remakes on the shipped LTX 2.5 Ingredients graph.

``ltx25_msr`` is one ``LTXAddVideoICLoRAGuide`` fed by a repeated still, plus
``LTXICLoRALoaderModelOnly`` on the Ingredients LoRA. Cached object_info does
not register ``ComfyUILTX25MSRMultiReferenceGuide``. This module does not add
a second stack: it patches that graph with a Scott guide and a chained Blaze
guide, then builds ``buddy.clip.provenance/v1``.

The rough source verticals are 720×1280×241. That count is legal ``8n+1`` and
is not the 16GB default. Safe smoke canvas is 448×800×97. This recipe will
not queue the separate landscape 30s intro (frame cap 241, own filename
prefixes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from master_agent.comfy.workflow_patcher import (
    _find_nodes_by_class,
    load_and_patch_workflow,
    media_wiring_error,
)
from master_agent.config import KNOWLEDGE_DIR, snap_ltx_frames
from master_agent.provenance import (
    CLIP_PROVENANCE_SCHEMA,
    build_clip_provenance,
    sha256_file,
)

VARIANT = "ltx25_msr"
FPS = 24
SAFE_WIDTH = 448
SAFE_HEIGHT = 800
SAFE_FRAMES = 97  # 8*12+1, ~4.04s, 9:16, under the 768×512 pixel budget
SOURCE_WIDTH = 720
SOURCE_HEIGHT = 1280
SOURCE_FRAMES = 241  # 8*30+1, the rough verticals. Not the 16GB default.
MAX_EDGE = 1280
MAX_FRAMES = SOURCE_FRAMES
GUIDE_STRENGTH = 1

_SCOTT_TITLE = "Scott reference"
_BLAZE_TITLE = "Blaze reference"
_BLAZE_LOAD = "buddy_blaze_load"
_BLAZE_VIDEO = "buddy_blaze_video"
_BLAZE_PARTS = "buddy_blaze_parts"
_BLAZE_RESIZE = "buddy_blaze_resize"
_BLAZE_REPEAT = "buddy_blaze_repeat"
_BLAZE_GUIDE = "buddy_blaze_guide"

_ALIASES = {
    "blaze-concert": "blaze-concert",
    "concert": "blaze-concert",
    "blaze-pier": "blaze-pier",
    "pier": "blaze-pier",
    "clearwater": "blaze-pier",
    "clearwater-pier": "blaze-pier",
}


@dataclass(frozen=True)
class RemakeRecipe:
    id: str
    filename_prefix: str
    brief_name: str
    positive: str
    negative: str


@dataclass(frozen=True)
class PreparedRemake:
    recipe_id: str
    variant: str
    workflow: dict[str, Any]
    provenance: dict[str, Any]
    warning: str
    width: int
    height: int
    frames: int
    scott: str
    blaze: str


def is_blaze_remake(name: str | None) -> bool:
    return str(name or "").strip().lower() in _ALIASES


def resolve_recipe_id(name: str) -> str:
    key = str(name or "").strip().lower()
    try:
        return _ALIASES[key]
    except KeyError as exc:
        known = ", ".join(sorted(set(_ALIASES.values())))
        raise ValueError(f"unknown blaze remake {name!r}. Known: {known}") from exc


def _brief_path(recipe_id: str) -> Path:
    slug = {
        "blaze-concert": "2026-09-29-blaze-concert-remake.md",
        "blaze-pier": "2026-09-29-blaze-pier-remake.md",
    }[recipe_id]
    return KNOWLEDGE_DIR / "prompts" / slug


def _section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"prompt brief missing {marker}")
    rest = text[start + len(marker) :]
    nxt = rest.find("\n## ")
    body = rest if nxt < 0 else rest[:nxt]
    return " ".join(body.split())


def load_recipe(name: str) -> RemakeRecipe:
    recipe_id = resolve_recipe_id(name)
    path = _brief_path(recipe_id)
    if not path.is_file():
        raise ValueError(f"missing prompt brief {path.name}")
    text = path.read_text(encoding="utf-8")
    return RemakeRecipe(
        id=recipe_id,
        filename_prefix=recipe_id,
        brief_name=path.name,
        positive=_section(text, "Positive"),
        negative=_section(text, "Negative"),
    )


def resolve_canvas(
    width: int | None,
    height: int | None,
    frames: int | None,
) -> tuple[int, int, int, str]:
    """Safe 448×800×97 unless the operator overrides. Snap, do not invent a render."""
    if (width is None) ^ (height is None):
        raise ValueError("pass both width and height, or neither for the safe default")
    requested_w = int(width) if width is not None else None
    requested_h = int(height) if height is not None else None
    if requested_w is None:
        canvas_w, canvas_h = SAFE_WIDTH, SAFE_HEIGHT
    else:
        assert requested_h is not None
        canvas_w = max(256, (requested_w // 32) * 32)
        canvas_h = max(256, (requested_h // 32) * 32)
    if frames is None:
        count = SAFE_FRAMES
    else:
        count = snap_ltx_frames(int(frames))
    if canvas_w > MAX_EDGE or canvas_h > MAX_EDGE:
        raise ValueError(
            f"blaze remake refuses {canvas_w}x{canvas_h}: long edge cap is {MAX_EDGE} on 16GB. "
            "The landscape 30s intro is a different job."
        )
    if count > MAX_FRAMES:
        raise ValueError(
            f"blaze remake refuses {count} frames. Cap is {MAX_FRAMES} "
            "(the rough vertical length). This recipe does not queue the landscape 30s intro."
        )
    warning = ""
    if (canvas_w, canvas_h, count) != (SAFE_WIDTH, SAFE_HEIGHT, SAFE_FRAMES):
        snapped = ""
        if requested_w is not None and (
            canvas_w != requested_w or canvas_h != requested_h
        ):
            snapped = (
                f" Requested {requested_w}x{requested_h} snapped to the 32-pixel grid "
                f"as {canvas_w}x{canvas_h}."
            )
        warning = (
            "16GB WARNING: safe blaze remake default is "
            f"{SAFE_WIDTH}x{SAFE_HEIGHT}x{SAFE_FRAMES} "
            f"(~{SAFE_FRAMES / FPS:.2f}s, 9:16, under the 768x512 pixel budget). "
            f"This job is {canvas_w}x{canvas_h}x{count} and is unmeasured on a 16GB card. "
            f"{SOURCE_WIDTH}x{SOURCE_HEIGHT}x{SOURCE_FRAMES} is the rough source geometry, "
            "not the default. ltx25_msr is already catalog class tight (~13.2 GB) "
            "before this canvas."
            + snapped
        )
    return canvas_w, canvas_h, count, warning


def _still_name(path: str) -> str:
    name = Path(str(path or "").strip()).name
    if not name or name in {".", ".."}:
        raise ValueError("character still path is empty")
    return name


def _apply_guide_resize(node: dict[str, Any], width: int, height: int) -> None:
    inputs = node.setdefault("inputs", {})
    inputs.pop("resize_type.shorter_size", None)
    inputs["resize_type"] = "scale dimensions"
    inputs["resize_type.width"] = int(width)
    inputs["resize_type.height"] = int(height)
    inputs["scale_method"] = "lanczos"


def _set_scott(workflow: dict[str, Any], name: str) -> None:
    for _nid, node in _find_nodes_by_class(workflow, "LoadImage"):
        title = str((node.get("_meta") or {}).get("title") or "")
        if "blaze" in title.lower():
            continue
        node.setdefault("inputs", {})["image"] = name
        node.setdefault("_meta", {})["title"] = _SCOTT_TITLE
        return
    raise ValueError("ltx25_msr has no guide LoadImage for Scott")


def _set_blaze_image(workflow: dict[str, Any], name: str) -> None:
    node = workflow.get(_BLAZE_LOAD)
    if not isinstance(node, dict):
        raise ValueError("blaze guide LoadImage is missing")
    node.setdefault("inputs", {})["image"] = name
    node.setdefault("_meta", {})["title"] = _BLAZE_TITLE


def attach_blaze_guide(
    workflow: dict[str, Any],
    *,
    scott_name: str,
    blaze_name: str,
    width: int,
    height: int,
) -> None:
    """Chain a second Ingredients guide. Scott stays on the shipped LoadImage."""
    _set_scott(workflow, scott_name)
    for _nid, node in _find_nodes_by_class(workflow, "ResizeImageMaskNode"):
        _apply_guide_resize(node, width, height)
    guides = _find_nodes_by_class(workflow, "LTXAddVideoICLoRAGuide")
    if not guides:
        raise ValueError("ltx25_msr has no LTXAddVideoICLoRAGuide")
    if _BLAZE_GUIDE in workflow:
        _set_blaze_image(workflow, blaze_name)
        return
    if len(guides) != 1:
        raise ValueError("ltx25_msr expected one IC-LoRA guide before the Blaze chain")
    first_id, first = guides[0]
    first_inputs = first.setdefault("inputs", {})
    first_inputs["strength"] = GUIDE_STRENGTH
    repeats = _find_nodes_by_class(workflow, "RepeatImageBatch")
    if not repeats:
        raise ValueError("ltx25_msr has no RepeatImageBatch for the guide")
    amount = (repeats[0][1].get("inputs") or {}).get("amount")
    if not (isinstance(amount, list) and len(amount) == 2):
        raise ValueError("guide RepeatImageBatch amount is not linked to the frame count")
    workflow[_BLAZE_LOAD] = {
        "class_type": "LoadImage",
        "inputs": {"image": blaze_name},
        "_meta": {"title": _BLAZE_TITLE},
    }
    workflow[_BLAZE_VIDEO] = {
        "class_type": "CreateVideo",
        "inputs": {
            "fps": float(FPS),
            "bit_depth": 8,
            "images": [_BLAZE_LOAD, 0],
        },
        "_meta": {"title": "Blaze guide video"},
    }
    workflow[_BLAZE_PARTS] = {
        "class_type": "GetVideoComponents",
        "inputs": {"video": [_BLAZE_VIDEO, 0]},
        "_meta": {"title": "Blaze guide components"},
    }
    workflow[_BLAZE_RESIZE] = {
        "class_type": "ResizeImageMaskNode",
        "inputs": {
            "input": [_BLAZE_PARTS, 0],
            "resize_type": "scale dimensions",
            "resize_type.width": int(width),
            "resize_type.height": int(height),
            "scale_method": "lanczos",
        },
        "_meta": {"title": "Blaze guide resize"},
    }
    workflow[_BLAZE_REPEAT] = {
        "class_type": "RepeatImageBatch",
        "inputs": {"image": [_BLAZE_RESIZE, 0], "amount": amount},
        "_meta": {"title": "repeat the Blaze reference"},
    }
    workflow[_BLAZE_GUIDE] = {
        "class_type": "LTXAddVideoICLoRAGuide",
        "inputs": {
            "frame_idx": 0,
            "strength": GUIDE_STRENGTH,
            "crop": "disabled",
            "use_tiled_encode": False,
            "tile_size": int(first_inputs.get("tile_size") or 256),
            "tile_overlap": int(first_inputs.get("tile_overlap") or 64),
            "positive": [first_id, 0],
            "negative": [first_id, 1],
            "vae": first_inputs.get("vae"),
            "latent": [first_id, 2],
            "image": [_BLAZE_REPEAT, 0],
            "latent_downscale_factor": first_inputs.get("latent_downscale_factor"),
        },
        "_meta": {"title": "Blaze IC-LoRA guide"},
    }
    for nid, node in workflow.items():
        if nid == _BLAZE_GUIDE or not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            if (
                isinstance(value, list)
                and len(value) == 2
                and str(value[0]) == str(first_id)
            ):
                inputs[key] = [_BLAZE_GUIDE, value[1]]


def _manual_steps(workflow: dict[str, Any]) -> int | None:
    for _nid, node in _find_nodes_by_class(workflow, "ManualSigmas"):
        raw = str((node.get("inputs") or {}).get("sigmas") or "")
        parts = [part for part in raw.split(",") if part.strip()]
        if len(parts) >= 2:
            return len(parts) - 1
    return None


def _graph_cfg(workflow: dict[str, Any]) -> float:
    for _nid, node in _find_nodes_by_class(workflow, "CFGGuider"):
        raw = (node.get("inputs") or {}).get("cfg")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    return 1.0


def _latent_size(workflow: dict[str, Any]) -> tuple[int, int, int]:
    nodes = _find_nodes_by_class(workflow, "EmptyLTXVLatentVideo")
    if not nodes:
        raise ValueError("ltx25_msr has no EmptyLTXVLatentVideo")
    inputs = nodes[0][1].get("inputs") or {}
    return int(inputs["width"]), int(inputs["height"]), int(inputs["length"])


def prepare_blaze_remake(
    recipe: str,
    *,
    scott: str,
    blaze: str,
    prompt: str = "",
    negative: str | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
) -> PreparedRemake:
    spec = load_recipe(recipe)
    scott_name = _still_name(scott)
    blaze_name = _still_name(blaze)
    if scott_name == blaze_name:
        raise ValueError("Scott and Blaze stills must be different files")
    canvas_w, canvas_h, count, warning = resolve_canvas(width, height, frames)
    positive = prompt.strip() if isinstance(prompt, str) and prompt.strip() else spec.positive
    negative_text = (
        negative.strip()
        if isinstance(negative, str) and negative.strip()
        else spec.negative
    )
    used_seed = 42 if seed is None else int(seed)
    workflow, meta = load_and_patch_workflow(
        VARIANT,
        prompt=positive,
        negative_prompt=negative_text,
        width=canvas_w,
        height=canvas_h,
        frames=count,
        duration_s=count / float(FPS),
        seed=used_seed,
        image_name=scott_name,
        filename_prefix=spec.filename_prefix,
        clamp_canvas=False,
    )
    attach_blaze_guide(
        workflow,
        scott_name=scott_name,
        blaze_name=blaze_name,
        width=canvas_w,
        height=canvas_h,
    )
    for label, name in (("Scott", scott_name), ("Blaze", blaze_name)):
        wiring = media_wiring_error(workflow, image_name=name)
        if wiring:
            raise ValueError(f"{label} guide was not wired into the sampler: {wiring}")
    lat_w, lat_h, lat_len = _latent_size(workflow)
    if (lat_w, lat_h, lat_len) != (canvas_w, canvas_h, count):
        raise ValueError(
            f"latent {lat_w}x{lat_h}x{lat_len} does not match {canvas_w}x{canvas_h}x{count}"
        )
    steps = _manual_steps(workflow)
    if steps is None:
        steps = int(meta.get("steps") or 0)
    state = SimpleNamespace(
        request=f"{spec.id} vertical remake",
        prompt=positive,
        negative_prompt=negative_text,
        variant=VARIANT,
        seed=used_seed,
        steps=steps,
        cfg=_graph_cfg(workflow),
        width=canvas_w,
        height=canvas_h,
        fps=FPS,
        duration_s=count / float(FPS),
        ref_names=[scott_name, blaze_name],
        attempt=1,
        shot_index=1,
        shot_id="shot-1",
        judge_score=0.0,
        judge_issues=[],
        revise_history=[],
        shot=None,
    )
    payload = build_clip_provenance(
        state,
        path=f"outputs/{spec.filename_prefix}.mp4",
        omit_hash=True,
    )
    if payload.get("schema") != CLIP_PROVENANCE_SCHEMA:
        raise ValueError("remake provenance left buddy.clip.provenance/v1")
    return PreparedRemake(
        recipe_id=spec.id,
        variant=VARIANT,
        workflow=workflow,
        provenance=payload,
        warning=warning,
        width=canvas_w,
        height=canvas_h,
        frames=count,
        scott=scott_name,
        blaze=blaze_name,
    )


def upload_remake_stills(
    workflow: dict[str, Any],
    upload: Callable[[Path], str],
    scott: str,
    blaze: str,
) -> None:
    """Upload operator stills before queue. Prepare mode does not call this."""
    wanted = {
        _SCOTT_TITLE: scott,
        _BLAZE_TITLE: blaze,
    }
    found: set[str] = set()
    for _nid, node in _find_nodes_by_class(workflow, "LoadImage"):
        title = str((node.get("_meta") or {}).get("title") or "")
        src = wanted.get(title)
        if not src:
            continue
        file_path = Path(src)
        if not file_path.is_file():
            who = "Scott" if title == _SCOTT_TITLE else "Blaze"
            raise FileNotFoundError(
                f"{who} still not found ({src}). Keep identity stills on the tower; do not commit them."
            )
        name = upload(file_path) or file_path.name
        node.setdefault("inputs", {})["image"] = str(name)
        found.add(title)
    missing = [title for title in wanted if title not in found]
    if missing:
        raise ValueError(f"remake graph is missing guide LoadImage titles: {missing}")


def provenance_for_clip(prepared: PreparedRemake, clip_path: str | Path) -> dict[str, Any]:
    """Same payload, hash only when the clip file exists. No absolute paths."""
    payload = json.loads(json.dumps(prepared.provenance))
    path = Path(clip_path)
    payload["output_path"] = f"outputs/{path.name}"
    payload["hash"] = sha256_file(path)
    payload["schema"] = CLIP_PROVENANCE_SCHEMA
    return payload
