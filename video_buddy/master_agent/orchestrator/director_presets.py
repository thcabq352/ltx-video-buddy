"""Director presets — named craft packs that pin a catalog variant plus recipe params.

Phase 0 ships the ``rainey1`` pack (prompt, negative, frames, size). It does
not train a LoRA and does not install a face identity. Frame counts go through
``snap_ltx_frames`` (8n+1, minimum 9).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

from master_agent.config import DEFAULT_FPS, snap_ltx_frames

PACK_PATH = Path(__file__).resolve().parent / "presets" / "rainey1.json"
PRESET_ID = "rainey1"
# Geometry (frames / width / height) is only written onto LTX 2.3 graphs
# whose latent length is 8n+1. Wan and H3 keep their own snap.
LTX_GEOMETRY_VARIANTS = frozenset({"base", "eros", "directors"})
_KEYWORD = re.compile(r"(?<![a-z0-9])rainey1(?![a-z0-9])")
# Story before myth so "9:16" is not eaten by a "16:9" substring check.
_MATCH_ORDER = (
    "rainey1_story_9x16",
    "rainey1_myth_16x9",
    "rainey1_density",
    "rainey1_breach",
    "rainey1_lock_open",
)


class PresetError(ValueError):
    """Recipe pack is missing or illegal."""


@dataclass(frozen=True)
class DirectorRecipe:
    id: str
    cues: tuple[str, ...]
    frames: int
    width: int
    height: int
    variant: str
    purpose: str
    anchor: str
    prompt_pattern: str
    delivery: str
    vertical_latent: bool
    notes: str


@dataclass(frozen=True)
class DirectorPresetPack:
    id: str
    keyword: str
    variant: str
    variant_note: str
    identity_policy: str
    negative: str
    additives: tuple[str, ...]
    recipes: tuple[DirectorRecipe, ...]
    path: str


@dataclass(frozen=True)
class Rainey1Plan:
    """Resolved generation params for one rainey1 brief."""

    recipe_id: str
    variant: str
    frames: Optional[int]
    width: int
    height: int
    negative: str
    prompt: str
    delivery: str
    notes: str
    purpose: str
    apply_geometry: bool


def recipe_frames(n: int) -> int:
    """Legal latent length for a director recipe: 8n+1, minimum 9."""
    return snap_ltx_frames(n)


def mentions_rainey1(text: str | None) -> bool:
    return _KEYWORD.search((text or "").lower()) is not None


def preset_variant_for_request(text: str | None) -> Optional[str]:
    """Catalog variant a rainey1 brief should pin, or None."""
    if not mentions_rainey1(text):
        return None
    return load_rainey1_pack().variant


@lru_cache(maxsize=1)
def load_rainey1_pack() -> DirectorPresetPack:
    """Load and snap the shipped rainey1 recipe pack."""
    if not PACK_PATH.is_file():
        raise PresetError(f"rainey1 preset pack not found: {PACK_PATH}")
    try:
        data = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PresetError(f"rainey1 preset pack is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PresetError("rainey1 preset pack must be a JSON object")
    schema = str(data.get("schema") or "")
    if schema != "buddy.director.preset/v1":
        raise PresetError(f"unsupported preset schema {schema!r}")
    negative = str(data.get("negative") or "").strip()
    if not negative:
        raise PresetError("rainey1 preset is missing its negative prompt")
    additives_raw = data.get("additives") or []
    if not isinstance(additives_raw, list) or not additives_raw:
        raise PresetError("rainey1 preset is missing positive additives")
    additives = tuple(str(item).strip() for item in additives_raw if str(item).strip())
    recipes_raw = data.get("recipes") or []
    if not isinstance(recipes_raw, list) or not recipes_raw:
        raise PresetError("rainey1 preset has no recipes")
    recipes = tuple(_recipe_from_mapping(item) for item in recipes_raw)
    variant = str(data.get("preferred_variant") or "base").strip() or "base"
    return DirectorPresetPack(
        id=str(data.get("id") or PRESET_ID),
        keyword=str(data.get("keyword") or PRESET_ID),
        variant=variant,
        variant_note=str(data.get("variant_note") or ""),
        identity_policy=str(data.get("identity_policy") or ""),
        negative=negative,
        additives=additives,
        recipes=recipes,
        path=str(PACK_PATH),
    )


def clear_preset_cache() -> None:
    load_rainey1_pack.cache_clear()


def recipe_by_id(recipe_id: str) -> DirectorRecipe:
    pack = load_rainey1_pack()
    for recipe in pack.recipes:
        if recipe.id == recipe_id:
            return recipe
    raise PresetError(f"unknown rainey1 recipe {recipe_id!r}")


def select_recipe(text: str | None) -> Optional[DirectorRecipe]:
    """Pick a rainey1 recipe from brief cues. None when the cue is absent."""
    if not mentions_rainey1(text):
        return None
    pack = load_rainey1_pack()
    by_id = {recipe.id: recipe for recipe in pack.recipes}
    lowered = (text or "").lower()
    for recipe_id in _MATCH_ORDER:
        recipe = by_id.get(recipe_id)
        if recipe is not None and _cues_match(lowered, recipe):
            return recipe
    return by_id.get("rainey1_lock_open") or pack.recipes[0]


def compose_positive(
    brief: str,
    recipe: DirectorRecipe,
    additives: Sequence[str],
) -> str:
    """Brief plus shot grammar and the shared craft additives."""
    text = (brief or "").strip().rstrip(",").strip()
    anchor = (recipe.anchor or "").strip()
    pattern = (recipe.prompt_pattern or "").strip()
    if pattern and (not anchor or anchor.lower() not in text.lower()):
        text = f"{text}, {pattern}" if text else pattern
    missing: list[str] = []
    lowered = text.lower()
    for phrase in additives:
        token = phrase.strip()
        if token and token.lower() not in lowered:
            missing.append(token)
    if missing:
        text = text.rstrip(" ,") + ", " + ", ".join(missing)
    return text


def fill_rainey1(
    request: str,
    *,
    variant: str | None = None,
    width: int = 768,
    height: int = 512,
    frames: int | None = None,
    negative: str | None = None,
) -> Optional[Rainey1Plan]:
    """Fill gaps on a rainey1 brief.

    When ``frames`` is already set, geometry is left alone (the caller
    resolved size and length). Otherwise an LTX 2.3 pin adopts the recipe
    size when the caller is still on the 768×512 default, and the recipe
    frame count.
    """
    recipe = select_recipe(request)
    if recipe is None:
        return None
    pack = load_rainey1_pack()
    chosen = (variant or "").strip() or recipe.variant
    geometry = chosen in LTX_GEOMETRY_VARIANTS
    apply_geometry = geometry and frames is None
    out_w, out_h = int(width), int(height)
    out_frames: Optional[int] = frames
    if apply_geometry:
        out_frames = recipe.frames
        if (out_w, out_h) == (768, 512):
            out_w, out_h = recipe.width, recipe.height
    out_negative = negative.strip() if isinstance(negative, str) and negative.strip() else pack.negative
    return Rainey1Plan(
        recipe_id=recipe.id,
        variant=chosen,
        frames=out_frames,
        width=out_w,
        height=out_h,
        negative=out_negative,
        prompt=compose_positive(request, recipe, pack.additives),
        delivery=recipe.delivery,
        notes=recipe.notes,
        purpose=recipe.purpose,
        apply_geometry=apply_geometry,
    )


def apply_rainey1_namespace(args: Any) -> Optional[Rainey1Plan]:
    """Pin variant, size, frames, negative, and additives on a CLI namespace.

    ``--variant``, ``--width``, ``--height``, and ``--duration`` win when the
    user passed them. An unlocked brief takes the recipe. ``--variant`` outside
    the LTX 2.3 geometry set keeps that graph's own size and frame snap.
    """
    request = str(getattr(args, "request", "") or "")
    recipe = select_recipe(request)
    if recipe is None:
        return None
    pack = load_rainey1_pack()
    variant_locked = bool(getattr(args, "variant", None))
    chosen = str(args.variant) if variant_locked else recipe.variant
    geometry = (not variant_locked) or chosen in LTX_GEOMETRY_VARIANTS
    width = int(getattr(args, "width", 768) or 768)
    height = int(getattr(args, "height", 512) or 512)
    if geometry and not getattr(args, "width_set", False):
        width = recipe.width
        args.width = width
    if geometry and not getattr(args, "height_set", False):
        height = recipe.height
        args.height = height
    frames: Optional[int]
    if geometry and not getattr(args, "duration_set", False):
        frames = recipe.frames
        args.frames = frames
        args.duration = frames / float(DEFAULT_FPS)
    elif geometry and getattr(args, "duration_set", False):
        from master_agent.config import frames_for_duration

        frames = frames_for_duration(float(args.duration), variant=chosen)
        args.frames = frames
    else:
        frames = getattr(args, "frames", None)
    if not variant_locked:
        args.variant = chosen
    if not str(getattr(args, "negative_prompt", "") or "").strip():
        args.negative_prompt = pack.negative
    args.request = compose_positive(request, recipe, pack.additives)
    args.rainey_recipe_id = recipe.id
    return Rainey1Plan(
        recipe_id=recipe.id,
        variant=chosen,
        frames=frames if isinstance(frames, int) else recipe.frames,
        width=int(args.width),
        height=int(args.height),
        negative=str(args.negative_prompt or pack.negative),
        prompt=str(args.request),
        delivery=recipe.delivery,
        notes=recipe.notes,
        purpose=recipe.purpose,
        apply_geometry=geometry and not getattr(args, "duration_set", False),
    )


def reapply_rainey1_prompt(args: Any) -> None:
    """Put shot grammar and additives back after intake rewrites the brief."""
    recipe_id = getattr(args, "rainey_recipe_id", None)
    if not recipe_id:
        return
    try:
        recipe = recipe_by_id(str(recipe_id))
    except PresetError:
        return
    pack = load_rainey1_pack()
    args.request = compose_positive(str(getattr(args, "request", "") or ""), recipe, pack.additives)


def list_recipe_rows() -> list[dict[str, Any]]:
    """Rows for ``python -m master_agent workflows``."""
    pack = load_rainey1_pack()
    rows: list[dict[str, Any]] = []
    for recipe in pack.recipes:
        rows.append(
            {
                "id": recipe.id,
                "kind": "recipe",
                "family": pack.id,
                "variant": recipe.variant,
                "path": "master_agent/orchestrator/presets/rainey1.json",
                "name": recipe.id,
                "description": recipe.purpose,
                "frames": recipe.frames,
                "width": recipe.width,
                "height": recipe.height,
                "delivery": recipe.delivery,
            }
        )
    return rows


def format_preset_line(plan: Rainey1Plan) -> str:
    frames = plan.frames if plan.frames is not None else "?"
    return (
        f"preset: {plan.recipe_id} variant={plan.variant} "
        f"{plan.width}x{plan.height} frames={frames}"
    )


def _recipe_from_mapping(raw: Any) -> DirectorRecipe:
    if not isinstance(raw, dict):
        raise PresetError("recipe entries must be objects")
    recipe_id = str(raw.get("id") or "").strip()
    if not recipe_id:
        raise PresetError("recipe is missing id")
    try:
        frames = recipe_frames(int(raw.get("frames")))
    except (TypeError, ValueError) as exc:
        raise PresetError(f"{recipe_id} has no integer frames") from exc
    try:
        width = int(raw.get("width"))
        height = int(raw.get("height"))
    except (TypeError, ValueError) as exc:
        raise PresetError(f"{recipe_id} is missing width/height") from exc
    if width % 32 or height % 32:
        raise PresetError(f"{recipe_id} size {width}x{height} is not on the 32px latent grid")
    cues_raw = raw.get("cues") or []
    if not isinstance(cues_raw, list):
        raise PresetError(f"{recipe_id} cues must be a list")
    return DirectorRecipe(
        id=recipe_id,
        cues=tuple(str(cue).strip().lower() for cue in cues_raw if str(cue).strip()),
        frames=frames,
        width=width,
        height=height,
        variant=str(raw.get("variant") or "base").strip() or "base",
        purpose=str(raw.get("purpose") or ""),
        anchor=str(raw.get("anchor") or ""),
        prompt_pattern=str(raw.get("prompt_pattern") or ""),
        delivery=str(raw.get("delivery") or ""),
        vertical_latent=bool(raw.get("vertical_latent")),
        notes=str(raw.get("notes") or ""),
    )


def _cues_match(text: str, recipe: DirectorRecipe) -> bool:
    if recipe.id in text or recipe.id.replace("_", " ") in text:
        return True
    if recipe.id == "rainey1_story_9x16":
        if re.search(r"\bstory\b", text) and "storyboard" not in text:
            return True
    if recipe.id == "rainey1_myth_16x9":
        scrubbed = (
            text.replace("9:16", " ")
            .replace("9x16", " ")
            .replace("9×16", " ")
        )
        if any(token in scrubbed for token in ("16:9", "16x9", "16×9")):
            return True
    for cue in recipe.cues:
        if cue in ("16:9", "16x9", "16×9"):
            continue
        if cue and cue in text:
            return True
    return False
