"""Rainey1 recipes for the batch top-cut loop.

Geometry, prompts, and the anti-slop negative come from the director preset
pack ``orchestrator/presets/rainey1.json`` (``director_presets``). Frame
counts in that pack are already snapped to 8n+1.
"""

from __future__ import annotations

from dataclasses import dataclass

from master_agent.config import (
    DEFAULT_CFG,
    DEFAULT_FPS,
    DOWNSCALE_LADDER,
    is_valid_ltx_frames,
    snap_ltx_frames,
)
from master_agent.judge.rubric import LOOK_RETRY_BELOW
from master_agent.orchestrator.director_presets import (
    PresetError,
    load_rainey1_pack,
    recipe_by_id,
)

# Same gate as the shipped rubric: retry / drop below this look.
LOOK_FLOOR = LOOK_RETRY_BELOW

# Short practice clips. The preset pack does not pin sampler steps.
RAINEY1_STEPS = 8


@dataclass(frozen=True)
class RaineyRecipe:
    id: str
    slug: str
    variant: str
    frames: int
    width: int
    height: int
    positive: str
    negative: str
    brief: str
    additives: tuple[str, ...]
    purpose: str
    source: str
    ladder_index: int
    steps: int = RAINEY1_STEPS
    cfg: float = DEFAULT_CFG
    fps: int = DEFAULT_FPS
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return float(self.frames) / float(self.fps or DEFAULT_FPS)


def _ladder_index(width: int, height: int, frames: int) -> int:
    exact = (int(width), int(height), int(frames))
    for index, rung in enumerate(DOWNSCALE_LADDER):
        if rung == exact:
            return index
    for index, (w, h, _frames) in enumerate(DOWNSCALE_LADDER):
        if w == int(width) and h == int(height):
            return index
    return 0


def _slug(recipe_id: str) -> str:
    prefix = "rainey1_"
    return recipe_id[len(prefix) :] if recipe_id.startswith(prefix) else recipe_id


def recipe_slugs() -> list[str]:
    return [_slug(recipe.id) for recipe in load_rainey1_pack().recipes]


def recipe_choices() -> list[str]:
    slugs = recipe_slugs()
    return slugs + [f"rainey1_{slug}" for slug in slugs]


def resolve_recipe(name: str) -> RaineyRecipe:
    """Resolve a CLI slug or ``rainey1_*`` id from the shipped preset pack."""
    key = (name or "").strip().lower().replace("-", "_")
    recipe_id = key if key.startswith("rainey1_") else f"rainey1_{key}"
    try:
        recipe = recipe_by_id(recipe_id)
    except PresetError as exc:
        known = ", ".join(recipe_slugs())
        raise KeyError(f"unknown rainey1 recipe {name!r}; choose {known}") from exc
    pack = load_rainey1_pack()
    frames = recipe.frames if is_valid_ltx_frames(recipe.frames) else snap_ltx_frames(recipe.frames)
    notes = " ".join(part for part in (recipe.notes, recipe.delivery) if part).strip()
    slug = _slug(recipe.id)
    return RaineyRecipe(
        id=recipe.id,
        slug=slug,
        variant=recipe.variant or pack.variant,
        frames=frames,
        width=int(recipe.width),
        height=int(recipe.height),
        positive=recipe.prompt_pattern,
        negative=pack.negative,
        brief=f"rainey1 {slug.replace('_', ' ')}: {recipe.purpose}",
        additives=pack.additives,
        purpose=recipe.purpose,
        source="preset",
        ladder_index=_ladder_index(recipe.width, recipe.height, frames),
        notes=notes,
    )
