"""Rainey1 recipe table (Phase 0.1 / 0.2).

TODO(rainey1-preset): Cursor prompt 01 should load these from a ``rainey1``
director preset (keyword ``rainey1``). Until that PR is merged, the constants
below are the source of truth. ``resolve_recipe`` still prefers
``recipe_for()`` from ``master_agent.presets.rainey1`` or
``master_agent.rainey1.preset`` when either module exists.

Frames are snapped with ``snap_ltx_frames`` (8n+1, minimum 9). Never 8.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import Any, Callable

from master_agent.config import (
    DEFAULT_CFG,
    DEFAULT_FPS,
    DOWNSCALE_LADDER,
    is_valid_ltx_frames,
    snap_ltx_frames,
)

# Plan 0.3: reject look below this. Human veto stays on brief_adherence.
LOOK_FLOOR = 0.55

# Style-guide DRAFT §7 anti-slop negative. Preset 01 should own this string.
RAINEY1_NEGATIVE = (
    "identity drift, face morph, twin faces, melted face, extra fingers, "
    "glow soup, rainbow noise, sparse melt, soft beauty filter, plastic skin, "
    "cartoon, anime, lowres, jpeg artifacts, watermark, subtitle, UI overlay, "
    "handheld phone selfie, empty flat lighting, toy MIDI vibe, stock b-roll look, "
    "random scene jump every frame, text burned in, logo burn"
)

# Plan 0.2 additives. Not a face of any real person.
RAINEY1_ADDITIVES: tuple[str, ...] = (
    "emissive multi-light",
    "identity lock",
    "zero slop",
    "music-video hold",
)

# Draft-length practice clips. Tower may raise steps after sec/step exists.
RAINEY1_STEPS = 8

_PRESET_MODULES = (
    "master_agent.presets.rainey1",
    "master_agent.rainey1.preset",
)


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


def _snap_dim(value: int, *, default: int) -> int:
    try:
        raw = int(value)
    except (TypeError, ValueError):
        raw = default
    snapped = max(256, (raw // 32) * 32)
    return snapped


def _ladder_index(width: int, height: int, frames: int) -> int:
    exact = (int(width), int(height), int(frames))
    for index, rung in enumerate(DOWNSCALE_LADDER):
        if rung == exact:
            return index
    for index, (w, h, _frames) in enumerate(DOWNSCALE_LADDER):
        if w == int(width) and h == int(height):
            return index
    return 0


def _fallback_rows() -> dict[str, RaineyRecipe]:
    """Inline recipes. Prompt 01 replaces this table."""
    negative = RAINEY1_NEGATIVE
    additives = RAINEY1_ADDITIVES
    rows = [
        RaineyRecipe(
            id="rainey1_lock_open",
            slug="lock_open",
            # TODO(rainey1-preset): eros distilled pin when prompt 01 selects it.
            variant="base",
            frames=9,
            width=768,
            height=512,
            positive=(
                "photoreal cinematic, locked character, medium shot at cluttered desk, "
                "open PC tower, keyboard hands, high-key window, shallow DOF, "
                "filmic contrast, identity stable, no morph, no text overlay"
            ),
            negative=negative,
            brief=(
                "rainey1 lock open: photoreal locked character at desk, open PC tower, "
                "keyboard, high-key window, identity stable"
            ),
            additives=additives,
            purpose="Identity desk plate I2V",
            source="inline-fallback",
            ladder_index=0,
            notes="Start rung (768, 512, 9). OOM walks DOWNSCALE_LADDER downward.",
        ),
        RaineyRecipe(
            id="rainey1_breach",
            slug="breach",
            variant="base",
            frames=17,
            width=640,
            height=384,
            positive=(
                "same locked character, night desert mesa, metallic saucer overhead, "
                "soft bioluminescent mushrooms, white plasma arcs, deep blacks, "
                "emissive rim light on face, craft owns top third"
            ),
            negative=negative,
            brief=(
                "rainey1 breach: same locked character, night desert mesa, "
                "metallic saucer, emissive rim light"
            ),
            additives=additives,
            purpose="Same character into desert/UFO grammar",
            source="inline-fallback",
            ladder_index=1,
            notes="Ladder rung (640, 384, 17).",
        ),
        RaineyRecipe(
            id="rainey1_density",
            slug="density",
            variant="base",
            frames=25,
            width=512,
            height=384,
            positive=(
                "same locked character looking up, multicolor energy umbilicus from "
                "desk into craft belly, nested neon mycelium, magenta cyan light-trails, "
                "RGB tower glow keys face, maximal detail still readable on phone, "
                "zero slop, HDR highlight roll-off"
            ),
            negative=negative,
            brief=(
                "rainey1 density: locked character looking up, energy umbilicus, "
                "emissive multi-light, zero slop"
            ),
            additives=additives,
            purpose="Climax umbilicus / look-up",
            source="inline-fallback",
            ladder_index=2,
            notes="Ladder rung (512, 384, 25).",
        ),
        RaineyRecipe(
            id="rainey1_myth_16x9",
            slug="myth_16x9",
            variant="base",
            frames=25,
            width=768,
            height=512,
            positive=(
                "cinematic 16:9 music-video plate, myth set-piece, photoreal, "
                "dramatic volumetric light, stable character continuity, "
                "music-video pacing hold, no watermark, no UI chrome"
            ),
            negative=negative,
            brief=(
                "rainey1 myth 16:9: cinematic music-video plate, stable character, "
                "volumetric light"
            ),
            additives=additives,
            purpose="MV set-piece",
            source="inline-fallback",
            ladder_index=0,
            notes=(
                "Start 768x512 at 25 frames. Climb to 33 only after diagnose "
                "recorded sec/step. OOM walks DOWNSCALE_LADDER downward "
                "(next rung 640x384x17), never length 8."
            ),
        ),
        RaineyRecipe(
            id="rainey1_story_9x16",
            slug="story_9x16",
            variant="base",
            frames=17,
            # Patcher clamp is 768x512 (balanced). Queue landscape, crop to 9:16.
            width=768,
            height=512,
            positive=(
                "same locked character, night desert mesa, metallic saucer overhead, "
                "soft bioluminescent mushrooms, white plasma arcs, deep blacks, "
                "emissive rim light on face, 9:16 story frame, craft owns top third"
            ),
            negative=negative,
            brief=(
                "rainey1 story 9:16: locked character, vertical story crop, "
                "desert craft overhead"
            ),
            additives=additives,
            purpose="Vertical story crop",
            source="inline-fallback",
            ladder_index=0,
            notes=(
                "Queue 768x512x17 because the patcher clamp is 768x512. "
                "ffmpeg-crop the keeper to 9:16. Frame count stays 17 (8n+1)."
            ),
        ),
    ]
    table: dict[str, RaineyRecipe] = {}
    for row in rows:
        table[row.slug] = row
        table[row.id] = row
    return table


_FALLBACK = _fallback_rows()


def recipe_slugs() -> list[str]:
    seen: list[str] = []
    for row in _FALLBACK.values():
        if row.slug not in seen:
            seen.append(row.slug)
    return seen


def recipe_choices() -> list[str]:
    slugs = recipe_slugs()
    return slugs + [f"rainey1_{slug}" for slug in slugs]


def rainey1_rubric_file():
    """Path to the Rainey1 judge rubric, or None when prompt 02 is unmerged.

    TODO(rainey1-judge): ``video_buddy/master_agent/judge/prompts/rainey1.md``.
    Until that file exists the batch judge uses the default ``judge.md``.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "judge" / "prompts" / "rainey1.md"
    return path if path.is_file() else None


def _external_preset(recipe_id: str) -> dict[str, Any] | None:
    """Best-effort preset lookup. Missing modules are the expected case today."""
    for name in _PRESET_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        loader = getattr(module, "recipe_for", None)
        if not callable(loader):
            continue
        loaded = loader(recipe_id)
        if isinstance(loaded, dict) and loaded:
            return loaded
    return None


def _apply_preset(base: RaineyRecipe, loaded: dict[str, Any]) -> RaineyRecipe:
    frames = snap_ltx_frames(int(loaded.get("frames", base.frames)))
    width = _snap_dim(int(loaded.get("width", base.width)), default=base.width)
    height = _snap_dim(int(loaded.get("height", base.height)), default=base.height)
    if not is_valid_ltx_frames(frames):
        frames = snap_ltx_frames(frames)
    additives = loaded.get("additives", base.additives)
    if isinstance(additives, str):
        additives = tuple(part.strip() for part in additives.split(",") if part.strip())
    else:
        additives = tuple(str(item) for item in additives)
    positive = str(loaded.get("positive") or loaded.get("prompt") or base.positive)
    negative = str(loaded.get("negative") or loaded.get("negative_prompt") or base.negative)
    variant = str(loaded.get("variant") or base.variant)
    return replace(
        base,
        variant=variant,
        frames=frames,
        width=width,
        height=height,
        positive=positive,
        negative=negative,
        brief=str(loaded.get("brief") or base.brief),
        additives=additives,
        purpose=str(loaded.get("purpose") or base.purpose),
        source="preset",
        ladder_index=_ladder_index(width, height, frames),
        steps=int(loaded.get("steps", base.steps)),
        cfg=float(loaded.get("cfg", base.cfg)),
        notes=str(loaded.get("notes") or base.notes),
    )


def resolve_recipe(
    name: str,
    *,
    loader: Callable[[str], dict[str, Any] | None] | None = None,
) -> RaineyRecipe:
    """Resolve a CLI slug or ``rainey1_*`` id. Frames are always 8n+1."""
    key = (name or "").strip().lower().replace("-", "_")
    if key.startswith("rainey1_"):
        key = key[len("rainey1_") :]
    base = _FALLBACK.get(key) or _FALLBACK.get(name.strip().lower())
    if base is None:
        known = ", ".join(recipe_slugs())
        raise KeyError(f"unknown rainey1 recipe {name!r}; choose {known}")
    if not is_valid_ltx_frames(base.frames):
        base = replace(base, frames=snap_ltx_frames(base.frames))
    fetch = loader if loader is not None else _external_preset
    loaded = fetch(base.id)
    if not loaded:
        return base
    return _apply_preset(base, loaded)
