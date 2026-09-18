"""CapabilityContract — ``buddy.capability.contract/v1`` (Rust buddy-core aligned).

Brain-side guarantee for one run: model / family / variant, resolution,
duration, audio, control layers. **No** VRAM, slot, or weight-path fields.

Target schema is Rust ``CapabilityContract`` from sibling your-video-buddy
PR #9 (``docs/BRAIN_HANDS.md``). That repo was **not fetchable** from this
environment (404). Field names below match the PR #9 contract Scott listed.
If the Rust struct lands extra keys, add them without dropping these.
Capacity keys are never serialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from master_agent.models.vram_policy import family_for_slug

CAPABILITY_CONTRACT_SCHEMA = "buddy.capability.contract/v1"
CAPABILITY_CONTRACT_ALIASES = frozenset(
    {
        CAPABILITY_CONTRACT_SCHEMA,
        "buddy.capability.contract",
        "https://buddy.video/schema/capability.contract/v1",
    }
)

# Hands-owned. Must never appear on a contract payload.
CAPACITY_KEYS = frozenset(
    {
        "vram",
        "vram_gb",
        "expected_vram",
        "expected_vram_gb",
        "slot",
        "slots",
        "weight_path",
        "weights_path",
        "weight_paths",
        "models_dir",
        "free_vram",
        "free_vram_gb",
        "peak_vram",
        "vram_free",
        "vram_total",
        "vram_class",
    }
)

_MODEL_FOR_FAMILY = {
    "ltx25": "ltx-2.5",
    "ltx23": "ltx-2.3",
    "wan22": "wan-2.2",
    "h3": "minimax-h3",
    "flux": "flux.1",
    "krea2": "krea-2",
    "vace": "wan-vace",
    "lipsync": "ltx-2.3",
    "movie_builder": "ltx-2.3",
    "ccc": "ccc",
    "qwen_edit": "qwen-image-edit",
    "dataset": "tagger",
    "upscale": "rtx-vsr",
    "zimage": "z-image",
    "air_render": "ltx-2.3",
    "ideogram": "ideogram",
}

_STILL_FAMILIES = frozenset(
    {"flux", "krea2", "qwen_edit", "dataset", "ideogram", "upscale"}
)


def model_for_family(family: str) -> str:
    key = (family or "").strip().lower()
    return _MODEL_FOR_FAMILY.get(key, key or "unknown")


def default_audio_for_family(family: str) -> bool:
    return (family or "").strip().lower() not in _STILL_FAMILIES


def _as_resolution(value: Any, *, width: int = 768, height: int = 512) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, dict):
        w = value.get("width", value.get("w", width))
        h = value.get("height", value.get("h", height))
        return int(w), int(h)
    return int(width), int(height)


def _as_layers(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(x) for x in value if str(x).strip())


def _strip_capacity(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in CAPACITY_KEYS}


@dataclass(frozen=True)
class CapabilityContract:
    """What one run guarantees. Brain writes this; Hands answers fit."""

    model: str
    family: str
    variant: str
    resolution: tuple[int, int] = (768, 512)
    duration_s: float = 8.0
    story_duration_s: float = 8.0
    audio: bool = True
    control_layers: tuple[str, ...] = field(default_factory=tuple)
    schema: str = CAPABILITY_CONTRACT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema or CAPABILITY_CONTRACT_SCHEMA,
            "model": self.model,
            "family": self.family,
            "variant": self.variant,
            "resolution": [int(self.resolution[0]), int(self.resolution[1])],
            "duration_s": float(self.duration_s),
            "story_duration_s": float(self.story_duration_s),
            "audio": bool(self.audio),
            "control_layers": list(self.control_layers),
        }
        return _strip_capacity(payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CapabilityContract":
        raw = _strip_capacity(dict(data or {}))
        schema = str(raw.get("schema") or CAPABILITY_CONTRACT_SCHEMA)
        if schema not in CAPABILITY_CONTRACT_ALIASES and not schema.endswith(
            "capability.contract/v1"
        ):
            schema = CAPABILITY_CONTRACT_SCHEMA
        variant = str(raw.get("variant") or "base")
        family = str(raw.get("family") or family_for_slug(variant))
        model = str(raw.get("model") or model_for_family(family))
        width, height = _as_resolution(raw.get("resolution"))
        duration = float(raw.get("duration_s") or 8.0)
        story = float(raw.get("story_duration_s") or duration)
        audio = raw.get("audio")
        if audio is None:
            audio = default_audio_for_family(family)
        return cls(
            schema=CAPABILITY_CONTRACT_SCHEMA,
            model=model,
            family=family,
            variant=variant,
            resolution=(width, height),
            duration_s=duration,
            story_duration_s=story,
            audio=bool(audio),
            control_layers=_as_layers(raw.get("control_layers")),
        )


def contract_from_variant(
    variant: str,
    *,
    duration_s: float = 8.0,
    story_duration_s: float | None = None,
    width: int = 768,
    height: int = 512,
    audio: Optional[bool] = None,
    control_layers: Iterable[str] | None = None,
) -> CapabilityContract:
    family = family_for_slug(variant)
    return CapabilityContract(
        model=model_for_family(family),
        family=family,
        variant=variant,
        resolution=(int(width), int(height)),
        duration_s=float(duration_s),
        story_duration_s=float(
            story_duration_s if story_duration_s is not None else duration_s
        ),
        audio=default_audio_for_family(family) if audio is None else bool(audio),
        control_layers=tuple(control_layers or ()),
    )
