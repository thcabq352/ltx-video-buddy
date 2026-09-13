"""Single 16GB (RTX 5060 Ti class) policy for doctor, catalog, patcher, prepare.

Scott preference: **GGUF first, then NVFP4**. Defaults stay VRAM-safe
(~12–14 GB peak where practical). Higher-quality packs remain accepted
locally; they are not the default pick.

Do not invent weight filenames. Every pack name here is already referenced
in this repo (MODEL_FILES, WEIGHT_FILES, download_models.py, or a shipped
API graph).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

VRAM_CLASS_SAFE = "16gb"
VRAM_CLASS_OFFLOAD = "offload"
VRAM_CLASS_MORE = "needs_more_vram"

# Sentinel used by load_and_patch_workflow when the caller left size at the
# function default (768×512). Family presets may remap that.
DEFAULT_PATCH_WIDTH = 768
DEFAULT_PATCH_HEIGHT = 512


@dataclass(frozen=True)
class VramPreset:
    family: str
    vram_class: str
    default_pack: str
    expected_vram: str
    width: int
    height: int
    steps: int
    cfg: float
    notes: str
    alt_16gb: str = ""
    max_duration_s: float | None = None
    frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Family-level defaults. Variant overrides live in VARIANT_PRESETS.
FAMILY_PRESETS: dict[str, VramPreset] = {
    "ltx25": VramPreset(
        family="ltx25",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="GGUF Q4 (then NVFP4 / int8 / bf16)",
        expected_vram="12–14 GB",
        width=768,
        height=512,
        steps=8,
        cfg=1.0,
        max_duration_s=3.0,
        notes="Doctor pick: GGUF Q4 → NVFP4 (VRAM≥14) → int8-convrot → bf16. Distilled, CFG 1.0.",
        alt_16gb="Diagnose 9 frames first; two-stage uses a shorter clip.",
    ),
    "h3": VramPreset(
        family="h3",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="GGUF Q4_K DiT + Comfy TE (NVFP4/int8)",
        expected_vram="12–14 GB",
        width=1152,
        height=640,
        steps=4,
        cfg=1.0,
        max_duration_s=12.0,
        frames=124,
        notes="CFG stays 1.0. 0.6–0.8 MP, ≤12 s, 4-step turbo LoRA when present.",
    ),
    "ltx23": VramPreset(
        family="ltx23",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="LTX 2.3 distilled AIO / fp8 (EROS baked)",
        expected_vram="12–14 GB",
        width=768,
        height=512,
        steps=12,
        cfg=1.0,
        max_duration_s=6.0,
        notes="Rainey path. 8n+1 frames, DOWNSCALE_LADDER. Diagnose 9-frame hull first.",
        alt_16gb="GGUF LTX-2.3-dev-Q4_K_S if swapping the Movie Builder / split UNET.",
    ),
    "lipsync": VramPreset(
        family="lipsync",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="ltx-2.3-22b-dev-fp8 + LipDub IC-LoRA",
        expected_vram="12–14 GB",
        width=512,
        height=512,
        steps=12,
        cfg=1.0,
        notes="Latent 512². Source video attached. Distilled LoRA keeps steps low.",
    ),
    "wan": VramPreset(
        family="wan",
        vram_class=VRAM_CLASS_OFFLOAD,
        default_pack="fp8 dual 14B + LightX2V distill LoRA",
        expected_vram="14–16 GB (sequential / --normalvram)",
        width=640,
        height=384,
        steps=8,
        cfg=1.0,
        max_duration_s=3.0,
        notes=(
            "No public Wan 2.2 T2V GGUF is catalogued here — do not invent one. "
            "Dual 14B fp8 at 1280×720/81f will OOM. 16GB path: 640×384, ≤33f @16fps, "
            "8 steps, LightX2V LoRA (already in the graph). K3NK AIO I2V is not shipped."
        ),
        alt_16gb="Raise to 768×512 / 49f only after diagnose sec/step; else offload / lowvram.",
    ),
    "vace": VramPreset(
        family="vace",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="VACE Skyreels GGUF Q4_K_M (then e4m3fn)",
        expected_vram="12–14 GB",
        width=720,
        height=720,
        steps=5,
        cfg=1.0,
        notes="Prefer mickmumpitz VACE GGUF Q4_K_M over the e4m3fn 14B. FusionX LoRA stays.",
        alt_16gb="e4m3fn is the high-VRAM / quality pack — not the 16GB default.",
    ),
    "image": VramPreset(
        family="image",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="fp8 / NVFP4 stills (Flux, Krea turbo)",
        expected_vram="10–13 GB",
        width=1024,
        height=1024,
        steps=20,
        cfg=1.0,
        notes="Flux.1-dev fp8 or Krea-2 turbo NVFP4. Not dual-UNET video.",
        alt_16gb="Krea 1920×1080 needs more headroom — default 1024×576.",
    ),
    "qwen": VramPreset(
        family="qwen",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="Qwen-Image-Edit GGUF Q5_0 + Lightning 4-step",
        expected_vram="10–13 GB",
        width=1024,
        height=512,
        steps=4,
        cfg=1.0,
        notes="Start-image / 360 graphs already ship GGUF + Lightning LoRAs.",
        alt_16gb="vb_qwen_edit_360 baked 2048×1024 is the 360 plate — use 1024 on 16GB.",
    ),
    "movie": VramPreset(
        family="movie",
        vram_class=VRAM_CLASS_MORE,
        default_pack="Flux 2 Klein fp8 + LTX 2.3 fp8 (GGUF Q4_K_S alternate)",
        expected_vram=">16 GB / sequential offload",
        width=768,
        height=512,
        steps=4,
        cfg=1.0,
        notes=(
            "189-node Movie Builder cannot be a silent 16GB default. "
            "Swap LTX UNET to LTX-2.3-dev-Q4_K_S.gguf and keep start-frames ≤1024."
        ),
        alt_16gb="Prefer --template after swapping GGUF; or route shots through base/ltx25.",
    ),
    "ccc": VramPreset(
        family="ccc",
        vram_class=VRAM_CLASS_MORE,
        default_pack="Flux 2 Klein fp8 / Krea NVFP4 (still a 300–777 node graph)",
        expected_vram=">16 GB if SeedVR2 + detailer + many passes",
        width=1024,
        height=1024,
        steps=8,
        cfg=1.0,
        notes="CCC ADV / 4.1 queue baked widgets. Disable SeedVR2 + Face Detailer on 16GB.",
        alt_16gb="Use flux / krea2_img for single sheets; keep CCC as --template.",
    ),
    "other": VramPreset(
        family="other",
        vram_class=VRAM_CLASS_OFFLOAD,
        default_pack="see notes",
        expected_vram="varies",
        width=768,
        height=512,
        steps=12,
        cfg=1.0,
        notes="Example / gitignored / preprocess graphs. Do not claim they fit 16GB.",
        alt_16gb="Prefer --template and inspect loaders before queueing.",
    ),
}

# Per-variant overrides (optional). Missing slugs inherit FAMILY_PRESETS[family].
VARIANT_PRESETS: dict[str, VramPreset] = {
    "ltx25_t2v_i2v_two_stage": VramPreset(
        family="ltx25",
        vram_class=VRAM_CLASS_OFFLOAD,
        default_pack="GGUF Q4 + spatial upscaler",
        expected_vram="14–16 GB",
        width=768,
        height=512,
        steps=6,
        cfg=1.0,
        max_duration_s=2.0,
        notes="Two-stage latent upscale is the 16GB offload LTX 2.5 path, not the default clip.",
        alt_16gb="Use ltx25_t2v_i2v (single-stage) on 16GB unless you have headroom.",
    ),
    "wan22": FAMILY_PRESETS["wan"],
    "vb_wan22_vid": FAMILY_PRESETS["wan"],
    "krea2_img": VramPreset(
        family="image",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="krea2_turbo_nvfp4 + qwen3vl_4b_fp8",
        expected_vram="10–12 GB",
        width=1024,
        height=576,
        steps=10,
        cfg=1.0,
        notes="NVFP4 is the 5060 Ti default. fp8 and bf16 remain accepted.",
        alt_16gb="1920×1080 is the quality pack — not the 16GB default.",
    ),
    "vb_krea2_img": VramPreset(
        family="image",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="krea2_turbo_nvfp4 + qwen3vl_4b_fp8",
        expected_vram="10–12 GB",
        width=1024,
        height=576,
        steps=10,
        cfg=1.0,
        notes="Same graph as krea2_img.",
    ),
    "flux": VramPreset(
        family="image",
        vram_class=VRAM_CLASS_SAFE,
        default_pack="flux1-dev-fp8 + t5xxl_fp8",
        expected_vram="10–13 GB",
        width=1024,
        height=1024,
        steps=20,
        cfg=1.0,
        notes="Non-gated fp8 set. Full bf16 Flux is the high-VRAM option.",
    ),
    "vb_qwen_edit_360": VramPreset(
        family="qwen",
        vram_class=VRAM_CLASS_OFFLOAD,
        default_pack="Qwen-Image-Edit-2509-Q5_0 GGUF + Lightning",
        expected_vram="14–16 GB at 2048×1024",
        width=1024,
        height=512,
        steps=4,
        cfg=1.0,
        notes="Baked 2048×1024 360 plate. On 16GB run 1024×512 (or the ImageResizeKJv2 1024 path).",
        alt_16gb="Keep GGUF Q5_0; do not load a full bf16 Qwen UNET.",
    ),
    "vb_aivfx_startimage": FAMILY_PRESETS["qwen"],
    "vb_aivfx_adv": FAMILY_PRESETS["vace"],
    "vb_aivfx_adv_13": FAMILY_PRESETS["vace"],
    "vb_aivfx_preprocess": VramPreset(
        family="vace",
        vram_class=VRAM_CLASS_OFFLOAD,
        default_pack="SAM3 + DepthCrafter + CoTracker (no DiT)",
        expected_vram="10–16 GB (SAM3 spike)",
        width=720,
        height=720,
        steps=1,
        cfg=1.0,
        notes="Preprocess only. First SAM3 run downloads sam3.pt. Not a dual-UNET generate.",
        alt_16gb="Run at 720²; do not raise to 1080 on 16GB.",
    ),
    "vb_movie_builder": FAMILY_PRESETS["movie"],
    "vb_ccc_adv": FAMILY_PRESETS["ccc"],
    "vb_ccc41_krea2": VramPreset(
        family="ccc",
        vram_class=VRAM_CLASS_MORE,
        default_pack="krea2_turbo_nvfp4 (was fp8) + identity LoRA",
        expected_vram=">16 GB if all 48 grounded encodes run",
        width=1024,
        height=1024,
        steps=8,
        cfg=1.0,
        notes="NVFP4 is the 16GB loader; the 321-node graph still needs offload / fewer passes.",
        alt_16gb="Prefer krea2_img for a single edit; --template for CCC 4.1.",
    ),
    "k3nk_wan_aio": VramPreset(
        family="wan",
        vram_class=VRAM_CLASS_MORE,
        default_pack="not shipped (K3NK AIO I2V HIGH/LOW fp8 — tower hypothesis)",
        expected_vram="unknown — no graph",
        width=640,
        height=384,
        steps=8,
        cfg=1.0,
        notes="AUDIT: not in MODEL_FILES. Do not spray onto wan22 T2V UNETs. No invented filenames.",
        alt_16gb="Use wan22 T2V LightX2V path or wait for a real AIO I2V template.",
    ),
}

# Slug → family when not in VARIANT_PRESETS and catalog.family is missing.
_SLUG_FAMILY: dict[str, str] = {
    "base": "ltx23",
    "eros": "ltx23",
    "directors": "ltx23",
    "lipsync": "lipsync",
    "ltx23_lipsync_v08": "lipsync",
    "wan22": "wan",
    "vb_wan22_vid": "wan",
    "flux": "image",
    "krea2_img": "image",
    "vb_krea2_img": "image",
    "vb_ideogram": "image",
    "vb_movie_builder": "movie",
    "vb_ccc_adv": "ccc",
    "vb_ccc41_krea2": "ccc",
    "vb_qwen_edit_360": "qwen",
    "vb_aivfx_startimage": "qwen",
    "vb_aivfx_adv": "vace",
    "vb_aivfx_adv_13": "vace",
    "vb_aivfx_preprocess": "vace",
    "vb_dataset_tagger": "other",
    "vb_tag_review": "other",
    "vb_ai_renderer_smpl": "vace",
    "vb_ai_renderer_adv": "vace",
    "vb_ai_renderer_adv_20": "vace",
    "vb_zimage_turbo_cn": "other",
    "vb_rtx_superres": "other",
    "air_render_030": "other",
    "air_render_050": "other",
    "air_render_businesswoman": "other",
}


def _canonical_variant(variant: str | None) -> str:
    key = (variant or "").strip()
    if not key:
        return ""
    try:
        from master_agent.comfy.catalog import H3_ALIASES, RESEARCH_ALIASES

        return H3_ALIASES.get(key, RESEARCH_ALIASES.get(key, key))
    except Exception:
        return key


def family_for_variant(variant: str | None) -> str:
    key = _canonical_variant(variant)
    if key in VARIANT_PRESETS:
        return VARIANT_PRESETS[key].family
    if key in _SLUG_FAMILY:
        return _SLUG_FAMILY[key]
    lowered = key.lower().replace("\\", "/")
    if key.startswith("ltx25") or "ltx-2.5" in lowered:
        return "ltx25"
    if key.startswith("h3") or "minimax" in lowered or "fl2va" in lowered or "ref2va" in lowered:
        return "h3"
    if "wan" in lowered:
        return "wan"
    if "vace" in lowered or "aivfx" in lowered or "vfx" in lowered:
        return "vace"
    if "movie" in lowered:
        return "movie"
    if "ccc" in lowered:
        return "ccc"
    if "qwen" in lowered:
        return "qwen"
    if "flux" in lowered or "krea" in lowered:
        return "image"
    if "lip" in lowered:
        return "lipsync"
    if "ltx" in lowered:
        return "ltx23"
    try:
        from master_agent.comfy.catalog import catalog_by_id

        entry = catalog_by_id().get(key)
        if entry is not None and entry.family:
            return entry.family
    except Exception:
        pass
    return "other"


def preset_for_variant(variant: str | None) -> VramPreset:
    key = _canonical_variant(variant)
    if key in VARIANT_PRESETS:
        return VARIANT_PRESETS[key]
    return FAMILY_PRESETS.get(family_for_variant(key), FAMILY_PRESETS["other"])


def uses_default_patch_size(width: int, height: int) -> bool:
    return int(width) == DEFAULT_PATCH_WIDTH and int(height) == DEFAULT_PATCH_HEIGHT


def apply_16gb_size(variant: str | None, width: int, height: int) -> tuple[int, int]:
    """Remap the patcher 768×512 default to the family 16GB size when appropriate."""
    preset = preset_for_variant(variant)
    if uses_default_patch_size(width, height) and (preset.width, preset.height) != (
        DEFAULT_PATCH_WIDTH,
        DEFAULT_PATCH_HEIGHT,
    ):
        return int(preset.width), int(preset.height)
    return int(width), int(height)


def max_duration_for_variant(variant: str | None) -> float | None:
    return preset_for_variant(variant).max_duration_s


def default_steps_cfg(variant: str | None) -> tuple[int, float]:
    preset = preset_for_variant(variant)
    return int(preset.steps), float(preset.cfg)


def vram_warnings(variant: str | None, *, mode: str = "generate") -> list[str]:
    """`--prepare` / doctor notes. Empty when the default path is 16GB-safe."""
    preset = preset_for_variant(variant)
    warnings: list[str] = []
    label = _canonical_variant(variant) or (variant or "workflow")
    if preset.vram_class == VRAM_CLASS_MORE:
        warnings.append(
            f"vram: {label} is labeled needs_more_vram / manual offload "
            f"(expected {preset.expected_vram}). 16GB alternate: {preset.alt_16gb or preset.notes}"
        )
    elif preset.vram_class == VRAM_CLASS_OFFLOAD:
        warnings.append(
            f"vram: {label} fits 16GB only with the offload path "
            f"({preset.default_pack}; {preset.width}×{preset.height}, {preset.steps} steps). "
            f"{preset.alt_16gb or preset.notes}"
        )
    if mode == "template" and preset.vram_class != VRAM_CLASS_SAFE:
        warnings.append(
            f"vram: --template uses baked widgets. Review loaders before queueing on a 16GB card."
        )
    return warnings


def policy_rows(variants: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Workflow family → default pack → expected VRAM → notes (PR / doctor table)."""
    if variants is None:
        variants = _default_table_slugs()
    seen_families: set[str] = set()
    rows: list[dict[str, Any]] = []
    for vid in variants:
        preset = preset_for_variant(vid)
        key = f"{preset.family}:{preset.default_pack}:{preset.vram_class}"
        # One row per distinct family/pack, plus important variant overrides.
        if preset.family in seen_families and vid not in VARIANT_PRESETS:
            continue
        seen_families.add(preset.family)
        rows.append(
            {
                "variant": vid,
                "family": preset.family,
                "vram_class": preset.vram_class,
                "default_pack": preset.default_pack,
                "expected_vram": preset.expected_vram,
                "default_size": f"{preset.width}×{preset.height}",
                "steps": preset.steps,
                "cfg": preset.cfg,
                "notes": preset.notes,
                "alt_16gb": preset.alt_16gb,
            }
        )
    return rows


def _default_table_slugs() -> list[str]:
    return [
        "ltx25_t2v_i2v",
        "ltx25_t2v_i2v_two_stage",
        "h3_t2v",
        "base",
        "lipsync",
        "wan22",
        "vb_aivfx_adv_13",
        "krea2_img",
        "flux",
        "vb_qwen_edit_360",
        "vb_movie_builder",
        "vb_ccc_adv",
        "vb_ccc41_krea2",
        "k3nk_wan_aio",
    ]


def format_policy_table(rows: list[dict[str, Any]] | None = None) -> str:
    data = rows if rows is not None else policy_rows()
    lines = [
        "16GB VRAM policy  (RTX 5060 Ti class — GGUF first, then NVFP4)",
        "",
        f"{'Family / variant':<28} {'Class':<16} {'Peak':<18} Default pack",
        "-" * 100,
    ]
    for row in data:
        label = f"{row['family']} ({row['variant']})"
        lines.append(
            f"{label:<28} {row['vram_class']:<16} {row['expected_vram']:<18} {row['default_pack']}"
        )
    lines.append("")
    lines.append("Higher-quality bf16 / dual-UNET packs stay accepted; they are not defaults.")
    return "\n".join(lines) + "\n"


def catalog_vram_fields(variant: str | None) -> dict[str, str]:
    preset = preset_for_variant(variant)
    return {
        "vram_class": preset.vram_class,
        "default_pack": preset.default_pack,
        "expected_vram": preset.expected_vram,
        "vram_note": preset.alt_16gb or preset.notes,
    }


__all__ = [
    "DEFAULT_PATCH_HEIGHT",
    "DEFAULT_PATCH_WIDTH",
    "FAMILY_PRESETS",
    "VRAM_CLASS_MORE",
    "VRAM_CLASS_OFFLOAD",
    "VRAM_CLASS_SAFE",
    "VARIANT_PRESETS",
    "VramPreset",
    "apply_16gb_size",
    "catalog_vram_fields",
    "default_steps_cfg",
    "family_for_variant",
    "format_policy_table",
    "max_duration_for_variant",
    "policy_rows",
    "preset_for_variant",
    "uses_default_patch_size",
    "vram_warnings",
]
