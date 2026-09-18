"""Single 16GB-class pack policy (RTX 5060 Ti).

Director-routed workflows inherit this. Doctor, download-models, the cost
gate, and loader remaps all read from here.

Do not invent weight files. Every filename is attested in
``master_agent.models.weights``, ``state/download_models.py``,
``config.MODEL_FILES``, or a Hugging Face listing verified this change
(QuantStack Wan2.2 / LTX-2.3 GGUF, city96 FLUX.1-dev-gguf, Comfy-Org/Krea-2).
K3NK AIO I2V packs were searched and are **not** attested — they stay
off the default table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

TARGET_GPU = "NVIDIA RTX 5060 Ti"
TARGET_VRAM_GB = 16.0
NVFP4_MIN_VRAM_GB = 14.0
PEAK_HEADROOM_GB = (12.0, 14.0)

# Quantization rank: lower is the 16GB pick.
_KIND_RANK = {
    "gguf_q4": 0,
    "gguf_q5": 1,
    "gguf": 2,
    "nvfp4": 3,
    "int4": 4,
    "int8": 5,
    "fp8": 6,
    "fp16": 7,
    "bf16": 8,
    "other": 9,
}


def pack_kind(filename: str) -> str:
    name = (filename or "").lower().replace("\\", "/")
    base = name.rsplit("/", 1)[-1]
    if base.endswith(".gguf"):
        if "q4" in base:
            return "gguf_q4"
        if "q5" in base:
            return "gguf_q5"
        return "gguf"
    if "nvfp4" in base:
        return "nvfp4"
    if "int4" in base:
        return "int4"
    if "int8" in base:
        return "int8"
    if "fp8" in base or "e4m3" in base:
        return "fp8"
    if "fp16" in base:
        return "fp16"
    if "bf16" in base:
        return "bf16"
    return "other"


def preference_order(
    names: Iterable[str],
    *,
    vram_gb: float | None = None,
) -> tuple[str, ...]:
    """GGUF Q4/Q5 → NVFP4 (if VRAM_GB ≥ 14) → int8/int4 → fp8 → fp16/bf16."""
    from master_agent.config import VRAM_GB

    gb = float(VRAM_GB if vram_gb is None else vram_gb)
    seen: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    if gb < NVFP4_MIN_VRAM_GB:
        seen = [n for n in seen if pack_kind(n) != "nvfp4"] + [
            n for n in seen if pack_kind(n) == "nvfp4"
        ]
        ranked = [n for n in seen if pack_kind(n) != "nvfp4"]
        return tuple(sorted(ranked, key=lambda n: (_KIND_RANK.get(pack_kind(n), 9), n)))
    return tuple(sorted(seen, key=lambda n: (_KIND_RANK.get(pack_kind(n), 9), n)))


def describe_pack(filename: str | None) -> str:
    if not filename:
        return "no local pack"
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    kind = pack_kind(name)
    labels = {
        "gguf_q4": "GGUF Q4 — 16GB-class preference #1",
        "gguf_q5": "GGUF Q5 — 16GB-class preference #1",
        "gguf": "GGUF — 16GB-class preference #1",
        "nvfp4": "NVFP4 — 16GB-class preference #2",
        "int8": "int8 — 16GB-class preference #3",
        "int4": "int4 — 16GB-class preference #3",
        "fp8": "fp8 — fallback (still 16GB-friendly)",
        "bf16": "bf16 — high-VRAM fallback (not a 16GB default)",
        "fp16": "fp16 — high-VRAM fallback (not a 16GB default)",
    }
    return f"{name} ({labels.get(kind, kind)})"


# --- attested filenames (no inventions) ------------------------------------

# LTX 2.5 / H3 (weights.py)
_LTX25 = (
    "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf",
    "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
)
_H3 = (
    "minimax_h3_fl2va_pruned-Q4_K.gguf",
    "minimax_h3_ref2va_pruned-Q4_K.gguf",
    "minimax_h3_fl2va_pruned_nvfp4.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
)

# config.MODEL_FILES + state/download_models.py
_LTX23 = (
    "LTX2.3_DISTILLED-1.1_BAKED_LTX_10Eros_v14_r768.safetensors",
    "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
    "ltx-2.3-22b-dev-fp8.safetensors",
    "ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
    "LTX-2.3-dev-Q4_K_S.gguf",
    "LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf",
    "gemma_3_12B_it_fp4_mixed.safetensors",
    "gemma_3_12B_it_fp8_scaled.safetensors",
)
_WAN = (
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    "Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf",
    "Wan2.2-T2V-A14B-LowNoise-Q4_K_S.gguf",
    "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
    "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
    "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "wan_2.1_vae.safetensors",
)
_VACE = (
    "wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1-Q4_K_M.gguf",
    "wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors",
)
_QWEN = (
    "Qwen-Image-Edit-2509-Q5_0.gguf",
    "qwen-image-edit-2511-Q5_0.gguf",
    "Qwen-Image-Lightning-4steps-V2.0.safetensors",
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
)
_KREA = (
    "krea2_turbo_nvfp4.safetensors",
    "krea2_turbo_int8_convrot.safetensors",
    "krea2_turbo_fp8_scaled.safetensors",
    "krea2_turbo_bf16.safetensors",
    "qwen3vl_4b_fp8_scaled.safetensors",
)
_FLUX = (
    "flux1-dev-Q4_K_S.gguf",
    "flux1-dev-Q4_0.gguf",
    "flux1-dev-fp8.safetensors",
    "flux-2-klein-9b-fp8.safetensors",
    "t5xxl_fp8_e4m3fn.safetensors",
    "clip_l.safetensors",
    "ae.safetensors",
)
_OTHER = (
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
    "z_image_turbo_bf16.safetensors",
)

ATTESTED_FILENAMES: frozenset[str] = frozenset(
    (*_LTX25, *_H3, *_LTX23, *_WAN, *_VACE, *_QWEN, *_KREA, *_FLUX, *_OTHER)
)

WAN22_HIGH_PREFERENCE: tuple[str, ...] = (
    "Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf",
    "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
)
WAN22_LOW_PREFERENCE: tuple[str, ...] = (
    "Wan2.2-T2V-A14B-LowNoise-Q4_K_S.gguf",
    "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
)
VACE_PREFERENCE: tuple[str, ...] = _VACE
KREA_PREFERENCE: tuple[str, ...] = _KREA[:4]
FLUX_PREFERENCE: tuple[str, ...] = (
    "flux1-dev-Q4_K_S.gguf",
    "flux1-dev-Q4_0.gguf",
    "flux1-dev-fp8.safetensors",
)
QWEN_EDIT_PREFERENCE: tuple[str, ...] = (
    "Qwen-Image-Edit-2509-Q5_0.gguf",
    "qwen-image-edit-2511-Q5_0.gguf",
)
LTX23_PREFERENCE: tuple[str, ...] = (
    "LTX2.3_DISTILLED-1.1_BAKED_LTX_10Eros_v14_r768.safetensors",
    "LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf",
    "LTX-2.3-dev-Q4_K_S.gguf",
    "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
    "ltx-2.3-22b-dev-fp8.safetensors",
)


@dataclass(frozen=True)
class WorkflowVramRow:
    slug: str
    family: str
    default_pack: str
    expected_vram_gb: float
    vram_class: str
    safer_alternate: str = ""
    prepare_warning: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# Families that cannot be 16GB defaults (keep available; warn + safer path).
HEAVY_SLUGS: frozenset[str] = frozenset(
    {
        "vb_movie_builder",
        "vb_ccc_adv",
        "vb_ccc41_krea2",
        "vb_aivfx_adv",
        "vb_aivfx_preprocess",
        "ltx23_lipsync_v08",
    }
)

_FAMILY_BY_PREFIX: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ltx25", "ltx-2.5", "t2v_i2v", "flf2v", "msr", "v2v_ic", "a2v", "t2a"), "ltx25"),
    (("h3", "fl2va", "ref2va", "minimax"), "h3"),
    (("wan22", "vb_wan22", "wan"), "wan22"),
    (("vb_aivfx", "vace", "vb_ai_renderer"), "vace"),
    (("krea",), "krea2"),
    (("flux",), "flux"),
    (("qwen", "vb_aivfx_startimage"), "qwen_edit"),
    (("lipsync", "ltx23_lip"), "lipsync"),
    (("movie",), "movie_builder"),
    (("ccc",), "ccc"),
    (("ideogram",), "ideogram"),
    (("tagger", "tag_review", "tag-review"), "dataset"),
    (("rtx", "superres"), "upscale"),
    (("zimage", "z-image"), "zimage"),
    (("air_render",), "air_render"),
    (("base", "eros", "directors"), "ltx23"),
)


def family_for_slug(slug: str) -> str:
    key = (slug or "").strip().lower().replace("\\", "/")
    if key in {
        "base",
        "eros",
        "directors",
    }:
        return "ltx23"
    if key in {"lipsync"}:
        return "lipsync"
    if key in {"flux"}:
        return "flux"
    if key in {"krea2_img", "vb_krea2_img"}:
        return "krea2"
    if key in {"wan22", "vb_wan22_vid"}:
        return "wan22"
    if key.startswith("ltx25") or key in {
        "t2v_i2v",
        "t2v_i2v_two_stage",
        "flf2v",
        "msr",
        "v2v_ic_lora",
        "a2v",
        "t2a",
    }:
        return "ltx25"
    if key.startswith("h3") or key in {"fl2va", "ref2va", "minimax", "minimax_h3"}:
        return "h3"
    if "movie" in key:
        return "movie_builder"
    if "ccc" in key:
        return "ccc"
    if "qwen" in key or key == "vb_aivfx_startimage":
        return "qwen_edit"
    if key.startswith("vb_aivfx") or "renderer" in key:
        return "vace"
    if "ideogram" in key:
        return "ideogram"
    if "tag" in key:
        return "dataset"
    if "rtx" in key or "superres" in key:
        return "upscale"
    if "zimage" in key or "z-image" in key:
        return "zimage"
    if key.startswith("air_"):
        return "air_render"
    if "lip" in key:
        return "lipsync"
    if "wan" in key:
        return "wan22"
    if "ltx" in key:
        return "ltx23"
    return "other"


def _warn_heavy(kind: str, alt: str) -> str:
    return (
        f"16GB WARNING: {kind} is not a 16GB default on an RTX 5060 Ti. "
        f"Expect >14GB peak or a prepare-only graph. Safer alternate: `{alt}`. "
        "Use `comfy run --template` if you really want the full graph."
    )


def _build_rows() -> dict[str, WorkflowVramRow]:
    rows: dict[str, WorkflowVramRow] = {}

    def add(row: WorkflowVramRow) -> None:
        rows[row.slug] = row

    ltx25_notes = (
        "GGUF Q4 → NVFP4 (VRAM≥14) → int8 → bf16. welltop-cn TeaCache "
        "inject-when-registered (soft-bypass if missing). DOWNSCALE_LADDER 9/17/25/33."
    )
    for slug, pack, vram, klass, notes in (
        ("ltx25_t2v_i2v", _LTX25[0], 12.5, "safe", ltx25_notes),
        ("ltx25_flf2v", _LTX25[0], 12.8, "safe", ltx25_notes),
        ("ltx25_msr", _LTX25[0], 13.2, "tight", ltx25_notes + " Multi-ref adds image pressure."),
        ("ltx25_v2v_ic_lora", _LTX25[0], 13.0, "tight", ltx25_notes),
        ("ltx25_a2v", _LTX25[0], 12.8, "safe", ltx25_notes),
        ("ltx25_t2a", _LTX25[0], 8.5, "safe", "Audio-only; still prefers GGUF transformer if wired."),
        (
            "ltx25_t2v_i2v_two_stage",
            _LTX25[0],
            14.0,
            "tight",
            ltx25_notes + " Spatial upscale is the quality path — not the 16GB default.",
        ),
    ):
        add(
            WorkflowVramRow(
                slug=slug,
                family="ltx25",
                default_pack=pack,
                expected_vram_gb=vram,
                vram_class=klass,
                safer_alternate="ltx25_t2v_i2v" if slug == "ltx25_t2v_i2v_two_stage" else "",
                notes=notes,
            )
        )

    h3_notes = (
        "GGUF Q4_K DiT + Comfy Qwen3-VL TE (NVFP4 AWQ / int8). "
        "CFG 1.0, 4 steps, ≤12s, ~0.8 MP. Turbo / LightX2V LoRA when present."
    )
    for slug, pack, vram in (
        ("h3_t2v", _H3[0], 13.0),
        ("h3_i2v", _H3[0], 13.0),
        ("h3_flf", _H3[0], 13.2),
        ("h3_r2v", _H3[1], 13.4),
    ):
        add(
            WorkflowVramRow(
                slug=slug,
                family="h3",
                default_pack=pack,
                expected_vram_gb=vram,
                vram_class="tight",
                notes=h3_notes,
            )
        )

    ltx23_notes = (
        "EROS baked all-in-one is the proven 16GB default. "
        "QuantStack LTX-2.3 GGUF Q4_K_S (~16.7 GB) is optional / tight. "
        "Official bf16/dev is not a default. DOWNSCALE_LADDER + distilled LoRA."
    )
    for slug, vram, klass in (
        ("base", 9.5, "safe"),
        ("eros", 11.0, "safe"),
        ("directors", 13.5, "tight"),
    ):
        add(
            WorkflowVramRow(
                slug=slug,
                family="ltx23",
                default_pack=_LTX23[0],
                expected_vram_gb=vram,
                vram_class=klass,
                notes=ltx23_notes,
            )
        )

    wan_notes = (
        "16GB path: sequential high/low UNET + Lightx2v distill LoRA "
        "(Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32). "
        "Prefer QuantStack GGUF Q4_K_S when present; Comfy-Org fp8 is the "
        "on-disk fallback. Dual bf16 is not a default. Wan TeaCache is "
        "bypass-only (do not inject onto native UNET→KSampler)."
    )
    add(
        WorkflowVramRow(
            slug="wan22",
            family="wan22",
            default_pack=WAN22_HIGH_PREFERENCE[0],
            expected_vram_gb=13.2,
            vram_class="tight",
            notes=wan_notes,
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_wan22_vid",
            family="wan22",
            default_pack=WAN22_HIGH_PREFERENCE[0],
            expected_vram_gb=13.2,
            vram_class="tight",
            notes=wan_notes,
        )
    )

    add(
        WorkflowVramRow(
            slug="vb_aivfx_adv_13",
            family="vace",
            default_pack=_VACE[0],
            expected_vram_gb=13.6,
            vram_class="tight",
            notes="VACE Skyreels Q4_K_M GGUF (mickmumpitz/VACE_Skyreels_V3_R2V_Merge-GGUF). e4m3fn is the quality fallback.",
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_aivfx_adv",
            family="vace",
            default_pack=_VACE[1],
            expected_vram_gb=15.2,
            vram_class="heavy",
            safer_alternate="vb_aivfx_adv_13",
            prepare_warning=_warn_heavy("AI-VFX 1.0 (VACE e4m3fn, no public fp32)", "vb_aivfx_adv_13"),
            notes="v1.0 ships e4m3fn only. 16GB default is v1.3 GGUF.",
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_aivfx_preprocess",
            family="vace",
            default_pack="sam3.pt",
            expected_vram_gb=15.0,
            vram_class="heavy",
            safer_alternate="vb_aivfx_startimage",
            prepare_warning=_warn_heavy("AI-VFX preprocess (SAM3 + DepthCrafter + CoTracker)", "vb_aivfx_startimage"),
            notes="Control-video preprocess. First run downloads sam3.pt. Not a gen default.",
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_aivfx_startimage",
            family="qwen_edit",
            default_pack=_QWEN[1],
            expected_vram_gb=11.5,
            vram_class="safe",
            notes="Qwen-Image-Edit-2511 GGUF Q5_0 + Lightning LoRA.",
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_qwen_edit_360",
            family="qwen_edit",
            default_pack=_QWEN[0],
            expected_vram_gb=12.0,
            vram_class="safe",
            notes="Qwen-Image-Edit-2509 GGUF Q5_0 + Lightning + 360 LoRA.",
        )
    )

    krea_notes = "Comfy-Org Krea-2 turbo: NVFP4 (Blackwell) → int8 → fp8. bf16 is not a default."
    add(
        WorkflowVramRow(
            slug="krea2_img",
            family="krea2",
            default_pack=_KREA[0],
            expected_vram_gb=10.5,
            vram_class="safe",
            notes=krea_notes,
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_krea2_img",
            family="krea2",
            default_pack=_KREA[0],
            expected_vram_gb=10.5,
            vram_class="safe",
            notes=krea_notes,
        )
    )
    add(
        WorkflowVramRow(
            slug="flux",
            family="flux",
            default_pack=_FLUX[0],
            expected_vram_gb=11.0,
            vram_class="safe",
            notes="city96 flux1-dev-Q4_K_S.gguf when present; Comfy-Org flux1-dev-fp8 is the download fallback.",
        )
    )

    add(
        WorkflowVramRow(
            slug="lipsync",
            family="lipsync",
            default_pack="ltx-2.3-22b-dev-fp8.safetensors",
            expected_vram_gb=10.5,
            vram_class="safe",
            notes="LipDub IC-LoRA on LTX 2.3 fp8. DOWNSCALE_LADDER applies.",
        )
    )
    add(
        WorkflowVramRow(
            slug="ltx23_lipsync_v08",
            family="lipsync",
            default_pack="ltx-2.3-22b-dev-fp8.safetensors",
            expected_vram_gb=15.4,
            vram_class="heavy",
            safer_alternate="lipsync",
            prepare_warning=_warn_heavy("LTX-2.3 3D-rendering lip-sync (clay/depth/OmniNFT)", "lipsync"),
            notes="ADV clay/depth/mouth graph. Safer path is the small LipDub template.",
        )
    )

    add(
        WorkflowVramRow(
            slug="vb_movie_builder",
            family="movie_builder",
            default_pack="flux-2-klein-9b-fp8.safetensors",
            expected_vram_gb=15.8,
            vram_class="heavy",
            safer_alternate="ltx25_t2v_i2v",
            prepare_warning=_warn_heavy(
                "Movie Builder ADV (Flux Klein start-frames + LTX 2.3 + voice + 360)",
                "ltx25_t2v_i2v",
            ),
            notes="Prefer LTX-2.3-dev-Q4_K_S.gguf on the video loader if present. Not a 16GB default.",
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_ccc_adv",
            family="ccc",
            default_pack="flux-2-klein-9b-fp8.safetensors",
            expected_vram_gb=15.5,
            vram_class="heavy",
            safer_alternate="flux",
            prepare_warning=_warn_heavy("CCC 4.01 ADV (516+ nodes, SeedVR2, face detailer)", "flux"),
            notes="SeedVR2 already Q4_K_M GGUF. Still too wide for a 16GB default.",
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_ccc41_krea2",
            family="ccc",
            default_pack=_KREA[0],
            expected_vram_gb=15.3,
            vram_class="heavy",
            safer_alternate="krea2_img",
            prepare_warning=_warn_heavy("CCC 4.1 Krea2-Edit (48 grounded encodes)", "krea2_img"),
            notes="Patcher remaps the UNET to NVFP4 when present. Full graph stays heavy.",
        )
    )

    add(
        WorkflowVramRow(
            slug="vb_ideogram",
            family="ideogram",
            default_pack="ideogram4_fp8_scaled.safetensors",
            expected_vram_gb=0.0,
            vram_class="external",
            notes="External Ideogram API. Local UNET is unused unless the key is set.",
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_dataset_tagger",
            family="dataset",
            default_pack="qwen3vl",
            expected_vram_gb=10.0,
            vram_class="tight",
            notes="QwenVL captioner (first run downloads). Not a video gen default.",
        )
    )
    add(
        WorkflowVramRow(
            slug="vb_tag_review",
            family="dataset",
            default_pack="qwen3vl",
            expected_vram_gb=4.0,
            vram_class="cpu",
            notes="Review UI — no diffusion UNET.",
        )
    )
    return rows


_ROWS = _build_rows()


def _fallback_row(slug: str) -> WorkflowVramRow:
    family = family_for_slug(slug)
    templates = {
        "ltx25": WorkflowVramRow(slug, family, _LTX25[0], 12.5, "safe", notes="Inherits LTX 2.5 16GB pick."),
        "h3": WorkflowVramRow(slug, family, _H3[0], 13.0, "tight", notes="Inherits H3 16GB pick."),
        "ltx23": WorkflowVramRow(slug, family, _LTX23[0], 10.0, "safe", notes="Inherits LTX 2.3 EROS baked."),
        "wan22": WorkflowVramRow(slug, family, WAN22_HIGH_PREFERENCE[0], 13.2, "tight", notes="Inherits Wan 2.2 Lightx2v path."),
        "vace": WorkflowVramRow(slug, family, _VACE[0], 13.6, "tight", notes="Inherits VACE GGUF Q4_K_M."),
        "krea2": WorkflowVramRow(slug, family, _KREA[0], 10.5, "safe", notes="Inherits Krea-2 NVFP4."),
        "flux": WorkflowVramRow(slug, family, _FLUX[0], 11.0, "safe", notes="Inherits Flux GGUF/fp8."),
        "qwen_edit": WorkflowVramRow(slug, family, _QWEN[0], 12.0, "safe", notes="Inherits Qwen Edit GGUF Q5."),
        "other": WorkflowVramRow(slug, family, _LTX23[0], 10.0, "tight", notes="Unclassified — inherit LTX 2.3 16GB baked."),
    }
    return templates.get(family, templates["other"])


def workflow_row(slug: str) -> WorkflowVramRow:
    key = (slug or "").strip()
    if key in _ROWS:
        return _ROWS[key]
    return _fallback_row(key)


def workflow_vram_rows() -> list[WorkflowVramRow]:
    """One row per on-disk workflow slug (gitignored examples are omitted)."""
    from master_agent.config import load_workflow_files

    slugs = list(load_workflow_files())
    if not slugs:
        slugs = list(_ROWS)
    out: list[WorkflowVramRow] = []
    seen: set[str] = set()
    for slug in slugs:
        if slug in seen:
            continue
        seen.add(slug)
        out.append(workflow_row(slug))
    return out


def vram_class(slug: str) -> str:
    return workflow_row(slug).vram_class


def safer_alternate(slug: str) -> str:
    return workflow_row(slug).safer_alternate


def prepare_warning(slug: str) -> str:
    return workflow_row(slug).prepare_warning


def expected_vram_gb(slug: str) -> float:
    return float(workflow_row(slug).expected_vram_gb)


def is_16gb_default(slug: str) -> bool:
    return workflow_row(slug).vram_class in {"safe", "tight"}


def format_vram_table() -> str:
    """Family → default pack → expected VRAM → notes (PR / doctor)."""
    families: list[tuple[str, str, str, float, str]] = [
        ("LTX 2.3", "base / eros / directors", _LTX23[0], 9.5, "EROS baked; GGUF Q4_K_S optional; bf16 not default"),
        ("LTX 2.5", "ltx25_t2v_i2v (default 2.5)", _LTX25[0], 12.5, "GGUF Q4 → NVFP4 → int8 → bf16; two-stage is quality"),
        ("MiniMax H3", "h3_t2v / i2v / flf / r2v", _H3[0], 13.0, "Q4_K + NVFP4 TE; ≤12s / 0.8MP / 4 steps / CFG 1.0"),
        ("Wan 2.2", "wan22", WAN22_HIGH_PREFERENCE[0], 13.2, "GGUF Q4_K_S or fp8 + Lightx2v; sequential high/low; no dual bf16"),
        ("AI-VFX / VACE", "vb_aivfx_adv_13", _VACE[0], 13.6, "Q4_K_M GGUF default; v1.0 e4m3fn is heavy"),
        ("Movie Builder", "vb_movie_builder", "flux-2-klein-9b-fp8.safetensors", 15.8, "HEAVY — safer: ltx25_t2v_i2v"),
        ("CCC", "vb_ccc_adv / vb_ccc41_krea2", "flux-2-klein-9b-fp8.safetensors", 15.5, "HEAVY — safer: flux / krea2_img"),
        ("Flux / Krea", "flux / krea2_img", f"{_FLUX[0]} / {_KREA[0]}", 11.0, "Flux GGUF Q4 or fp8; Krea NVFP4"),
        ("Qwen Edit", "vb_qwen_edit_360 / startimage", _QWEN[0], 12.0, "GGUF Q5_0 + Lightning"),
        ("K3NK AIO I2V", "(not shipped)", "—", 0.0, "No attested AIO pack; do not invent. Use wan22 T2V."),
    ]
    lines = [
        "16GB policy (RTX 5060 Ti) — expected VRAM is a hypothesis, not a bench.",
        "",
        f"{'Family':<18} {'Default slug':<32} {'Default pack':<42} {'VRAM':>6}  Notes",
        "-" * 130,
    ]
    for family, slug, pack, vram, notes in families:
        vram_s = "n/a" if vram <= 0 else f"{vram:.1f}G"
        lines.append(f"{family:<18} {slug:<32} {pack:<42} {vram_s:>6}  {notes}")
    lines.append("")
    lines.append(
        "Accelerators on the 16GB path: Lightx2v / turbo LoRAs when present; "
        "LTX TeaCache inject-when-registered (soft-bypass if missing); "
        "WanVideoTeaCache stays bypass-only; "
        "DOWNSCALE_LADDER 768x512/9 → 512x320/33 after diagnose sec/step."
    )
    return "\n".join(lines) + "\n"


def format_doctor_line() -> str:
    return (
        f"{TARGET_GPU} {TARGET_VRAM_GB:.0f}GB — GGUF Q4/Q5 then NVFP4 "
        f"(≥{NVFP4_MIN_VRAM_GB:.0f}GB) then int8/fp8. Peak headroom "
        f"~{PEAK_HEADROOM_GB[0]:.0f}–{PEAK_HEADROOM_GB[1]:.0f}GB. "
        "See `python -m master_agent workflows --vram`."
    )


def downscale_ladder_for(slug: str) -> list[tuple[int, int, int]]:
    """OOM rungs. H3 stays on its 32px / 17k+5 grid; others use Rainey 8n+1."""
    family = family_for_slug(slug)
    if family == "h3":
        return [
            (1152, 640, 124),
            (896, 512, 73),
            (768, 448, 56),
            (640, 352, 39),
        ]
    if family == "wan22":
        return [
            (768, 512, 17),
            (640, 384, 17),
            (512, 384, 9),
            (512, 320, 9),
        ]
    from master_agent.config import DOWNSCALE_LADDER

    return list(DOWNSCALE_LADDER)


__all__ = [
    "ATTESTED_FILENAMES",
    "FLUX_PREFERENCE",
    "HEAVY_SLUGS",
    "KREA_PREFERENCE",
    "LTX23_PREFERENCE",
    "NVFP4_MIN_VRAM_GB",
    "PEAK_HEADROOM_GB",
    "QWEN_EDIT_PREFERENCE",
    "TARGET_GPU",
    "TARGET_VRAM_GB",
    "VACE_PREFERENCE",
    "WAN22_HIGH_PREFERENCE",
    "WAN22_LOW_PREFERENCE",
    "WorkflowVramRow",
    "describe_pack",
    "downscale_ladder_for",
    "expected_vram_gb",
    "family_for_slug",
    "format_doctor_line",
    "format_vram_table",
    "is_16gb_default",
    "pack_kind",
    "preference_order",
    "prepare_warning",
    "safer_alternate",
    "vram_class",
    "workflow_row",
    "workflow_vram_rows",
]
