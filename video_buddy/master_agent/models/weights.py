"""Required-weight manifest, scan-first, and ask-to-download.

Inventory first. Do not assume a download is needed.

1. Scan ``MODELS_DIR``, ``COMFYUI_ROOT/models``, ``EXTRA_MODELS_DIRS``,
   Comfy ``extra_model_paths.yaml`` bases, common relative ``models/`` trees,
   and Hugging Face hub snapshots.
2. If every mandatory slot is already filled (any accepted local name) →
   proceed silently and wire the graph to those files.
3. If something is confirmed missing → raise ``MissingWeightsError`` with a
   clear ask. Never auto-download.
4. Only after consent: ``download-models --ltx25 --yes`` or
   ``download-models --h3 --yes`` fetches the confirmed-missing set.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Research-graph stub names → official Lightricks/LTX-2.5 split-pack filenames.
# Hub listing 2026-09: https://huggingface.co/Lightricks/LTX-2.5
STUB_ALIASES: dict[str, str] = {
    "ltx-2.5-22b-distilled.safetensors": (
        "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    ),
    "ltx-2.5-ic-lora.safetensors": "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
    # No public Lightricks file named "msr"; IC-LoRA Ingredients is the closest
    # official adapter the MSR loader can consume. Users may drop a real MSR
    # LoRA in models/loras/ and the patcher will prefer a file that exists.
    "ltx-2.5-msr.safetensors": "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
}

# Optional placeholders in the research templates — never mandatory.
H3_OPTIONAL_LORAS = frozenset({
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
})
OPTIONAL_STUBS = frozenset({"style.safetensors", "camera-orbit.safetensors"}) | H3_OPTIONAL_LORAS

HF_LTX25 = "Lightricks/LTX-2.5"
HF_ICLORA = "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients"
HF_LICENSE = "https://huggingface.co/Lightricks/LTX-2.5"
HF_ICLORA_LICENSE = "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients"
HF_H3_GGUF = "unsloth/MiniMax-H3-GGUF"
HF_H3_COMFY = "Comfy-Org/MiniMax-H3"
HF_H3_LICENSE = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"


@dataclass(frozen=True)
class WeightFile:
    key: str
    filename: str
    dest_folder: str
    repo_id: str
    repo_filename: str
    size_bytes: int
    mandatory: bool
    gated: bool
    note: str = ""
    license_url: str = HF_LICENSE
    # Local alternatives that satisfy this slot. First is preferred when several exist.
    # Official ``filename`` is the Hub download target (not the only accepted name).
    accepts: tuple[str, ...] = ()

    @property
    def size_label(self) -> str:
        n = float(self.size_bytes)
        if n >= 1e9:
            return f"{n / 1e9:.1f} GB"
        if n >= 1e6:
            return f"{n / 1e6:.0f} MB"
        return f"{int(n)} B"

    @property
    def candidates(self) -> tuple[str, ...]:
        """Preference-ordered names that count as present (plus research stubs)."""
        seen: list[str] = []
        for name in (*self.accepts, self.filename):
            if name and name not in seen:
                seen.append(name)
        for stub, official in STUB_ALIASES.items():
            if official == self.filename and stub not in seen:
                seen.append(stub)
        return tuple(seen)


# 16GB-class GPU preference for the distilled transformer (doctor + default loader).
# 1) GGUF Q4 when present  2) NVFP4 if VRAM_GB >= 14  3) int8-convrot / bf16
NVFP4_MIN_VRAM_GB = 14.0
TRANSFORMER_PREFERENCE: tuple[str, ...] = (
    "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf",
    "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
)
TE_PREFERENCE: tuple[str, ...] = (
    "gemma4-12b-heretic-ltx25-int8convrot.safetensors",
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
)

# MiniMax H3 16GB-class pick: GGUF Q4_K DiT, then NVFP4 / int8 official if present.
# Prefer Comfy TE (NVFP4 AWQ on Blackwell, else int8/int4). GGUF TE Q4_K_M (~17GB) is last.
H3_FL2VA_PREFERENCE: tuple[str, ...] = (
    "minimax_h3_fl2va_pruned-Q4_K.gguf",
    "minimax_h3_fl2va_pruned_nvfp4.safetensors",
    "minimax_h3_fl2va_nvfp4.safetensors",
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_fl2va_int8_convrot.safetensors",
    "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "minimax_h3_fl2va_pruned_bf16.safetensors",
    "minimax_h3_fl2va_bf16.safetensors",
)
H3_REF2VA_PREFERENCE: tuple[str, ...] = (
    "minimax_h3_ref2va_pruned-Q4_K.gguf",
    "minimax_h3_ref2va_pruned_nvfp4.safetensors",
    "minimax_h3_ref2va_nvfp4.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
    "minimax_h3_ref2va_pruned_bf16.safetensors",
    "minimax_h3_ref2va_bf16.safetensors",
)
H3_TE_PREFERENCE: tuple[str, ...] = (
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4.safetensors",
    "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_bf16.safetensors",
    "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
)
H3_TRANSFORMER_KEYS = frozenset({"h3_fl2va", "h3_ref2va"})

# Official distilled split pack on gated Lightricks/LTX-2.5 (Hub file listing).
# Download / default-wire names are the bf16 pack. 16GB-class installs may
# already have comfy-int8-convrot / nvfp4 / GGUF Q4 — those still count.
WEIGHT_FILES: dict[str, WeightFile] = {
    "transformer": WeightFile(
        key="transformer",
        filename="ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        dest_folder="diffusion_models",
        repo_id=HF_LTX25,
        repo_filename="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        size_bytes=42_000_000_000,
        mandatory=True,
        gated=True,
        note="Official distilled transformer (bf16, 42 GB). 16GB-class: comfy-int8-convrot, nvfp4, or GGUF Q4 also count.",
        accepts=TRANSFORMER_PREFERENCE,
    ),
    "text_encoder": WeightFile(
        key="text_encoder",
        filename="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        dest_folder="text_encoders",
        repo_id=HF_LTX25,
        repo_filename="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        size_bytes=26_300_000_000,
        mandatory=True,
        gated=True,
        note="Official Gemma 4 12B + LTX 2.5 projection (bf16, 26.3 GB). Comfy int8 or heretic int8 also count.",
        accepts=TE_PREFERENCE,
    ),
    "video_vae": WeightFile(
        key="video_vae",
        filename="ltx-2.5-video-vae-bf16.safetensors",
        dest_folder="vae",
        repo_id=HF_LTX25,
        repo_filename="vae/ltx-2.5-video-vae-bf16.safetensors",
        size_bytes=1_500_000_000,
        mandatory=True,
        gated=True,
        note="Video VAE (bf16). The conv-bf16 variant also counts.",
        accepts=(
            "ltx-2.5-video-vae-bf16.safetensors",
            "ltx-2.5-video-vae-conv-bf16.safetensors",
        ),
    ),
    "audio_vae": WeightFile(
        key="audio_vae",
        filename="ltx-2.5-audio-vae-bf16.safetensors",
        dest_folder="vae",
        repo_id=HF_LTX25,
        repo_filename="vae/ltx-2.5-audio-vae-bf16.safetensors",
        size_bytes=365_000_000,
        mandatory=True,
        gated=True,
        note="Audio VAE (bf16). Required for T2A / A2V and AV joint graphs.",
    ),
    "spatial_upscaler": WeightFile(
        key="spatial_upscaler",
        filename="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        dest_folder="latent_upscale_models",
        repo_id=HF_LTX25,
        repo_filename="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        size_bytes=996_000_000,
        mandatory=True,
        gated=True,
        note="2× latent spatial upscaler — official pack (required for two-stage; part of the Hub split).",
    ),
    "ic_lora": WeightFile(
        key="ic_lora",
        filename="ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        dest_folder="loras",
        repo_id=HF_ICLORA,
        repo_filename="ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        size_bytes=1_300_000_000,
        mandatory=True,
        gated=True,
        license_url=HF_ICLORA_LICENSE,
        note="Official IC-LoRA Ingredients. Pixel-spatial IC-LoRA and research stubs also count.",
        accepts=(
            "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
            "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
        ),
    ),
    "distilled_lora": WeightFile(
        key="distilled_lora",
        filename="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        dest_folder="loras",
        repo_id=HF_LTX25,
        repo_filename="loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        size_bytes=8_900_000_000,
        mandatory=False,
        gated=True,
        note="Optional distilled LoRA (450). Not required for the default distilled transformer pack.",
    ),
    "temporal_upscaler": WeightFile(
        key="temporal_upscaler",
        filename="ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
        dest_folder="latent_upscale_models",
        repo_id=HF_LTX25,
        repo_filename="latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
        size_bytes=262_000_000,
        mandatory=False,
        gated=True,
        note="Optional 2× latent temporal upscaler.",
    ),
    "duration_head": WeightFile(
        key="duration_head",
        filename="ltx-2.5-duration-head-bf16.safetensors",
        dest_folder="model_patches",
        repo_id=HF_LTX25,
        repo_filename="model_patches/ltx-2.5-duration-head-bf16.safetensors",
        size_bytes=3_800_000,
        mandatory=False,
        gated=True,
        note=(
            "Optional duration-head model patch (3.8 MB). No shipped LTX 2.5 graph loads it. "
            "Zero-byte placeholders still count as not present."
        ),
    ),
    "text_enhancer": WeightFile(
        key="text_enhancer",
        filename="gemma4_e2b_it_bf16.safetensors",
        dest_folder="text_encoders",
        repo_id="Comfy-Org/gemma-4",
        repo_filename="text_encoders/gemma4_e2b_it_bf16.safetensors",
        size_bytes=10_300_000_000,
        mandatory=False,
        gated=False,
        note=(
            "Optional LTX 2.5 text-enhancer CLIP. When absent, the enhancer "
            "CLIPLoader is retargeted at the resolved Gemma TE."
        ),
    ),
}

# MiniMax H3 — GGUF DiT download target; official Comfy int8 / NVFP4 / bf16 still count.
WEIGHT_FILES["h3_fl2va"] = WeightFile(
    key="h3_fl2va",
    filename="minimax_h3_fl2va_pruned-Q4_K.gguf",
    dest_folder="diffusion_models",
    repo_id=HF_H3_GGUF,
    repo_filename="minimax_h3_fl2va_pruned-Q4_K.gguf",
    size_bytes=11_423_744_163,
    mandatory=True,
    gated=False,
    license_url=HF_H3_LICENSE,
    note="Unsloth fl2va GGUF Q4_K (~10.6 GiB). Official NVFP4 / int8 / fp8 / bf16 also count.",
    accepts=H3_FL2VA_PREFERENCE,
)
WEIGHT_FILES["h3_ref2va"] = WeightFile(
    key="h3_ref2va",
    filename="minimax_h3_ref2va_pruned-Q4_K.gguf",
    dest_folder="diffusion_models",
    repo_id=HF_H3_GGUF,
    repo_filename="minimax_h3_ref2va_pruned-Q4_K.gguf",
    size_bytes=11_380_755_661,
    mandatory=True,
    gated=False,
    license_url=HF_H3_LICENSE,
    note="Unsloth ref2va GGUF Q4_K (~10.6 GiB). Official NVFP4 / int8 / fp8 / bf16 also count.",
    accepts=H3_REF2VA_PREFERENCE,
)
WEIGHT_FILES["h3_text_encoder"] = WeightFile(
    key="h3_text_encoder",
    filename="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    dest_folder="text_encoders",
    repo_id=HF_H3_COMFY,
    repo_filename="text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    size_bytes=15_687_142_551,
    mandatory=True,
    gated=False,
    license_url=HF_H3_LICENSE,
    note="Comfy Qwen3-VL TE (NVFP4 AWQ, Blackwell). int8/int4 convrot also count. GGUF TE Q4_K_M (~17GB) is last-resort.",
    accepts=H3_TE_PREFERENCE,
)
WEIGHT_FILES["h3_video_vae"] = WeightFile(
    key="h3_video_vae",
    filename="minimax_h3_video_vae_fp16.safetensors",
    dest_folder="vae",
    repo_id=HF_H3_COMFY,
    repo_filename="vae/minimax_h3_video_vae_fp16.safetensors",
    size_bytes=5_207_808_496,
    mandatory=True,
    gated=False,
    license_url=HF_H3_LICENSE,
    note="Official Comfy-Org video VAE (fp16).",
)
WEIGHT_FILES["h3_audio_vae"] = WeightFile(
    key="h3_audio_vae",
    filename="minimax_h3_audio_vae_fp32.safetensors",
    dest_folder="vae",
    repo_id=HF_H3_COMFY,
    repo_filename="vae/minimax_h3_audio_vae_fp32.safetensors",
    size_bytes=605_254_808,
    mandatory=True,
    gated=False,
    license_url=HF_H3_LICENSE,
    note="Official Comfy-Org audio VAE (fp32, 32 kHz stereo).",
)
WEIGHT_FILES["h3_fl2v_turbo"] = WeightFile(
    key="h3_fl2v_turbo",
    filename="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    dest_folder="loras",
    repo_id=HF_H3_COMFY,
    repo_filename="loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    size_bytes=1_956_192_992,
    mandatory=False,
    gated=False,
    license_url=HF_H3_LICENSE,
    note="Optional LightX2V / official 4-step turbo LoRA for fl2va. Bypassed when absent.",
    accepts=(
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    ),
)
WEIGHT_FILES["h3_ref2v_turbo"] = WeightFile(
    key="h3_ref2v_turbo",
    filename="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    dest_folder="loras",
    repo_id=HF_H3_COMFY,
    repo_filename="loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    size_bytes=1_956_193_000,
    mandatory=False,
    gated=False,
    license_url=HF_H3_LICENSE,
    note="Optional LightX2V / official 4-step turbo LoRA for ref2va. Bypassed when absent.",
)

# --- other 16GB-class families (attested names only; scan-first) -----------
from master_agent.models.vram_policy import (
    FLUX_PREFERENCE,
    KREA_PREFERENCE,
    QWEN_EDIT_PREFERENCE,
    VACE_PREFERENCE,
    WAN22_HIGH_PREFERENCE,
    WAN22_LOW_PREFERENCE,
)

HF_WAN22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
HF_WAN22_GGUF = "QuantStack/Wan2.2-T2V-A14B-GGUF"
HF_VACE_GGUF = "mickmumpitz/VACE_Skyreels_V3_R2V_Merge-GGUF"
HF_KREA = "Comfy-Org/Krea-2"
HF_FLUX_GGUF = "city96/FLUX.1-dev-gguf"
HF_QWEN_EDIT = "QuantStack/Qwen-Image-Edit-2509-GGUF"

WEIGHT_FILES["wan22_high"] = WeightFile(
    key="wan22_high",
    filename="Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf",
    dest_folder="diffusion_models",
    repo_id=HF_WAN22_GGUF,
    repo_filename="HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf",
    size_bytes=8_700_000_000,
    mandatory=True,
    gated=False,
    license_url="https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF",
    note="Wan 2.2 high-noise UNET. 16GB download: QuantStack GGUF Q4_K_S. Comfy-Org fp8 also counts.",
    accepts=WAN22_HIGH_PREFERENCE,
)
WEIGHT_FILES["wan22_low"] = WeightFile(
    key="wan22_low",
    filename="Wan2.2-T2V-A14B-LowNoise-Q4_K_S.gguf",
    dest_folder="diffusion_models",
    repo_id=HF_WAN22_GGUF,
    repo_filename="LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_S.gguf",
    size_bytes=8_700_000_000,
    mandatory=True,
    gated=False,
    license_url="https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF",
    note="Wan 2.2 low-noise UNET. Sequential with high-noise + Lightx2v on 16GB.",
    accepts=WAN22_LOW_PREFERENCE,
)
WEIGHT_FILES["wan22_lightx2v"] = WeightFile(
    key="wan22_lightx2v",
    filename="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
    dest_folder="loras",
    repo_id="Kijai/WanVideo_comfy",
    repo_filename="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
    size_bytes=307_000_000,
    mandatory=False,
    gated=False,
    license_url="https://huggingface.co/Kijai/WanVideo_comfy",
    note="Lightx2v distill LoRA — 16GB Wan path (already in the native graph Power Lora).",
)
WEIGHT_FILES["vace"] = WeightFile(
    key="vace",
    filename="wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1-Q4_K_M.gguf",
    dest_folder="diffusion_models",
    repo_id=HF_VACE_GGUF,
    repo_filename="wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1-Q4_K_M.gguf",
    size_bytes=8_500_000_000,
    mandatory=True,
    gated=False,
    license_url="https://huggingface.co/mickmumpitz/VACE_Skyreels_V3_R2V_Merge-GGUF",
    note="VACE Skyreels Q4_K_M GGUF (16GB default). e4m3fn fp8 also counts.",
    accepts=VACE_PREFERENCE,
)
WEIGHT_FILES["krea2"] = WeightFile(
    key="krea2",
    filename="krea2_turbo_nvfp4.safetensors",
    dest_folder="diffusion_models",
    repo_id=HF_KREA,
    repo_filename="diffusion_models/krea2_turbo_nvfp4.safetensors",
    size_bytes=7_700_000_000,
    mandatory=True,
    gated=False,
    license_url="https://huggingface.co/Comfy-Org/Krea-2",
    note="Krea-2 turbo NVFP4 (Blackwell 16GB default). int8 / fp8 also count. bf16 is not default.",
    accepts=KREA_PREFERENCE,
)
WEIGHT_FILES["flux"] = WeightFile(
    key="flux",
    filename="flux1-dev-Q4_K_S.gguf",
    dest_folder="diffusion_models",
    repo_id=HF_FLUX_GGUF,
    repo_filename="flux1-dev-Q4_K_S.gguf",
    size_bytes=6_800_000_000,
    mandatory=True,
    gated=False,
    license_url="https://huggingface.co/city96/FLUX.1-dev-gguf",
    note="Flux.1-dev. 16GB download: city96 Q4_K_S GGUF. Comfy-Org fp8 also counts.",
    accepts=FLUX_PREFERENCE,
)
WEIGHT_FILES["qwen_edit"] = WeightFile(
    key="qwen_edit",
    filename="Qwen-Image-Edit-2509-Q5_0.gguf",
    dest_folder="diffusion_models",
    repo_id=HF_QWEN_EDIT,
    repo_filename="Qwen-Image-Edit-2509-Q5_0.gguf",
    size_bytes=8_800_000_000,
    mandatory=True,
    gated=False,
    license_url="https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF",
    note="Qwen-Image-Edit GGUF Q5_0 (360 + start-image). 2511 Q5_0 also counts.",
    accepts=QWEN_EDIT_PREFERENCE,
)

# Official Hub pack keys. duration_head and text_enhancer are optional
# (mandatory=False on the WeightFile). IC-LoRA is a separate gated repo.
_LTX25_OFFICIAL = (
    "transformer",
    "text_encoder",
    "video_vae",
    "audio_vae",
    "duration_head",
    "spatial_upscaler",
    "text_enhancer",
)

_LTX25_ALL = (
    *_LTX25_OFFICIAL,
    "ic_lora",
    "distilled_lora",
    "temporal_upscaler",
)
_H3_SHARED = ("h3_text_encoder", "h3_video_vae", "h3_audio_vae")
_H3_FL2VA = ("h3_fl2va", *_H3_SHARED)
_H3_REF2VA = ("h3_ref2va", *_H3_SHARED)
_H3_ALL = (
    "h3_fl2va",
    "h3_ref2va",
    *_H3_SHARED,
    "h3_fl2v_turbo",
    "h3_ref2v_turbo",
)
_WAN22 = ("wan22_high", "wan22_low", "wan22_lightx2v")
_VACE = ("vace",)
_KREA2 = ("krea2",)
_FLUX = ("flux",)
_QWEN_EDIT = ("qwen_edit",)

# Bundle → weight keys. Mandatory flags on WeightFile still apply per key.
# ltx25_all stays LTX-only so adding H3 keys never pulls MiniMax into --ltx25.
BUNDLES: dict[str, tuple[str, ...]] = {
    "ltx25_core": _LTX25_OFFICIAL,
    "ltx25_two_stage": _LTX25_OFFICIAL,
    "ltx25_iclora": (*_LTX25_OFFICIAL, "ic_lora"),
    "ltx25_msr": (*_LTX25_OFFICIAL, "ic_lora"),
    "ltx25_all": _LTX25_ALL,
    "h3_fl2va": _H3_FL2VA,
    "h3_ref2va": _H3_REF2VA,
    "h3_all": _H3_ALL,
    "wan22": _WAN22,
    "vace": _VACE,
    "krea2": _KREA2,
    "flux": _FLUX,
    "qwen_edit": _QWEN_EDIT,
}

# Variant id / alias → bundle
VARIANT_BUNDLES: dict[str, str] = {
    "ltx25_t2v_i2v": "ltx25_core",
    "ltx25_t2v_i2v_two_stage": "ltx25_two_stage",
    "ltx25_flf2v": "ltx25_core",
    "ltx25_a2v": "ltx25_core",
    "ltx25_t2a": "ltx25_core",
    "ltx25_v2v_ic_lora": "ltx25_iclora",
    "ltx25_msr": "ltx25_msr",
    "t2v_i2v": "ltx25_core",
    "t2v_i2v_two_stage": "ltx25_two_stage",
    "flf2v": "ltx25_core",
    "a2v": "ltx25_core",
    "t2a": "ltx25_core",
    "v2v_ic_lora": "ltx25_iclora",
    "msr": "ltx25_msr",
    "h3_t2v": "h3_fl2va",
    "h3_i2v": "h3_fl2va",
    "h3_flf": "h3_fl2va",
    "h3_r2v": "h3_ref2va",
    "fl2va": "h3_fl2va",
    "h3_fl2va": "h3_fl2va",
    "ref2va": "h3_ref2va",
    "h3_ref2va": "h3_ref2va",
    "h3": "h3_fl2va",
    "minimax_h3": "h3_fl2va",
    "minimax": "h3_fl2va",
    "wan22": "wan22",
    "vb_wan22_vid": "wan22",
    "vb_aivfx_adv_13": "vace",
    "vb_ai_renderer_smpl": "vace",
    "krea2_img": "krea2",
    "vb_krea2_img": "krea2",
    "flux": "flux",
    "vb_qwen_edit_360": "qwen_edit",
    "vb_aivfx_startimage": "qwen_edit",
}


class MissingWeightsError(RuntimeError):
    """Queue-time hard stop: list missing files and how to download them."""

    def __init__(self, message: str, missing: list[WeightFile], *, bundle: str = ""):
        super().__init__(message)
        self.missing = missing
        self.bundle = bundle


@dataclass
class WeightStatus:
    present: list[WeightFile] = field(default_factory=list)
    missing_mandatory: list[WeightFile] = field(default_factory=list)
    missing_optional: list[WeightFile] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    bundle: str = ""
    found_paths: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing_mandatory

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "bundle": self.bundle,
            "roots": self.roots,
            "present": [
                {"key": w.key, "official": w.filename, "found": self.found_paths.get(w.key, w.filename)}
                for w in self.present
            ],
            "missing_mandatory": [_file_row(w) for w in self.missing_mandatory],
            "missing_optional": [_file_row(w) for w in self.missing_optional],
            "ask": format_ask(self) if not self.ok else "",
        }


def _file_row(w: WeightFile) -> dict[str, Any]:
    return {
        "key": w.key,
        "filename": w.filename,
        "dest_folder": w.dest_folder,
        "size": w.size_label,
        "repo": f"{w.repo_id}/{w.repo_filename}",
        "gated": w.gated,
        "mandatory": w.mandatory,
        "license": w.license_url,
        "note": w.note,
    }


def _is_usable_file(path: Path) -> bool:
    """Present means a real file with bytes. Zero-byte placeholders are missing."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def filename_search_names(name: str) -> tuple[str, ...]:
    """Exact name, slash-normalized path, and basename (Windows ``folder\\file``)."""
    raw = (name or "").strip()
    if not raw:
        return ()
    slash = raw.replace("\\", "/")
    base = slash.rsplit("/", 1)[-1]
    seen: list[str] = []
    for item in (raw, slash, base):
        if item and item not in seen:
            seen.append(item)
    return tuple(seen)


def _hf_hub_cache_dirs() -> list[Path]:
    """HF hub cache locations (env, XDG, default home, huggingface_hub constant)."""
    hubs: list[Path] = []
    env_cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if env_cache:
        hubs.append(Path(env_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hubs.append(Path(hf_home) / "hub")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        hubs.append(Path(xdg) / "huggingface" / "hub")
    hubs.append(Path.home() / ".cache" / "huggingface" / "hub")
    # Do not import huggingface_hub.constants.HF_HUB_CACHE — it freezes the
    # cache path at first import and ignores later HF_HOME / HF_HUB_CACHE.
    return hubs


def _hf_repo_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _hf_hub_snapshot_roots() -> list[Path]:
    """Every cached Hub snapshot tree, not just LTX / H3."""
    known = {
        "models--Lightricks--LTX-2.5",
        "models--Lightricks--LTX-2.5-22b-IC-LoRA-Ingredients",
        "models--Comfy-Org--MiniMax-H3",
        "models--unsloth--MiniMax-H3-GGUF",
        "models--Comfy-Org--flux1-dev",
        "models--comfyanonymous--flux_text_encoders",
    }
    for weight in WEIGHT_FILES.values():
        if weight.repo_id:
            known.add(_hf_repo_dir_name(weight.repo_id))
    out: list[Path] = []
    seen: set[Path] = set()

    def _keep(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(resolved)

    for hub in _hf_hub_cache_dirs():
        if not hub.is_dir():
            continue
        for repo in known:
            repo_dir = hub / repo
            if repo_dir.is_dir():
                _keep(repo_dir)
            snap = repo_dir / "snapshots"
            if snap.is_dir():
                _keep(snap)
        try:
            for repo_dir in hub.glob("models--*"):
                snap = repo_dir / "snapshots"
                if snap.is_dir():
                    _keep(snap)
        except OSError:
            continue
    return out


def _extra_model_paths_yaml_roots() -> list[Path]:
    """Comfy ``extra_model_paths.yaml`` base_path / folder entries (other volumes)."""
    from master_agent.config import COMFYUI_ROOT, PORTABLE_ROOT, PROJECT_ROOT

    candidates = (
        Path(COMFYUI_ROOT) / "extra_model_paths.yaml",
        Path(COMFYUI_ROOT).parent / "extra_model_paths.yaml",
        Path(PORTABLE_ROOT) / "extra_model_paths.yaml",
        Path(PROJECT_ROOT) / "extra_model_paths.yaml",
        Path.cwd() / "extra_model_paths.yaml",
    )
    roots: list[Path] = []
    try:
        import yaml
    except ImportError:
        return roots
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        for section in data.values():
            if not isinstance(section, dict):
                continue
            base = section.get("base_path")
            base_path = Path(str(base)) if base else None
            if base_path is not None:
                roots.append(base_path)
                roots.append(base_path / "models")
            for key, val in section.items():
                if key in {"base_path", "is_default", "custom_nodes"} or not isinstance(val, str):
                    continue
                folder = Path(val)
                if not folder.is_absolute() and base_path is not None:
                    folder = base_path / val
                roots.append(folder)
                if folder.name != "models":
                    roots.append(folder.parent)
    return roots


def model_search_roots() -> list[Path]:
    """``MODELS_DIR`` plus common Comfy / HF-cache / extra-volume layouts."""
    from master_agent.config import COMFYUI_ROOT, MODELS_DIR, PROJECT_ROOT, extra_models_dirs

    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    _add(Path(MODELS_DIR))
    _add(Path(COMFYUI_ROOT) / "models")
    _add(Path(PROJECT_ROOT) / "models")
    _add(Path(PROJECT_ROOT) / "ComfyUI" / "models")
    _add(Path(PROJECT_ROOT) / "ComfyUI_windows_portable" / "ComfyUI" / "models")
    parent = Path(PROJECT_ROOT).parent
    _add(parent / "models")
    _add(parent / "ComfyUI" / "models")
    _add(parent / "ComfyUI_windows_portable" / "ComfyUI" / "models")
    cwd = Path.cwd()
    _add(cwd / "models")
    _add(cwd / "video_buddy" / "models")
    _add(cwd / "ComfyUI" / "models")
    _add(cwd / "ComfyUI_windows_portable" / "ComfyUI" / "models")
    for extra in extra_models_dirs():
        _add(extra)
        _add(extra / "models")
    for extra in _extra_model_paths_yaml_roots():
        _add(extra)
    for snap in _hf_hub_snapshot_roots():
        _add(snap)
    return roots


def _search_name(name: str, roots: Iterable[Path]) -> Path | None:
    variants = filename_search_names(name)
    if not variants:
        return None
    subdirs = (
        "checkpoints",
        "diffusion_models",
        "diffusion_models/gguf",
        "unet",
        "loras",
        "vae",
        "text_encoders",
        "clip",
        "latent_upscale_models",
        "model_patches",
        "",
    )
    needles = tuple(
        dict.fromkeys(v.replace("\\", "/").rsplit("/", 1)[-1] for v in variants if v)
    )
    for root in roots:
        if not root.is_dir():
            continue
        for variant in variants:
            rel = variant.replace("\\", "/")
            parts = [p for p in rel.split("/") if p]
            if not parts:
                continue
            direct = root.joinpath(*parts)
            if _is_usable_file(direct):
                return direct
            base = parts[-1]
            for sub in subdirs:
                candidate = root.joinpath(*sub.split("/"), base) if sub else root / base
                if _is_usable_file(candidate):
                    return candidate
        for needle in needles:
            try:
                hits = root.rglob(needle)
            except OSError:
                hits = []
            for hit in hits:
                if _is_usable_file(hit):
                    return hit
    return None


def find_weight_file(filename: str, roots: Iterable[Path] | None = None) -> Path | None:
    """Return the first usable path for a bare filename (or stub / family alias)."""
    names: list[str] = []
    for raw in filename_search_names(filename):
        if raw not in names:
            names.append(raw)
    for candidate in list(names):
        aliased = STUB_ALIASES.get(candidate)
        if aliased and aliased not in names:
            names.append(aliased)
        for alias, official in STUB_ALIASES.items():
            if official == candidate and alias not in names:
                names.append(alias)
    base_names = {n.replace("\\", "/").rsplit("/", 1)[-1] for n in names}
    for weight in WEIGHT_FILES.values():
        cand = set(weight.candidates)
        cand_bases = {c.replace("\\", "/").rsplit("/", 1)[-1] for c in cand}
        if set(names) & cand or base_names & cand_bases:
            for extra in weight.candidates:
                if extra not in names:
                    names.append(extra)
            break
    search = list(roots) if roots is not None else model_search_roots()
    for name in names:
        found = _search_name(name, search)
        if found is not None:
            return found
    return None


def transformer_preference_order(*, vram_gb: float | None = None) -> tuple[str, ...]:
    """16GB-class pick order: GGUF → NVFP4 (if VRAM fits) → int8 → bf16 → stub."""
    from master_agent.models.vram_policy import preference_order

    names = list(preference_order(TRANSFORMER_PREFERENCE, vram_gb=vram_gb))
    for stub, official in STUB_ALIASES.items():
        if official == TRANSFORMER_PREFERENCE[-1] and stub not in names:
            names.append(stub)
    return tuple(names)


def h3_transformer_preference_order(
    key: str = "h3_fl2va",
    *,
    vram_gb: float | None = None,
) -> tuple[str, ...]:
    """H3 16GB-class pick: GGUF Q4_K → NVFP4 (if VRAM fits) → int8 → fp8 → bf16."""
    from master_agent.models.vram_policy import preference_order

    prefs = H3_REF2VA_PREFERENCE if key == "h3_ref2va" else H3_FL2VA_PREFERENCE
    return preference_order(prefs, vram_gb=vram_gb)


def describe_transformer_pick(path: Path | None) -> str:
    if path is None:
        return "no local transformer"
    name = path.name
    if name.lower().endswith(".gguf"):
        return f"GGUF Q4 ({name}) — 16GB-class preference #1"
    if "nvfp4" in name.lower():
        return f"NVFP4 ({name}) — 16GB-class preference #2"
    if "int8-convrot" in name.lower():
        return f"int8-convrot ({name}) — 16GB-class preference #3"
    return f"{name} — fallback"


def describe_h3_transformer_pick(path: Path | None) -> str:
    if path is None:
        return "no local H3 transformer"
    name = path.name
    if name.lower().endswith(".gguf"):
        return f"H3 GGUF Q4_K ({name}) — 16GB-class preference #1"
    if "nvfp4" in name.lower():
        return f"H3 NVFP4 ({name}) — 16GB-class preference #2"
    if "int8" in name.lower():
        return f"H3 int8 ({name}) — 16GB-class preference #3"
    return f"{name} — H3 fallback"


def resolve_weight(weight: WeightFile, roots: Iterable[Path] | None = None) -> Path | None:
    """Best local file for a slot (preference order). None if all candidates missing/empty."""
    search = list(roots) if roots is not None else model_search_roots()
    if weight.key == "transformer":
        order = transformer_preference_order()
    elif weight.key in H3_TRANSFORMER_KEYS:
        order = h3_transformer_preference_order(weight.key)
    elif weight.key == "h3_text_encoder":
        order = H3_TE_PREFERENCE
    elif weight.key in {
        "wan22_high",
        "wan22_low",
        "vace",
        "krea2",
        "flux",
        "qwen_edit",
    }:
        from master_agent.models.vram_policy import preference_order

        order = preference_order(weight.candidates)
    else:
        order = weight.candidates
    for name in order:
        found = _search_name(name, search)
        if found is not None:
            return found
    if weight.key == "transformer" or weight.key in H3_TRANSFORMER_KEYS:
        for name in weight.candidates:
            found = _search_name(name, search)
            if found is not None:
                return found
    return None


def files_for_bundle(bundle: str) -> list[WeightFile]:
    keys = BUNDLES.get(bundle) or BUNDLES["ltx25_core"]
    return [WEIGHT_FILES[k] for k in keys if k in WEIGHT_FILES]


def bundle_for_variant(variant: str | None) -> str | None:
    key = (variant or "").strip()
    if key in VARIANT_BUNDLES:
        return VARIANT_BUNDLES[key]
    lowered = key.lower().replace("\\", "/")
    if (
        lowered.startswith("h3")
        or "minimax" in lowered
        or "/minimax-h3/" in lowered
        or "fl2va" in lowered
        or "ref2va" in lowered
    ):
        if "ref" in lowered or "r2v" in lowered:
            return "h3_ref2va"
        return "h3_fl2va"
    if key.startswith("ltx25") or key.startswith("ltx-2.5") or "/ltx-2.5/" in key:
        if "two_stage" in key or "two-stage" in key:
            return "ltx25_two_stage"
        if "iclora" in key or "ic_lora" in key or "ic-lora" in key:
            return "ltx25_iclora"
        if "msr" in key:
            return "ltx25_msr"
        return "ltx25_core"
    from master_agent.models.vram_policy import family_for_slug

    family = family_for_slug(key)
    family_bundles = {
        "wan22": "wan22",
        "vace": "vace",
        "krea2": "krea2",
        "flux": "flux",
        "qwen_edit": "qwen_edit",
    }
    return family_bundles.get(family)


def is_h3_bundle(bundle: str | None) -> bool:
    return bool(bundle) and str(bundle).startswith("h3")


def is_ltx25_bundle(bundle: str | None) -> bool:
    return bool(bundle) and str(bundle).startswith("ltx25")


def scan_bundle(bundle: str, *, roots: Iterable[Path] | None = None) -> WeightStatus:
    search = list(roots) if roots is not None else model_search_roots()
    status = WeightStatus(roots=[str(p) for p in search], bundle=bundle)
    for weight in files_for_bundle(bundle):
        found = resolve_weight(weight, search)
        if found is not None:
            status.present.append(weight)
            status.found_paths[weight.key] = str(found)
            continue
        if weight.mandatory:
            status.missing_mandatory.append(weight)
        else:
            status.missing_optional.append(weight)
    return status


def scan_variant(variant: str, *, roots: Iterable[Path] | None = None) -> WeightStatus | None:
    bundle = bundle_for_variant(variant)
    if not bundle:
        return None
    return scan_bundle(bundle, roots=roots)


def format_ask(status: WeightStatus) -> str:
    lines = [
        "Missing required model files for this workflow.",
        "Video Buddy scanned configured model dirs and common Comfy models/ paths.",
        "Nothing was downloaded (consent required).",
        "",
        "Paths checked:",
    ]
    if status.roots:
        lines.extend(f"  - {root}" for root in status.roots)
    else:
        lines.append("  (none)")
    lines += [
        "",
        f"Bundle: {status.bundle or 'ltx25'}",
        "",
        "Already present (not re-downloaded):",
    ]
    h3 = is_h3_bundle(status.bundle)
    if status.found_paths:
        for key, path in status.found_paths.items():
            lines.append(f"  - {key}: {path}")
    else:
        lines.append("  (none)")
    lines += [
        "",
        "Missing (mandatory):",
    ]
    for w in status.missing_mandatory:
        if w.gated:
            gated = "  [gated Hugging Face — accept the LTX-2.x Community License first]"
        elif w.key.startswith("h3") or h3:
            gated = "  [MiniMax H3 Community License — verify you are entitled to run the weights]"
        else:
            gated = ""
        lines.append(
            f"  - {w.filename}  →  models/{w.dest_folder}/  ({w.size_label}){gated}"
        )
        lines.append(f"      {w.repo_id}  {w.repo_filename}")
        if w.note:
            lines.append(f"      {w.note}")
        alts = [n for n in w.candidates if n != w.filename]
        if alts:
            lines.append("      also accepted locally: " + ", ".join(alts))
        lines.append(f"      license: {w.license_url}")
    if status.missing_optional:
        lines.append("")
        lines.append("Optional (not blocking):")
        for w in status.missing_optional:
            lines.append(f"  - {w.filename}  →  models/{w.dest_folder}/  ({w.size_label})")
    lines.append("")
    lines.append("After you agree, download only the missing set with:")
    if h3:
        lines.append("  python -m master_agent download-models --h3 --yes")
        lines.append("")
        lines.append("MiniMax H3 Community License:")
        lines.append(f"  {HF_H3_LICENSE}")
    elif (status.bundle or "").startswith("ltx25"):
        lines.append("  python -m master_agent download-models --ltx25 --yes")
        lines.append("or:")
        lines.append("  python -m master_agent doctor --fix-models")
        lines.append("")
        lines.append("Requires a Hugging Face token with access to the gated repos")
        lines.append(f"({HF_LICENSE}) via HF_TOKEN / huggingface-cli login.")
    else:
        flag = {
            "wan22": "--wan",
            "vace": "--vace",
            "krea2": "--krea",
            "flux": "--flux-pack",
            "qwen_edit": "--qwen",
        }.get(status.bundle or "", f"--bundle {status.bundle}")
        lines.append(f"  python -m master_agent download-models {flag} --yes")
    return "\n".join(lines)


def require_weights(variant: str, *, roots: Iterable[Path] | None = None) -> WeightStatus | None:
    """Silent if present; raise MissingWeightsError if mandatory files are gone.

    Hard-stop stays LTX 2.5 / H3 (new catalog packs). Other families are
    scanned by doctor / download-models but do not block the queue — their
    weights often already live on the tower under folder-prefixed names.
    """
    bundle = bundle_for_variant(variant)
    if not bundle or not (is_ltx25_bundle(bundle) or is_h3_bundle(bundle)):
        return None
    status = scan_variant(variant, roots=roots)
    if status is None or status.ok:
        return status
    raise MissingWeightsError(format_ask(status), status.missing_mandatory, bundle=status.bundle)


def download_files(
    files: Iterable[WeightFile],
    *,
    dest_root: Path | None = None,
    progress: Callable[[str], None] = print,
    yes: bool = False,
) -> list[Path]:
    """Copy missing weights into ``dest_root``. Refuses unless ``yes=True``."""
    from master_agent.config import MODELS_DIR

    wanted = list(files)
    if not wanted:
        return []

    root = Path(dest_root or MODELS_DIR)
    paths: list[Path] = []
    still_missing: list[WeightFile] = []
    for weight in wanted:
        found = resolve_weight(weight)
        if found is not None:
            progress(
                f"SKIP download of {weight.filename} — local file found at {found} "
                "(not re-downloading)"
            )
            paths.append(found)
            continue
        still_missing.append(weight)
    if not still_missing:
        return paths
    if not yes:
        raise MissingWeightsError(
            "Refusing to download without consent. Re-run with --yes after reviewing:\n"
            + "\n".join(
                f"  {w.filename} → models/{w.dest_folder}/ ({w.size_label})"
                for w in still_missing
            ),
            list(still_missing),
        )
    from master_agent.models.download import download_hub_file

    for weight in still_missing:
        dest = root / weight.dest_folder / weight.filename
        paths.append(
            download_hub_file(
                repo_id=weight.repo_id,
                repo_filename=weight.repo_filename,
                dest=dest,
                progress=progress,
            )
        )
    return paths


def download_missing_bundle(
    bundle: str = "ltx25_all",
    *,
    dest_root: Path | None = None,
    progress: Callable[[str], None] = print,
    yes: bool = False,
    include_optional: bool = False,
) -> tuple[WeightStatus, list[Path]]:
    status = scan_bundle(bundle)
    missing = list(status.missing_mandatory)
    if include_optional:
        missing.extend(status.missing_optional)
    if not missing:
        progress(f"OK    all {bundle} weights already present")
        return status, []
    paths = download_files(missing, dest_root=dest_root, progress=progress, yes=yes)
    return status, paths
