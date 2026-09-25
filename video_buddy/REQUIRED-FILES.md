# Required files — LTX 2.5 default workflows

**Inventory first.** Do not assume a download is needed. LTX 2.5 weights are
often already on disk (Buddy `models/`, Comfy `models/`, another volume, or
the Hugging Face hub cache). Workflows stay in the default catalog and run
against **whatever accepted local name** Buddy finds.

1. Scan configured + common paths (below).
2. Every mandatory slot filled → proceed silently; wire loaders to those files.
3. Confirmed missing → ask (filename, dest folder, size, gated-HF note).
4. Download **only** that missing set, and only after `--yes` / `--fix-models`.

Buddy never auto-downloads gated weights. `setup --fix` does not fetch models.

## Where Buddy looks

- `MODELS_DIR` (default `video_buddy/models/`)
- `COMFYUI_ROOT/models`
- `EXTRA_MODELS_DIRS` / `LTX_MODELS_DIRS` (pathsep or comma — extra volumes)
- Comfy `extra_model_paths.yaml` `base_path` / folder entries
- `PROJECT_ROOT/models`, `ComfyUI/models`, portable Comfy `models/`
- `./models` relative to the current working directory
- Hugging Face hub snapshots: `HF_HOME` / `HUGGINGFACE_HUB_CACHE` /
  `~/.cache/huggingface/hub/models--Lightricks--LTX-2.5/`

Zero-byte placeholders count as **missing**.

```bash
cd video_buddy
python -m master_agent doctor                 # scan only (alias of setup) — never fetches
python -m master_agent download-models --ltx25   # scan + print ask if needed
python -m master_agent download-models --h3      # MiniMax H3 confirmed-missing only
# only if the scan listed confirmed-missing files:
python -m master_agent download-models --ltx25 --yes
python -m master_agent download-models --h3 --yes
python -m master_agent doctor --fix-models    # LTX 2.5 consent path (same as --ltx25 --yes)
```

## What `doctor` checks

`python -m master_agent doctor` (`setup`) prints `OK` / `NEED` for each row.
It does **not** download LTX weights. `setup --fix` installs venv / pip /
Playwright / `.env` / ffmpeg / Ollama models only.

| Row | Pass means |
|---|---|
| `python` | 3.10+ |
| `venv` | `.venv` exists |
| `pip` | `requirements.txt` frozen into that interpreter |
| `playwright` | package importable |
| `env` | `.env` present |
| `ffmpeg` | on PATH |
| `ollama` | on PATH and `qwen3-vl-heretic` + `nomic-embed-text` listed |
| `comfyui` | `COMFYUI_URL` `/system_stats` reachable |
| `vram-policy` | Shared 16GB pack policy (GGUF Q4/Q5 → NVFP4 → int8/fp8) |
| `ltx25-weights` | `ltx25_core` scan + loader pick (GGUF / NVFP4 / int8 / bf16) |
| `h3-weights` | `h3_fl2va` scan + loader pick (GGUF Q4_K / NVFP4 / int8) |

A typical ready line looks like:

`OK    ltx25-weights  GGUF Q4 (…-Q4_K_M.gguf) — 16GB-class preference #1; present (…)`

A duration-head that is **0 bytes** is not present. It is optional: no
shipped LTX 2.5 graph loads it, so a missing or zero-byte file does not
fail `ltx25-weights`. The ask lists it under optional weights.

Set extra trees in `.env` when weights live on another drive:

```
EXTRA_MODELS_DIRS=D:\ComfyUI\models
```

## 16GB-class Comfy install (reference names)

Typical files already under `video_buddy/models/` and/or `ComfyUI/models/`
on a real 16GB+ tower. **Any one transformer name satisfies the slot** —
official bf16 is not required. Doctor asks only for confirmed-missing or
zero-byte files.

| Path | Role |
|---|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | transformer (~20 GB) |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | transformer (~17 GB) |
| `diffusion_models/gguf/ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf` | transformer (~11 GB, preferred when present) |
| `text_encoders/gemma4-12b-heretic-ltx25-int8convrot.safetensors` | TE (alt to official `gemma4-12b-with-proj-…`) |
| `vae/ltx-2.5-video-vae-bf16.safetensors` / `ltx-2.5-video-vae-conv-bf16.safetensors` | video VAE |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | audio VAE |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | two-stage / official pack |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | optional distilled LoRA |
| `loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` | IC-LoRA stand-in (also Ingredients) |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | optional; 0 bytes = not present |

API graphs may also live under sibling `ltx_director/workflows/ltx-2.5/`
(same filenames). Buddy ships copies in `workflows/ltx-2.5/` and falls
back to that sibling tree.

## Names that already count as present

| Slot | Official Hub name (if you fetch later) | Already-on-disk names that also work |
|---|---|---|
| Transformer | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | GGUF Q4_K_M, NVFP4, `comfy-int8-convrot`, research stub `ltx-2.5-22b-distilled.safetensors` |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | Comfy int8, `gemma4-12b-heretic-ltx25-int8convrot.safetensors` |
| Video VAE | `ltx-2.5-video-vae-bf16.safetensors` | `ltx-2.5-video-vae-conv-bf16.safetensors` |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` | — |
| Duration head | `ltx-2.5-duration-head-bf16.safetensors` | optional (zero-byte is not present; does not block) |
| Spatial upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | — |
| IC-LoRA / MSR | `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` | pixel-spatial IC-LoRA, stubs `ltx-2.5-ic-lora.safetensors` / `ltx-2.5-msr.safetensors` |

**16GB-class GPU preference** (doctor + default loader, `VRAM_GB` default 16):

1. **GGUF Q4** when present (`UnetLoaderGGUF`)
2. Else **NVFP4** if `VRAM_GB` ≥ 14 (fits a ~16GB card)
3. Else **int8-convrot**, then official bf16

Set `VRAM_GB=12` in `.env` to skip the NVFP4 rung on a smaller card.
Official bf16 Gemma is not required if a working int8 / heretic TE is present.

Research JSON still says `ckpt_name: ltx-2.5-22b-distilled.safetensors` on
`CheckpointLoaderSimple`. Buddy remaps that stub and rewrites the loader to
`UNETLoader` / `UnetLoaderGGUF` + `LTXAVTextEncoderLoader`.

## Official Hub catalog (reference)

Gated [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)
(LTX-2.x Community License). This is the **name list**, not a download
checklist. Use it when doctor reports a slot empty.

| Hub path | Dest folder | Size |
|---|---|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `models/diffusion_models/` | 42.0 GB |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `models/text_encoders/` | 26.3 GB |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | `models/vae/` | 1.5 GB |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` | 365 MB |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | `models/model_patches/` | 3.8 MB, optional |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `models/latent_upscale_models/` | 996 MB |

IC-LoRA / MSR additionally use
[`Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients).

Optional Hub files (not blocking): distilled LoRA 450, temporal 2× upscaler.
Research placeholders `style.safetensors` / `camera-orbit.safetensors` are
bypassed when absent.

## Nodes (soft-fail if missing)

- `EmptyLTXVLatentVideo`, `LTXVEmptyLatentAudio`, `LTXVImgToVideo`, `LTXVLatentUpsampler`, `LTXVICLoRALoader`
- `ComfyUILTX25MSRICLoRALoader`, `ComfyUILTX25MSRMultiReferenceGuide` (MSR)
- `VHS_VideoCombine` (VideoHelperSuite)
- `UNETLoader` / `UnetLoaderGGUF`, `LTXAVTextEncoderLoader`, `CLIPTextEncode`, `KSampler`

Missing **nodes** produce a validator / Comfy error. Missing **weights**
produce the ask message. Workflow ids stay in the default menu either way.

## MiniMax H3 (default catalog — inventory first)

Omni video + native stereo audio. Modes: **fl2va** (`h3_t2v` / `h3_i2v` /
`h3_flf`) and **ref2va** (`h3_r2v`). ComfyUI ≥0.30 native nodes +
ComfyUI-GGUF. **CFG must stay 1.0** (distilled).

16GB-class defaults (`VRAM_GB=16`, RTX 5060 Ti class): **0.6–0.8 MP**
(template `1152×640`), **≤12 s** (17k+5 frame grid, 124 ≈ 5 s at 24 fps),
**4 steps** with LightX2V / official turbo LoRA when present. Enable
streaming / DynamicVRAM on Comfy 0.35.x.

```bash
python -m master_agent download-models --h3
python -m master_agent download-models --h3 --yes
python -m master_agent comfy run --mode generate --variant h3_t2v --prompt "BRIEF"
```

### 16GB-class names (any one transformer satisfies the slot)

| Path | Role |
|---|---|
| `diffusion_models/gguf/minimax_h3_fl2va_pruned-Q4_K.gguf` | fl2va DiT (~10.6 GiB, preferred) |
| `diffusion_models/gguf/minimax_h3_ref2va_pruned-Q4_K.gguf` | ref2va DiT (~10.6 GiB, preferred) |
| `diffusion_models/minimax_h3_*_pruned_nvfp4.safetensors` | DiT NVFP4 if present (`VRAM_GB` ≥ 14) |
| `diffusion_models/minimax_h3_*_pruned_int8_convrot.safetensors` | official Comfy int8 |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | TE (Blackwell). int8 / int4 convrot also count |
| `text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | TE last-resort (~17 GB — not the default) |
| `vae/minimax_h3_video_vae_fp16.safetensors` | video VAE |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | audio VAE |
| `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | optional 4-step turbo (bypassed if missing) |

**Preference** (doctor + default loader): GGUF Q4_K → NVFP4 → int8 → fp8/bf16.
Download target is the Unsloth GGUF + Comfy-Org TE/VAEs. Zero-byte files count
as missing. MiniMax H3 Community License:
https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

Hubs: [`unsloth/MiniMax-H3-GGUF`](https://huggingface.co/unsloth/MiniMax-H3-GGUF),
[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3).

Nodes: `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`, `UnetLoaderGGUF`,
`CLIPLoader` (`type=minimax`), `VAEDecode` + `VAEDecodeAudio`, `CreateVideo`.

## Other families (same 16GB pick, attested names only)

`python -m master_agent workflows --vram` is the table. Download scans
(never auto-fetch):

```bash
python -m master_agent download-models --wan
python -m master_agent download-models --vace
python -m master_agent download-models --krea
python -m master_agent download-models --qwen
python -m master_agent download-models --flux-pack
```

| Family | 16GB default (when present) | Hub (attested) |
|---|---|---|
| Wan 2.2 | `Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf` + Lightx2v | QuantStack `HighNoise/` + `LowNoise/` Q4_K_S; fp8 fallback Comfy-Org |
| VACE | `wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1-Q4_K_M.gguf` | mickmumpitz/VACE_Skyreels_V3_R2V_Merge-GGUF |
| Krea-2 | `krea2_turbo_nvfp4.safetensors` | Comfy-Org/Krea-2 |
| Flux | `flux1-dev-Q4_K_S.gguf` (else `flux1-dev-fp8`) | city96/FLUX.1-dev-gguf; Comfy-Org/flux1-dev |
| Qwen Edit | `Qwen-Image-Edit-2509-Q5_0.gguf` | QuantStack/Qwen-Image-Edit-2509-GGUF |
| LTX 2.3 | EROS baked all-in-one | QuantStack/LTX-2.3-GGUF is optional / tight |

K3NK AIO I2V: **no attested pack** (Hub search only found LoRAs). Not a default.
Do not invent filenames. Heavy graphs (Movie Builder, CCC ADV, AI-VFX 1.0)
print a prepare warning and point at a safer alternate.
