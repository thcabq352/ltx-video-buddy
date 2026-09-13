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
# only if the scan listed confirmed-missing files:
python -m master_agent download-models --ltx25 --yes
python -m master_agent doctor --fix-models    # same consent path as --yes
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
| `ltx25-weights` | `ltx25_core` scan + loader pick (GGUF / NVFP4 / int8 / bf16) |

A typical ready line looks like:

`OK    ltx25-weights  GGUF Q4 (…-Q4_K_M.gguf) — 16GB-class preference #1; present (…)`

A duration-head that is **0 bytes** fails this row even when every other
slot is filled. The hint then points at `download-models --ltx25`, not
`--fix`.

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
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | **mandatory; 0 bytes = missing** |

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
| Duration head | `ltx-2.5-duration-head-bf16.safetensors` | — (zero-byte is missing) |
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
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | `models/model_patches/` | 3.8 MB |
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
