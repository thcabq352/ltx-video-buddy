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
python -m master_agent doctor                 # scan only (alias of setup)
python -m master_agent download-models --ltx25   # scan + print ask if needed
# only if the scan listed confirmed-missing files:
python -m master_agent download-models --ltx25 --yes
python -m master_agent doctor --fix-models
```

Set extra trees in `.env` when weights live on another drive:

```
EXTRA_MODELS_DIRS=D:\ComfyUI\models
```

## Names that already count as present

| Slot | Official Hub name (if you fetch later) | Already-on-disk names that also work |
|---|---|---|
| Transformer | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | GGUF Q4_K_M, NVFP4, `comfy-int8-convrot`, research stub `ltx-2.5-22b-distilled.safetensors` |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | Comfy int8, `gemma4-12b-heretic-ltx25-int8convrot.safetensors` |
| Video VAE | `ltx-2.5-video-vae-bf16.safetensors` | `ltx-2.5-video-vae-conv-bf16.safetensors` |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` | — |
| Duration head | `ltx-2.5-duration-head-bf16.safetensors` | — |
| Spatial upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | — |
| IC-LoRA / MSR | `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` | stubs `ltx-2.5-ic-lora.safetensors` / `ltx-2.5-msr.safetensors` |

When several transformers exist, loaders prefer **GGUF Q4 → NVFP4 → int8-convrot → bf16**.
GGUF files use `UnetLoaderGGUF`. Official bf16 Gemma is not required if a
working int8 / heretic TE is present.

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
