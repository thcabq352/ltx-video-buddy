# Required files — LTX 2.5 default workflows

This is the **product** checklist for anyone running Video Buddy. It is not a
one-off inventory of a single machine. Buddy **scans first**, then **asks**
before downloading. It never auto-downloads gated weights.

Official split pack: gated Hugging Face repo
[`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)
(LTX-2.x Community License). Accept the license, then `huggingface-cli login`
or set `HF_TOKEN`.

Research-agent JSON still says `ckpt_name: ltx-2.5-22b-distilled.safetensors`
on `CheckpointLoaderSimple` (a placeholder monolith). Buddy **intentionally
remaps** that stub to the official split pack and rewrites the loader to
`UNETLoader` / `UnetLoaderGGUF` + `LTXAVTextEncoderLoader` so the graphs run
against the real Hub files below.

## Official Hub pack (mandatory)

Paths are relative to the repo root of `Lightricks/LTX-2.5`. Sizes from the
Hub listing (Sep 2026).

| Hub path | Dest folder | Size | Notes |
|---|---|---|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `models/diffusion_models/` | 42.0 GB | Official default. 16GB-class: `…-comfy-int8-convrot.safetensors` (21.5 GB) or `…-nvfp4.safetensors` (18.7 GB). Local GGUF Q4 also counts. |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `models/text_encoders/` | 26.3 GB | Official default. Comfy int8 (`…-comfy-int8-convrot.safetensors`, 15.4 GB) or heretic int8 also count. |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | `models/vae/` | 1.5 GB | `…-conv-bf16.safetensors` also counts. |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` | 365 MB | Required for T2A / A2V and AV joint graphs. |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | `models/model_patches/` | 3.8 MB | Official pack. Zero-byte placeholders count as **missing**. |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `models/latent_upscale_models/` | 996 MB | Official pack (two-stage uses it). |

IC-LoRA / MSR additionally need
`ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` from gated
[`Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients)
→ `models/loras/`.

## One-command flow

```bash
cd video_buddy
python -m master_agent setup                  # scan (includes ltx25-weights)
python -m master_agent doctor                 # same command (alias)
python -m master_agent download-models --ltx25
# review the missing list, then:
python -m master_agent download-models --ltx25 --yes
# or explicit consent via setup/doctor:
python -m master_agent doctor --fix-models
```

If any accepted local name for a slot is already in `MODELS_DIR`,
`COMFYUI_ROOT/models`, a common relative `ComfyUI/models` folder, or a
Hugging Face hub snapshot (`~/.cache/huggingface/hub/models--Lightricks--LTX-2.5/…`),
Buddy proceeds silently. Zero-byte placeholders count as **missing**.

Inventory first. Do not download “just in case.” `--yes` / `--fix-models` is
only for files the scan confirmed are absent. The download helper fetches the
**official bf16 Hub names** above, not a 16GB stand-in.

Loader preference when several transformers already exist locally:
**GGUF Q4 → NVFP4 → int8-convrot → official bf16**. GGUF files are wired to
`UnetLoaderGGUF`. Official bf16 Gemma is not required if a working int8 /
heretic TE is present.

## Optional (same Hub repo)

| Hub path | Dest folder | Notes |
|---|---|---|
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | `models/loras/` | Distilled LoRA 450. |
| `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | `models/latent_upscale_models/` | Temporal 2×. |
| `style.safetensors` / `camera-orbit.safetensors` | `models/loras/` | Research-template placeholders. Buddy bypasses these LoRA nodes if the file is absent. |

`--optional` on `download-models` fetches the official optional Hub files, not user style LoRAs.

## Stub name map (research JSON → official)

| Template widget | Official Hub filename |
|---|---|
| `ltx-2.5-22b-distilled.safetensors` | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` |
| `ltx-2.5-ic-lora.safetensors` | `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` |
| `ltx-2.5-msr.safetensors` | same IC-LoRA Ingredients file (no public Lightricks file named “msr”) |

A file dropped under the **stub** name still counts as present.

## Nodes (soft-fail if missing)

- `EmptyLTXVLatentVideo`, `LTXVEmptyLatentAudio`, `LTXVImgToVideo`, `LTXVLatentUpsampler`, `LTXVICLoRALoader`
- `ComfyUILTX25MSRICLoRALoader`, `ComfyUILTX25MSRMultiReferenceGuide` (MSR)
- `VHS_VideoCombine` (VideoHelperSuite)
- `UNETLoader` / `UnetLoaderGGUF`, `LTXAVTextEncoderLoader`, `CLIPTextEncode`, `KSampler`

Missing **nodes** produce a clear validator / Comfy error. Missing **weights**
produce the ask-to-download message. Workflows stay in the default menu either way.

## Scan roots

Buddy looks in `MODELS_DIR`, `COMFYUI_ROOT/models`, `PROJECT_ROOT/models`,
`PROJECT_ROOT/ComfyUI/models`, the portable Comfy tree, `./models` relative
to the current working directory, and Hugging Face hub snapshots for
`Lightricks/LTX-2.5`.
