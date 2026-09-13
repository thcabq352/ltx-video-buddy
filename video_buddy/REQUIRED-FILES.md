# Required files — LTX 2.5 default workflows

This is the **product** checklist for anyone running Video Buddy. It is not a
one-off inventory of a single machine. Buddy **scans first**, then **asks**
before downloading. It never auto-downloads gated weights.

Official split pack: gated Hugging Face repo
[`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)
(LTX-2.x Community License). Accept the license, then `huggingface-cli login`
or set `HF_TOKEN`.

Research-agent JSON still says `ltx-2.5-22b-distilled.safetensors` on
`CheckpointLoaderSimple`. Buddy remaps that stub to the official Comfy
transformer and rewrites the loader to `UNETLoader` + `LTXAVTextEncoderLoader`
so the graphs run against the real split pack.

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
only for files the scan confirmed are absent.

## Mandatory (default usability)

Hub download targets (gated `Lightricks/LTX-2.5`) plus local alternatives that
already satisfy the slot on a typical 16GB-class Comfy install:

| Slot | Hub download target | Also accepted locally | Dest | Size | Workflows |
|---|---|---|---|---|---|
| Transformer | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | **GGUF Q4_K_M** (preferred when present), NVFP4, official bf16, research stub `ltx-2.5-22b-distilled.safetensors` | `models/diffusion_models/` (GGUF often in `…/gguf/`) | ~21.5 GB int8 / ~17 GB NVFP4 / ~11 GB Q4 | all LTX 2.5 |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `gemma4-12b-heretic-ltx25-int8convrot.safetensors`, official `…-bf16.safetensors` | `models/text_encoders/` | ~15.4 GB | all LTX 2.5 |
| Video VAE | `ltx-2.5-video-vae-bf16.safetensors` | `ltx-2.5-video-vae-conv-bf16.safetensors` | `models/vae/` | ~1.5 GB | all LTX 2.5 |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` | — | `models/vae/` | ~365 MB | all LTX 2.5 (T2A/A2V) |
| Spatial upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | — | `models/latent_upscale_models/` | ~1.0 GB | `ltx25_t2v_i2v_two_stage` |
| IC-LoRA | `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` | research stubs `ltx-2.5-ic-lora.safetensors` / `ltx-2.5-msr.safetensors` | `models/loras/` | ~1.3 GB | `ltx25_v2v_ic_lora`, `ltx25_msr` |

Loader preference when several transformers exist: **GGUF Q4 → NVFP4 → int8-convrot → bf16**. GGUF files are wired to `UnetLoaderGGUF`. Official bf16 Gemma is **not** required if a working int8 / heretic TE is present.

## Optional

| File | Dest folder | Notes |
|---|---|---|
| `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | `models/loras/` | Distilled LoRA 450. Not required for the int8 transformer pack. |
| `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | `models/latent_upscale_models/` | Temporal 2×. |
| `ltx-2.5-duration-head-bf16.safetensors` | `models/model_patches/` | Auto-duration patch. |
| `style.safetensors` / `camera-orbit.safetensors` | `models/loras/` | Research-template placeholders. Buddy bypasses these LoRA nodes if the file is absent. |

`--optional` on `download-models` fetches the official optional Hub files, not user style LoRAs.

## Stub name map (research JSON → official)

| Template widget | Official filename |
|---|---|
| `ltx-2.5-22b-distilled.safetensors` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` |
| `ltx-2.5-ic-lora.safetensors` | `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` |
| `ltx-2.5-msr.safetensors` | same IC-LoRA Ingredients file (no public Lightricks file named “msr”) |

A file dropped under the **stub** name still counts as present.

## Nodes (soft-fail if missing)

- `EmptyLTXVLatentVideo`, `LTXVEmptyLatentAudio`, `LTXVImgToVideo`, `LTXVLatentUpsampler`, `LTXVICLoRALoader`
- `ComfyUILTX25MSRICLoRALoader`, `ComfyUILTX25MSRMultiReferenceGuide` (MSR)
- `VHS_VideoCombine` (VideoHelperSuite)
- `UNETLoader`, `LTXAVTextEncoderLoader`, `CLIPTextEncode`, `KSampler`

Missing **nodes** produce a clear validator / Comfy error. Missing **weights**
produce the ask-to-download message. Workflows stay in the default menu either way.

## Scan roots

Buddy looks in `MODELS_DIR`, `COMFYUI_ROOT/models`, `PROJECT_ROOT/models`,
`PROJECT_ROOT/ComfyUI/models`, the portable Comfy tree, and `./models` relative
to the current working directory.
