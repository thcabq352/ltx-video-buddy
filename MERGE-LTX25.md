# MERGE-LTX25 — research agent → Video Buddy

Upstream reference (read-only): https://github.com/thcabq352/ltx2.5-research-agent

Buddy stays the Comfy / video driver. The LangGraph research / scrape / A2A /
Gradio harness was **not** copied. Secrets and `.env` were **not** copied.

## What was copied

| Source (research agent) | Destination (Buddy) |
|---|---|
| `workflows/ltx-2.5/*.json` (7 API graphs) | `video_buddy/workflows/ltx-2.5/` |
| `workflows/ltx-2.5/README.md` | same + Buddy invocation notes |
| Load / patch / LoRA / duration patterns from `src/ltx_research_agent/comfy/{workflows,duration,loras}.py` | Adapted into `master_agent/comfy/workflow_patcher.py`, `config.snap_ltx_frames`, `master_agent/comfy/catalog.py` |
| Motion / I2V prompt rules | `video_buddy/master_agent/storyboard/prompts/motion_i2v.md` |
| Official weight filenames (HF split pack) | `master_agent/models/weights.py` + `REQUIRED-FILES.md` |

Not copied: `graph.py`, agents, scrape, UI, A2A harness, OAuth, Imagine client,
the `ltx_research_agent` package name.

## Default catalog (no env flags)

`python -m master_agent workflows` lists every default variant, including:

| Buddy id | Research id | File |
|---|---|---|
| `ltx25_t2v_i2v` | `t2v_i2v` | `LTX-2.5_T2V_I2V_Single_Stage_Distilled_api.json` |
| `ltx25_t2v_i2v_two_stage` | `t2v_i2v_two_stage` | `LTX-2.5_T2V_I2V_Two_Stage_Distilled_api.json` |
| `ltx25_flf2v` | `flf2v` | `LTX-2.5_FLF2V_api.json` |
| `ltx25_msr` | `msr` | `LTX-2.5_MSR_Multi_Reference_api.json` |
| `ltx25_v2v_ic_lora` | `v2v_ic_lora` | `LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled_api.json` |
| `ltx25_a2v` | `a2v` | `LTX-2.5_A2V_Two_Stage_Distilled_api.json` |
| `ltx25_t2a` | `t2a` | `LTX-2.5_T2A_Single_Stage_Distilled_api.json` |

Existing `base` / `eros` / `directors` / `lipsync` / `wan22` stay on the same
path. WAN / K3NK / TeaCache behavior is unchanged (TeaCache still soft-bypass).

New `*_api.json` under `workflows/` auto-registers. LTX 2.5 ids are always in
the default menu even if tower nodes or weights are missing.

## How to invoke (same path as existing I2V/T2V)

```bash
cd video_buddy

# list (CLI + GET /api/variants + Create-tab + Comfy-tab)
python -m master_agent workflows

# generate → patch (prompt, seed, 8n+1 frames) → lint → queue
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain"
python -m master_agent comfy run --mode generate --variant ltx25_flf2v --prompt "first to last"
python -m master_agent run "LTX 2.5 alley push-in" --variant ltx25_t2v_i2v --no-interview
python -m master_agent run "interpolate these frames" --variant flf2v --image start.png --no-interview

# prepare / dry-run does not require weights
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "x" --prepare
```

Studio UI: Create → Variant dropdown and Comfy → generate mode both load
`GET /api/variants` (default catalog, no experimental gate).

## Frame / duration

Buddy Rainey stop-lines win: `snap_ltx_frames()` (8n+1, min 9). LTX 2.5
variants use `fps=24`, `frame_snap=8`. Audio `frames_number` stays paired
with video `length`.

## Weights (scan first, ask, then download)

See [`video_buddy/REQUIRED-FILES.md`](video_buddy/REQUIRED-FILES.md).

Official Hub defaults (gated [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)):

- `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`
- `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`
- `vae/ltx-2.5-video-vae-bf16.safetensors`
- `vae/ltx-2.5-audio-vae-bf16.safetensors`
- `model_patches/ltx-2.5-duration-head-bf16.safetensors`
- `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`

Research JSON still uses stub `ltx-2.5-22b-distilled.safetensors` on
`CheckpointLoaderSimple`. Buddy remaps that to the official bf16 transformer
and rewrites the loader. 16GB-class files (`comfy-int8-convrot`, `nvfp4`,
GGUF Q4) satisfy the transformer slot when already on disk.

1. Scan configured + common Comfy `models/` trees and Hugging Face hub snapshots.
2. Any accepted local name → queue silently. Zero-byte files are missing.
3. Truly missing → `MissingWeightsError` with filename, dest folder, size, gated-HF note, and accepted alternatives.
4. User agrees: `python -m master_agent download-models --ltx25 --yes` (fetches official **bf16** Hub names).

Do not treat download as the default path. `setup --fix` / `doctor --fix` does **not** download models.
`setup --fix-models` / `doctor --fix-models` is explicit consent for the **missing** set only.

When several transformers are on disk, the patcher prefers GGUF Q4, then NVFP4, then int8-convrot, then official bf16.

## Tower / node prerequisites (from research README + official pack)

- ComfyUI with LTX 2.5 nodes (`ComfyUI-LTXVideo` / native 0.32+ LTX 2.5).
- VideoHelperSuite (`VHS_VideoCombine`).
- MSR nodes: `ComfyUILTX25MSRICLoRALoader`, `ComfyUILTX25MSRMultiReferenceGuide`.
- Official bf16 distilled 22B transformer + Gemma 4 TE + video/audio VAEs + duration head + spatial upscaler (see required files).
- 16GB-class towers may use int8-convrot / NVFP4 / GGUF instead of the 42 GB bf16 transformer.
- IC-LoRA / MSR also need `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors`.

Leftover LangGraph / research / Imagine / A2A pieces stay in the research repo.
