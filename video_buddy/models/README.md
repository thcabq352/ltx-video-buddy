# Model inventory (16GB / RTX 5060 Ti)

Weights live here so Video Buddy and ComfyUI (via `extra_model_paths.yaml`) can see them. **Not in git.**

## LTX 2.5 (default catalog — inventory first)

Do **not** assume a download. `python -m master_agent doctor` scans this tree plus Comfy `models/`, `EXTRA_MODELS_DIRS`, `extra_model_paths.yaml`, and the HF hub cache. It reports the 16GB-class loader pick and **does not fetch**.

Preference when several transformers exist (`VRAM_GB` default 16):

1. `diffusion_models/gguf/ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf`
2. `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` (if `VRAM_GB` ≥ 14)
3. `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
4. official `ltx-2.5-22b-distilled-transformer-bf16.safetensors`

Text encoder: `gemma4-12b-heretic-ltx25-int8convrot.safetensors` or Comfy int8 counts — official bf16 Gemma is not required. The duration head is optional (no shipped 2.5 graph loads it). A zero-byte `model_patches/ltx-2.5-duration-head-bf16.safetensors` is not present and does not fail the core pack.

```bash
python -m master_agent doctor
python -m master_agent download-models --ltx25        # list confirmed-missing
python -m master_agent download-models --ltx25 --yes  # after you agree
```

Full accepted-name table: [`../REQUIRED-FILES.md`](../REQUIRED-FILES.md).

## MiniMax H3 (default catalog — inventory first)

fl2va (`h3_t2v` / `h3_i2v` / `h3_flf`) and ref2va (`h3_r2v`). CFG **1.0**.
16GB sweet spot: 0.6–0.8 MP, ≤12 s, 4 steps. Prefer Unsloth GGUF Q4_K DiT +
Comfy TE (NVFP4 AWQ or int8/int4). GGUF TE Q4_K_M is last-resort.

```bash
python -m master_agent download-models --h3
python -m master_agent download-models --h3 --yes
```

| Path | Role |
|---|---|
| `diffusion_models/gguf/minimax_h3_fl2va_pruned-Q4_K.gguf` | fl2va DiT |
| `diffusion_models/gguf/minimax_h3_ref2va_pruned-Q4_K.gguf` | ref2va DiT |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | TE |
| `vae/minimax_h3_video_vae_fp16.safetensors` | video VAE |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | audio VAE |

## LTX 2.3 / Wan / Mickmumpitz (still used)

## Expected layout

| Subfolder | Role |
|-----------|------|
| `diffusion_models/` | FP8 / NVFP4 transformers (base, dev, EROS) |
| `loras/` | Distilled LoRA, IC-LoRA LipDub |
| `vae/` | Video VAE + audio VAE |
| `text_encoders/` | Gemma 3 12B FP4 mixed |
| `checkpoints/` | Optional all-in-one checkpoints |

## Filenames the agent looks for

### base
- `diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors`
- `vae/taeltx2_3.safetensors`
- `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors`

### eros
- `diffusion_models/ltx-2.3-22b-distilled-10-eros_fp8.safetensors`

### directors
- `diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors`
- `loras/ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors`

### lipsync
- Dev FP8 + `loras/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors`
- `vae/LTX23_audio_vae_bf16.safetensors`

### wan22 (Mickmumpitz Wan 2.2 variant)
- `diffusion_models/wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`
- `diffusion_models/wan/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors`
- `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `vae/wan_2.1_vae.safetensors`
- `loras/wan/` — lightx2v distill, Instareal high/low, stock_photography, Lenovo

### krea2_img / ideogram / flux-klein (Mickmumpitz image graphs)
- `diffusion_models/krea2_turbo_nvfp4.safetensors` + `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
- `diffusion_models/ideogram4*_fp8_scaled.safetensors` + `text_encoders/qwen3vl_8b_fp8_scaled.safetensors`

### utility models
- `SEEDVR2/` (video upscaler: ema_vae_fp16 + seedvr2_ema_7b_sharp GGUF)
- `ultralytics/bbox/face_yolov8m.pt` (Impact Pack face detailer)

### qwen_edit_360 / aivfx_startimage (Qwen-Image-Edit)
- `text_encoders/qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `vae/qwen/qwen_image_vae.safetensors`
- `diffusion_models/gguf/Qwen-Image-Edit-2509-Q5_0.gguf` and `qwen-image-edit-2511-Q5_0.gguf`
  (GGUF loaders read `unet/`, which `extra_model_paths.yaml` aliases to `diffusion_models/`)
- `loras/qwen/` — Lightning 4-step LoRAs + `251018_MICKMUMPITZ_QWEN-EDIT_360_03`

### aivfx / ai_renderer (Wan VACE)
- `diffusion_models/wan/wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors`
  (no public fp32 build exists — the mirror file with that name is an fp16 rename)
- `diffusion_models/gguf/wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1-Q4_K_M.gguf` (v1.3 GGUF option)
- `loras/wan/Wan2.1_T2V_14B_FusionX_LoRA.safetensors`

### ltx23_lipsync_v08 / ai-render examples
- `checkpoints/ltx-2.3-22b-dev-fp8.safetensors` + `ltx-2.3-22b-distilled-fp8.safetensors`
- `loras/ltx/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` + `loras/ltx/ltx-2-19b-ic-lora-union-control-ref0.5.safetensors`
- `loras/ltx-2-19b-ic-lora-detailer.safetensors` + `loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors`

### ccc41_krea2 (CCC 4.1 Krea2-Edit)
- `diffusion_models/krea2_turbo_fp8_scaled.safetensors` + `text_encoders/qwen3vl_4b_bf16.safetensors`
- `loras/krea/krea2_identity_edit_v1_2.safetensors`

### zimage_turbo_cn
- `diffusion_models/z_image/z_image_turbo_bf16.safetensors` + `text_encoders/qwen_3_4b.safetensors`
- `model_patches/Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors`

Mickmumpitz / LTX 2.3 / Wan packs (legacy filenames above) are scripted:

```bash
python state/download_models.py
```

LTX 2.5 is **not** that script. Use `doctor` + `download-models --ltx25` (gated Hub pack, consent-gated). There is no `scripts/download_ltx_models.ps1`.

Do not commit `.safetensors` files to git.
