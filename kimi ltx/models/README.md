# LTX 2.3 model inventory (16GB / RTX 5060 Ti)

Weights live here so both the LangGraph agent and ComfyUI (via `extra_model_paths.yaml`) can see them.

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

Mickmumpitz-set downloads are scripted (resumable, re-runnable):

```bash
python state/download_models.py
```

Download LTX weights with:

```powershell
.\scripts\download_ltx_models.ps1
```

Do not commit `.safetensors` files to git.
