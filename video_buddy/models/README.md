# models/

This folder is `MODELS_DIR` by default (set `MODELS_DIR` in `.env` to use another drive). Buddy passes it to ComfyUI through `state/extra_model_paths.yaml`, so ComfyUI sees every file here. Weights are not in git.

Download sources, the scan order, consent, and the loader order are in [`docs/WEIGHTS.md`](../../docs/WEIGHTS.md).

## Layout

`python -m master_agent setup --fix` (run by plain `python install.py`) and `python install.py --automatic-install --yes` create these empty folders. Nothing is downloaded by making them. `automatic-install --dry-run` lists the `mkdir` lines without creating anything. The list lives in `master_agent/models/layout.py`.

| Folder | What goes there |
|---|---|
| `checkpoints/` | All-in-one checkpoints (`CheckpointLoaderSimple`): LTX 2.3 distilled 1.1, EROS |
| `diffusion_models/` | Transformers / UNETs, including GGUF (`UNETLoader`, `UnetLoaderGGUF`): LTX 2.3 Q4_K_S, LTX 2.5, MiniMax H3, Wan, Flux, Krea, Qwen Edit |
| `unet/` | Older name for `diffusion_models/`. ComfyUI reads both |
| `text_encoders/` | Text encoders (`CLIPLoader`, `DualCLIPLoader`, `LTXAVTextEncoderLoader`): Gemma, Qwen3-VL, umt5 |
| `clip/` | Older name for `text_encoders/`. ComfyUI reads both |
| `vae/` | Video, audio, and image VAEs (`VAELoader`, `LTXVAudioVAELoader`) |
| `loras/` | LoRAs and IC-LoRAs (`LoraLoaderModelOnly`, `LTXICLoRALoaderModelOnly`) |
| `latent_upscale_models/` | LTX latent spatial / temporal upscalers. `LatentUpscaleModelLoader` reads only this folder |
| `upscale_models/` | Pixel-space upscalers (`UpscaleModelLoader`) |
| `model_patches/` | Model patches: the optional LTX 2.5 duration head |
| `clip_vision/` | CLIP vision encoders (IPAdapter graphs) |
| `controlnet/` | ControlNets (`ControlNetLoaderAdvanced`) |

Local LLM GGUFs for llama.cpp (`LLAMACPP_MODEL`, `KB_EMBED_MODEL`) go in a folder named after the model: `models/qwen3-vl-heretic/qwen3-vl-4b-heretic-Q4_K_M.gguf` and `models/nomic-embed-text/nomic-embed-text-v1.5.Q4_K_M.gguf`. `automatic-install --yes` downloads those two public files. They are not Comfy weights. See [`docs/WEIGHTS.md`](../../docs/WEIGHTS.md#local-llm-ggufs).

The legacy `state/download_models.py` also writes `SEEDVR2/` here for the SeedVR2 upscaler custom node.

## The LTX 2.3 pack

What `python install.py --automatic-install --yes` (or `python -m master_agent download-models --ltx23 --yes`) puts here:

| File | Folder |
|---|---|
| `ltx-2.3-22b-distilled-1.1.safetensors` | `checkpoints/` |
| `LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf` | `diffusion_models/` |
| `gemma_3_12B_it_fp4_mixed.safetensors` | `text_encoders/` |
| `ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors` | `loras/` |
| `LTX23_video_vae_bf16.safetensors` | `vae/` |
| `taeltx2_3.safetensors` | `vae/` |

`python -m master_agent doctor` names the exact folder for anything still missing. The pre-slim filename list is [`docs/archive/video_buddy/models/README.md`](../../docs/archive/video_buddy/models/README.md).
