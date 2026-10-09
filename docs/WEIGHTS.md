# Weights

Inventory first. This file is the only home for consent and the loader order. Other docs link [here](#loader-policy).

Buddy scans, in order:

1. `MODELS_DIR` (default `video_buddy/models/`)
2. `COMFYUI_ROOT/models`
3. `EXTRA_MODELS_DIRS` / `LTX_MODELS_DIRS` (path separator or comma)
4. Comfy `extra_model_paths.yaml`, including `video_buddy/state/extra_model_paths.yaml`
5. `PROJECT_ROOT/models`, portable Comfy `models/`, `./models`
6. Hugging Face hub snapshots (`HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `~/.cache/huggingface/hub/`)

A file in `MODELS_DIR` wins over the same name elsewhere. Zero-byte files count as missing. Weights are not in git.

```bash
cd video_buddy
python -m master_agent inventory
python -m master_agent doctor
python -m master_agent download-models --ltx25
python -m master_agent download-models --h3
python -m master_agent workflows --vram
```

## Consent

`doctor` (alias `setup`) reports. It does not download video weights.

| Flag | Effect |
|---|---|
| *(none)* | Scan and print `OK` / `NEED` |
| `--scan-only` | Report only. Does not fetch weights or pull Ollama |
| `--use-existing` | Keep files already on disk. Does not download |
| `--fix` | Venv, pip, Playwright, `.env`, ffmpeg, and Ollama models missing from `ollama list`. Still no video weights. Ollama pulls need `y` or `--yes` |
| `--fix-models` | After the scan, fetch confirmed-missing LTX 2.5 weights. Explicit consent |
| `download-models` | List confirmed-missing slots. Exit 2 if something is missing. Nothing downloaded |
| `download-models --yes` | Fetch that missing set only |
| `--download` | Fetch after `--yes` or a y/n prompt |

`setup --fix` does not fetch models. `--scan-only` overrides `--yes`. Pack pulls write only under `MODELS_DIR`. `HEARTMULA_MODELS_DIR`, when set, must stay inside `MODELS_DIR`.

doctor never fetches weights.

Gated Hub packs need `HF_TOKEN` or `huggingface-cli login`. LTX 2.5 is the LTX-2.x Community License on [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5).

## What doctor prints

| Row | Pass means |
|---|---|
| `python` | 3.10+ |
| `venv` | project `.venv` exists |
| `pip` | `requirements.txt` installed |
| `playwright` | package importable |
| `env` | `.env` exists |
| `ffmpeg` | on PATH |
| `ollama` | on PATH and `qwen3-vl-heretic` + `nomic-embed-text` listed (optional if llama.cpp is the local LLM) |
| `comfyui` | `COMFYUI_URL` `/system_stats` answers |
| `hardware` | One routing sentence. Does not block install |
| `vram-policy` | The loader order below |
| `ltx25-weights` | `ltx25_core` scan plus the loader pick |
| `h3-weights` | `h3_fl2va` scan plus the loader pick |
| `heartlib` / `heartmula-weights` / `heartmula-comfy` | Reported. A NEED here does not fail doctor and does not fetch |

A ready transformer line names the file the default loader will wire. A zero-byte duration head is optional: no shipped LTX 2.5 graph loads it, so it does not fail `ltx25-weights`.

Hardware sentence (also on managed `comfy start` / `status` / `restart`): NVIDIA at 12GB and above can use full checkpoints; 8–12GB and below, and no-GPU, are told to use a quantized pack or CPU. AMD with ROCm uses the same bands. AMD without ROCm is told to install ROCm first. `VRAM_GB`, when set, is the number in that sentence.

## Loader policy

When several transformers for one slot are already on disk, any VRAM:

1. **GGUF Q4** when a compatible file is on disk (`UnetLoaderGGUF`). Order inside that bucket: QuantStack Q4_K_S, then Sulphur `sulphur_dev-Q3_K_S.gguf`, then the other recognized GGUF aliases (including Q5). A machine that already has GGUF Q4 and NVFP4 loads GGUF Q4. `FORCE_LOADER` is not required for that win.
2. Else **NVFP4**, only if `VRAM_GB` ≥ 14.
3. Else **int8 / fp8**, then official bf16 or the EROS all-in-one.

`VRAM_GB` comes from the environment, else `nvidia-smi`, else 16. Below 14GB, doctor does not suggest NVFP4 or bf16. A heavy file that is already on disk still counts as present, so Buddy does not download a second copy. Leave `FORCE_LOADER` empty when the only local transformer is bf16 or EROS. Those files still load.

Research JSON may still say `ckpt_name: ltx-2.5-22b-distilled.safetensors` on `CheckpointLoaderSimple`. Buddy remaps that stub to `UNETLoader` / `UnetLoaderGGUF` and `LTXAVTextEncoderLoader`.

This order is shared by Hands, doctor, and the loaders (`master_agent.models.vram_policy`). `workflows --vram` prints the family table. The director does not use the table to pick a cheaper graph. See [Architecture](ARCHITECTURE.md#brain-and-hands).

K3NK AIO has no attested pack. It is retired. See [Audit](AUDIT.md).

### 12GB and local LLM

RTX 4000 Ada (Admeria / rainey1). Put this in `video_buddy/.env`:

```bash
VRAM_GB=12
FORCE_LOADER=gguf
LLM_PROVIDER=ollama
LLM_PANEL=local
PANEL_JUDGE=ollama
```

`FORCE_LOADER=gguf` hides NVFP4 and bf16 from suggestions. It does not block a machine whose only transformer is bf16. `LLM_PROVIDER=auto` stays local (llama.cpp, then Ollama); with both down, LLM calls fail instead of reaching Grok.

LoRA A/B locks the text encoder. The proven LTX 2.3 family is `gemma_3_12B_it_fp8_scaled`. Do not swap it for a Heretic encoder mid-comparison. Only the LoRA name and strength change. Windows folder-prefixed names stay backslash style (`wan\file.safetensors`).

## Accepted names

Any one transformer name fills the slot. Official bf16 Gemma is not required when a heretic or int8 encoder is present. A Gemma filename containing `heretic` counts.

| Slot | Also counts |
|---|---|
| LTX 2.5 transformer | `ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf`, NVFP4, `comfy-int8-convrot`, research stub `ltx-2.5-22b-distilled.safetensors`, official bf16 |
| LTX 2.5 text encoder | `gemma4-12b-heretic-ltx25-int8convrot.safetensors`, Comfy int8, other `*heretic*` Gemma, official `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` |
| LTX 2.5 video VAE | `ltx-2.5-video-vae-bf16.safetensors`, `ltx-2.5-video-vae-conv-bf16.safetensors` |
| LTX 2.5 audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` |
| Duration head | `ltx-2.5-duration-head-bf16.safetensors` (optional) |
| Spatial upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |
| IC-LoRA / MSR | Ingredients LoRA, pixel-spatial IC-LoRA, stubs `ltx-2.5-ic-lora.safetensors` / `ltx-2.5-msr.safetensors` |
| LTX 2.3 diffusion | QuantStack `LTX-2.3-*-Q4_K_S.gguf`, Sulphur Q3, other recognized Q3/Q4/Q5 names. fp8 and the EROS checkpoint are the fallback and are not a new download pack. EROS stays for VAE and text projection |
| H3 DiT | `minimax_h3_fl2va_pruned-Q4_K.gguf`, `minimax_h3_ref2va_pruned-Q4_K.gguf`, then NVFP4, then int8 |
| H3 text encoder | Comfy Qwen3-VL first: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, else int8 / int4. `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` is last resort (~17GB) and is not the default |
| H3 VAEs | `minimax_h3_video_vae_fp16.safetensors`, `minimax_h3_audio_vae_fp32.safetensors` |

H3 CFG stays 1.0. On a 16GB-class card the sweet spot is 0.6–0.8 MP, at most 12 seconds, 4 steps. Frame grid is 17k+5 (124 ≈ 5s at 24 fps). Do not use LTX `8n+1` lengths on H3 graphs.

`download-models` bundles: `ltx25_core`, `ltx25_two_stage`, `ltx25_iclora`, `ltx25_msr`, `ltx25_all`, `h3_fl2va`, `h3_ref2va`, `h3_all`. `--optional` also lists the distilled LoRA 450 and the temporal upscaler. Family scans that only list missing files until `--yes`: `--wan`, `--vace`, `--krea`, `--qwen`, `--flux-pack`, `--heartmula`.

Legacy Mickmumpitz / LTX 2.3 filenames still used by older graphs are installed with `python state/download_models.py` from `video_buddy/`. That script is not LTX 2.5 and is not part of `install.py`. On-disk layout notes from before this slim live in [`docs/archive/video_buddy/models/README.md`](archive/video_buddy/models/README.md).

### Pack table

`python -m master_agent workflows --vram` prints the live table. Expected VRAM is a hypothesis, not a guarantee. Official bf16/fp16 dual-UNET is not the default.

| Family | Default slug | When that pack is the one on disk | Notes |
|---|---|---|---|
| LTX 2.3 | `base` / `eros` / `directors` | Quantized GGUF, else EROS/fp8 | ~9.5–13.5GB |
| LTX 2.5 | `ltx25_t2v_i2v` | Q4_K_M file | ~12.5GB. Two-stage is the quality path |
| MiniMax H3 | `h3_t2v` | Q4_K DiT | ~13GB. ≤12s / 0.8MP / 4 steps / CFG 1.0 |
| Wan 2.2 | `wan22` | Q4_K_S or fp8 + Lightx2v | ~13.2GB. Sequential high/low. No dual bf16 |
| AI-VFX / VACE | `vb_aivfx_adv_13` | VACE Q4_K_M file | ~13.6GB. The v1.0 e4m3fn pack is heavy |
| Movie Builder | `vb_movie_builder` | Flux Klein fp8 | ~15.8GB. Heavy. Safer alternate: `ltx25_t2v_i2v` |
| CCC | `vb_ccc_adv` / `vb_ccc41_krea2` | Flux Klein / Krea NVFP4 | ~15.5GB. Heavy. Safer: `flux` / `krea2_img` |
| Flux / Krea | `flux` / `krea2_img` | Flux Q4_K_S file or Krea NVFP4 | ~11GB |
| Qwen Edit | `vb_qwen_edit_360` | Q5_0 + Lightning | ~12GB |
| HeartMuLa | `heartmula` | heartlib 3B | ~12GB, class tight, about 8–16GB by dtype. Run it before or after LTX, not beside it |

Attested Hub ids (fetch only after the ask): Wan QuantStack HighNoise/LowNoise Q4_K_S or Comfy-Org fp8; VACE `mickmumpitz/VACE_Skyreels_V3_R2V_Merge-GGUF`; Krea `Comfy-Org/Krea-2`; Flux `city96/FLUX.1-dev-gguf` or `Comfy-Org/flux1-dev`; Qwen `QuantStack/Qwen-Image-Edit-2509-GGUF`; H3 [`unsloth/MiniMax-H3-GGUF`](https://huggingface.co/unsloth/MiniMax-H3-GGUF) and [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3). QuantStack LTX-2.3 GGUF is not a download pack.

Lightx2v / turbo LoRAs apply when the file is present. LTX TeaCache is inject-when-registered and soft-bypasses when the pack is missing. See [Agents](AGENTS.md).

## HeartMuLa files

`download-models --heartmula` lists attested slots and returns before the LTX scan, so it does not also pull LTX. Default generate plus transcribe is about 25GB. Check free space before `--yes`. A tower check on 2026-09-30 saw about 28GB free on F:.

| Repo | On-disk folder | Size |
|---|---|---|
| `HeartMuLa/HeartMuLaGen` | `tokenizer.json`, `gen_config.json` | small |
| `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` (default) | `HeartMuLa-oss-3B/` | ~15.8GB |
| `HeartMuLa/HeartCodec-oss-20260123` (default) | `HeartCodec-oss/` | ~6.6GB |
| `HeartMuLa/HeartTranscriptor-oss` | `HeartTranscriptor-oss/` | ~3.1GB |

`HeartMuLa/HeartCodec-oss` (no date suffix) returned 401. Do not use it. `HEARTMULA_CODEC_REPO=HeartMuLa/HeartCodec-oss` is refused. Other attested 3B repos, same four shards: `HeartMuLa/HeartMuLa-oss-3B`, `HeartMuLa/HeartMuLa-RL-oss-3B-20260123`. Codec shards for the dated repo: `config.json`, `model.safetensors.index.json`, `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`.

`HEARTMULA_VERSION` stays `3B`. heartlib has no nf4. `HEARTMULA_DTYPE=nf4` is an error. Behavior and CLI: [Features](FEATURES.md#heartmula).

## Model selector

`models manifest` / `models select` is the LTX 2.3 vs 2.5 checklist. It does not run `comfy install` or `comfy update`. `--scan-only` prints the running total, an ETA (default 50 Mbit/s, `--mbps` overrides), and free space. It does not fetch and does not write `state/model_selector.json`. A fetch still needs `--yes` after the disk check. Files already on disk are skipped.

`--soundtrack` is Enable Soundtrack Studio. It calls the existing `--heartmula` consent path. Switching 2.3 and 2.5 asks whether to keep or wipe the previous pack under `MODELS_DIR` (`--keep` / `--wipe` when non-interactive). Sulphur GGUF, the Sulphur LoRA, and the EROS checkpoint are local detect flags. The selector does not download them. LTX 2.3 rows whose size is not attested in this repo are listed and are not fetched.

The studio Models tab calls the same functions in `master_agent/models/selector.py`. Download stays idle until the agreement checkbox is checked.
