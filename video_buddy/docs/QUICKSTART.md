# Operator quickstart — catalog, doctor, download-models

Post-merge `main` (LTX 2.5 PR #6 + capability audit PR #5). From the
`video_buddy/` directory, with the project venv on PATH.

```bash
cd video_buddy
python install.py                         # once: venv, pip, Playwright, .env, ffmpeg, Ollama
python -m master_agent doctor             # scan only — never fetches weights
python -m master_agent workflows          # default catalog (includes ltx25_*)
python -m master_agent download-models --ltx25   # list confirmed-missing; add --yes only then
python -m master_agent download-models --h3      # MiniMax H3 confirmed-missing; add --yes only then
```

## 1. Default catalog (no experimental flags)

`python -m master_agent workflows` prints every default variant. The seven
LTX 2.5 graphs and the four MiniMax H3 graphs are always listed, even if
tower nodes or weights are missing.

| Buddy id | Research alias | What it does |
|---|---|---|
| `ltx25_t2v_i2v` | `t2v_i2v` | Single-stage distilled T2V / I2V (default 2.5 path) |
| `ltx25_t2v_i2v_two_stage` | `t2v_i2v_two_stage` | Two-stage (latent spatial upscale) |
| `ltx25_flf2v` | `flf2v` | First + last frame → video |
| `ltx25_msr` | `msr` | Multi-reference (pic1–pic4 + background) |
| `ltx25_v2v_ic_lora` | `v2v_ic_lora` | Video-to-video IC-LoRA |
| `ltx25_a2v` | `a2v` | Audio-to-video (needs `--audio` / LoadAudio) |
| `ltx25_t2a` | `t2a` | Text-to-audio only |
| `h3_t2v` | `fl2va` | MiniMax H3 fl2va text-to-AV (native stereo) |
| `h3_i2v` | — | MiniMax H3 fl2va image-to-AV |
| `h3_flf` | — | MiniMax H3 fl2va first + last frame |
| `h3_r2v` | `ref2va` | MiniMax H3 ref2va reference-to-AV |

Existing director paths stay on the same CLI: `base`, `eros`, `directors`,
`lipsync`, `wan22`. Manifest slugs (`vb_movie_builder`, `vb_aivfx_adv`,
`flux`, …) resolve for `comfy run --template` without expanding the director.

### Invoke

```bash
# list (CLI + GET /api/variants + Create-tab + Comfy-tab)
python -m master_agent workflows
python -m master_agent workflows --json

# generate → patch (prompt, seed, 8n+1 frames) → lint → queue
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain"
python -m master_agent comfy run --mode generate --variant ltx25_flf2v --prompt "first to last"
python -m master_agent run "LTX 2.5 alley push-in" --variant ltx25_t2v_i2v --no-interview
python -m master_agent run "interpolate these frames" --variant flf2v --image start.png --no-interview

# research aliases resolve without an env flag
python -m master_agent comfy run --mode generate --variant t2v_i2v --prompt "x" --prepare

# MiniMax H3 (CFG 1.0, 4 steps, 0.6–0.8 MP / ≤12s on 16GB)
python -m master_agent comfy run --mode generate --variant h3_t2v --prompt "neon rain, stereo city bed"
python -m master_agent comfy run --mode generate --variant fl2va --prompt "x" --prepare
python -m master_agent run "lock this face" --variant h3_r2v --image ref.png --no-interview

# still-working LTX 2.3 / Wan
python -m master_agent comfy run --mode generate --variant base --prompt "a test shot"
python -m master_agent comfy run --mode generate --variant wan22 --prompt "photoreal street"
python -m master_agent run "talking head dub" --video input.mp4 --variant lipsync --no-interview
```

`--prepare` lints and writes JSON. It does **not** require weights or a GPU
queue.

## 2. Inventory first (scan local, then ask)

Do **not** assume a download. Buddy scans, in order of search roots:

- `MODELS_DIR` (default `video_buddy/models/`)
- `COMFYUI_ROOT/models`
- `EXTRA_MODELS_DIRS` / `LTX_MODELS_DIRS` (pathsep or comma — extra volumes)
- Comfy `extra_model_paths.yaml` `base_path` / folder entries
- `PROJECT_ROOT/models`, portable Comfy `models/`, `./models`
- Hugging Face hub snapshots (`HF_HOME` / `HUGGINGFACE_HUB_CACHE` /
  `~/.cache/huggingface/hub/models--Lightricks--LTX-2.5/`)

Zero-byte placeholders count as **missing**. Official bf16 Gemma is **not**
required if `gemma4-12b-heretic-ltx25-int8convrot.safetensors` or the Comfy
int8 TE is present.

1. Every mandatory slot filled → proceed silently; wire loaders to those files.
2. Confirmed missing → print an ask (filename, dest folder, size, gated-HF note).
3. Never auto-download. `setup --fix` does **not** fetch models.

## 3. `python -m master_agent doctor`

`doctor` is an alias of `setup`. It **reports**. It does **not** fetch LTX
weights.

| Check | What it means |
|---|---|
| `python` | 3.10+ |
| `venv` | project `.venv` exists |
| `pip` | `requirements.txt` packages installed |
| `playwright` | Python package present (Chromium install is `--fix`) |
| `env` | `.env` exists (copied from `.env.example`) |
| `ffmpeg` | on PATH |
| `ollama` | on PATH + `qwen3-vl-heretic` and `nomic-embed-text` pulled |
| `comfyui` | `COMFYUI_URL` (`http://127.0.0.1:8188`) answers `/system_stats` |
| `ltx25-weights` | scan of the `ltx25_core` bundle + **loader pick** |
| `h3-weights` | scan of the `h3_fl2va` bundle + **loader pick** |

The `ltx25-weights` line prints the 16GB-class pick, for example
`GGUF Q4 (ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf) — 16GB-class preference #1`.
That is the file the default loader will wire. Doctor never copies or
downloads it.

```bash
python -m master_agent doctor              # report only
python -m master_agent setup               # same command
python -m master_agent setup --fix         # venv / pip / Playwright / .env / ffmpeg / Ollama — still no weights
python -m master_agent doctor --fix-models # explicit consent: fetch confirmed-missing LTX 2.5 only
```

`--fix-models` prints the same ask as `download-models --ltx25`, then fetches
the missing mandatory set. Use it only after you have reviewed that list.

## 4. `python -m master_agent download-models --ltx25`

Lists or fetches **only confirmed-missing** files after the scan.

```bash
python -m master_agent download-models --ltx25
# → prints the ask (or "OK    all required weights present")
# → exit 2 if something is missing (nothing downloaded)

python -m master_agent download-models --ltx25 --yes
# → Hugging Face fetch of that missing set only (gated Lightricks/LTX-2.5)

python -m master_agent download-models --ltx25 --bundle ltx25_iclora --yes
python -m master_agent download-models --ltx25 --optional --yes   # also distilled LoRA 450 + temporal upscaler
python -m master_agent download-models --ltx25 --json
```

`--ltx25` is the default bundle family (`ltx25_all` when `--ltx25` is set).
`--h3` selects `h3_all` (fl2va + ref2va GGUF, Comfy TE, both VAEs). Bundles:
`ltx25_core` | `ltx25_two_stage` | `ltx25_iclora` | `ltx25_msr` | `ltx25_all`
| `h3_fl2va` | `h3_ref2va` | `h3_all`. IC-LoRA / MSR also need the Ingredients
(or pixel-spatial) LoRA.

Requires `HF_TOKEN` / `huggingface-cli login` with access to the gated repos.
See [`../REQUIRED-FILES.md`](../REQUIRED-FILES.md).

## 5. 16GB-class loader preference

`VRAM_GB` defaults to `16`. When several transformers are on disk:

1. **GGUF Q4** (`ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf`) → `UnetLoaderGGUF`
2. Else **NVFP4** if `VRAM_GB` ≥ 14
3. Else **int8-convrot**, then official bf16

A machine that already has GGUF Q4 + NVFP4 loads **GGUF Q4**. Research JSON
still says `ckpt_name: ltx-2.5-22b-distilled.safetensors` on
`CheckpointLoaderSimple`; Buddy remaps that stub and rewrites the loader.

## 6. Capabilities (what Buddy actually drives)

```bash
python -m master_agent capabilities --offline
python -m master_agent capabilities --json
python -m master_agent fetch-object-info    # refresh state/object_info.json from live Comfy
python -m master_agent capabilities         # prefer live /object_info
```

`--offline` uses the committed cache. The matrix is **read-only** — a live
class does not become a queued graph. Expected post-merge shape:

| Capability | In Buddy? | How to invoke |
|---|---|---|
| LTX 2.5 distilled family | **yes** | `--variant ltx25_*` / research aliases |
| MiniMax H3 fl2va / ref2va | **yes** | `--variant h3_t2v` / `h3_i2v` / `h3_flf` / `h3_r2v` |
| LTX 2.3 T2V/I2V | **yes** | `--variant base` / `eros` / `directors` |
| LTX lipsync | **yes** | `--video` or `--variant lipsync` |
| Wan 2.2 T2V (native UNET) | **yes** | `--variant wan22` |
| K3NK WAN 2.2 AIO I2V | **no** | Not in `MODEL_FILES` / no graph |
| WanVideoWrapper / Fun Inpaint / Fun Control | **no** (graphs) | Catalog / fail-closed |
| TeaCache / `WanVideoTeaCache` | **bypass** | Missing node → WARNING + rewire; **not** inject |
| Movie Builder / AI-VFX / CCC | **partial** | `comfy run --template vb_movie_builder` (not director-routed) |
| CPU fractal | **yes** | `python -m master_agent fractal` (not Comfy Voronoi/Perlin) |
| SeedVR2 | **yes (post)** | `--upscale seedvr2` |

Full matrix and wiring plan: [`../AUDIT.md`](../AUDIT.md).

## 7. What was **not** ported from ltx2.5-research-agent

Buddy stays the Comfy / video driver. Left in
https://github.com/thcabq352/ltx2.5-research-agent :

- LangGraph research / scrape / A2A / Gradio harness
- Secrets and that repo's `.env`
- The `ltx_research_agent` package name
- Imagine client / OAuth

WAN / K3NK / TeaCache behavior in this repo is **unchanged** (TeaCache still
soft-bypass). Historical copy table: [`../../MERGE-LTX25.md`](../../MERGE-LTX25.md).
