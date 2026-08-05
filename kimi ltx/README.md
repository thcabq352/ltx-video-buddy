# ComfyUI Master Agent

Autonomous video production agent with ComfyUI (portable, API on :8188) as the
rendering engine. Current components: ComfyUI bridge (API client, workflow
patcher, workflow validator), model inventory scanner, orchestrator state
machine, and a heuristic + LLM judge loop.

- **LLM:** local-first — `qwen3.6-27b-fable` via Ollama (imported from a local
  GGUF with `ollama create`; see `state/Modelfile.qwen36-fable`) drives the
  director, storyboard and judge. Fallback chain for `LLM_PROVIDER=auto`:
  ollama → kimi (kimi-code CLI OAuth, automatic) → grok (Hermes `xai-oauth` or
  `XAI_API_KEY`). Claude is an optional storyboard panel member
  (`ANTHROPIC_API_KEY`).
- **Models:** LTX 2.3 weights under `models/` (see `models/README.md`), 16GB RTX 5060 Ti profile
- **Workflows:** API-format JSON templates under `workflows/`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Start ComfyUI (leave the window open):

```
ComfyUI_windows_portable\run_api_8188.bat
```

## External dependencies (not pip-installable)

- **ffmpeg** on PATH — frame extraction, concat, audio muxing (imageio-ffmpeg
  or a system build both work).
- **Ollama** — local LLMs + embeddings: `ollama pull nomic-embed-text` (KB),
  the 27B director model via `ollama create` (see
  `state/Modelfile.qwen36-fable`), `qwen3-vl:30b` for the vision judge.
- **ComfyUI portable** (`ComfyUI_windows_portable/`) — self-contained embedded
  python; custom node packs live in `ComfyUI/custom_nodes/` and its pip deps
  are managed with its own `python_embeded\python.exe`, not the project venv.
- **ai-toolkit** (LoRA training only) — separate clone + own venv, set up with
  `python -m master_agent lora setup`. The project venv is untouched.
- **Model weights** (~450GB curated, not in git) — reproduced with
  `python state/download_models.py` (idempotent, skips existing files).

## CLI

```bash
python -m master_agent health              # ComfyUI reachable? GPU stats
python -m master_agent fetch-object-info   # cache node registry to state/object_info.json
python -m master_agent scan-models         # scan models/ → state/model_inventory.json
python -m master_agent validate file.json  # validate one workflow
python -m master_agent validate --all      # validate everything in workflows/
python -m master_agent validate --all --offline  # no server needed (uses cache)

# Orchestrated generation (the Director):
python -m master_agent run "cinematic close-up of rain on a window" --quality draft --duration 3
python -m master_agent run "talking head dub" --video input.mp4 --variant lipsync
python -m master_agent run "..." --no-judge  # skip the judge loop

# Multi-segment (duration beyond the per-clip VRAM cap auto-splits):
python -m master_agent run "premium coffee commercial" --duration 8 --quality draft
python -m master_agent run "..." --duration 8 --dry-run        # storyboard + validate, no GPU
python -m master_agent run "..." --storyboard off              # skip shot cards
python -m master_agent run "..." --llm-panel duo               # 2 local LLMs draft, judge picks
python -m master_agent run "..." --llm-panel ollama:gemma4:latest,kimi   # custom panel
```

The validator checks class types, required inputs, widget values, link type
integrity (match-type passthroughs like `COMFY_MATCHTYPE_V3` count as
wildcards), and that model filenames in loaders exist in the local inventory —
before anything is queued.

## Orchestrator + Judge

`run` drives a state machine: `SELECT_VARIANT → PATCH → VALIDATE → SUBMIT →
POLL → RESOLVE → JUDGE → DONE/ERROR`.

- Nothing is queued before the validator passes (fail fast, zero wasted GPU).
- CUDA OOM walks a downscale ladder (768×512 → 512×320) and retries instead
  of dying.
- The **Judge** (separate from the executor) scores each clip with three
  merged legs: heuristics (file size, duration vs. requested, frame-diff
  motion via ffmpeg), a text-LLM verdict, and a **vision evaluator** —
  `qwen3-vl:30b` reviews extracted frames for temporal consistency, subject
  lock and artifacts (`VISION_ENABLED=0` to disable). Weights:
  `JUDGE_HEURISTIC_WEIGHT` / `JUDGE_LLM_WEIGHT` / `JUDGE_VISION_WEIGHT`,
  renormalized over whichever legs are available. Below threshold the judge
  rewrites the prompt or
  retunes params (`steps`, `cfg`, `stg_scale`, `stg_blocks`, `sampler_name`,
  `seed` — whitelist, clamped) and regenerates, up to `--max-judge-rounds`.
- Every run writes a JSON record to `state/runs/` (params, judge history,
  transitions) — the seed of the later self-learning knowledge base.
- Variant routing is LLM-driven (`orchestrator/director.py`): the local 27B
  picks `base | directors | eros | lipsync | wan22` from the request,
  validated against known variants with keyword rules as fallback
  (`DIRECTOR_LLM=0` for rules-only). Hard constraints always win:
  `--variant` forces, a source video implies lipsync.

## Multi-segment pipeline (storyboard + stitch + full judge)

Requests longer than the per-clip VRAM cap (`SEGMENT_MAX_S`, per quality
profile) go through `orchestrator/pipeline.py`:

1. **Storyboard** — an LLM panel plans one shot card per segment (camera,
   action, continuity language, per-shot LTX prompt); heuristic fallback when
   no LLM is available. Modes: `smart` (default) / `always` / `multi_only` /
   `off`. Panel presets (`--llm-panel`, env `LLM_PANEL`): `default` = the
   local 27B alone, `duo` = 27B + `gemma4:latest` with a judge LLM
   (`--panel-judge`, env `PANEL_JUDGE`, default `ollama`) picking the winning
   board (it may also blend shots). Custom comma lists work too
   (`ollama:<model>`, `kimi`, `grok`, `claude`); unavailable or failing
   members are skipped, never fatal. Panel details (members, latencies,
   winner, judge reason) land in the run record as `panel_meta`.
2. **Per-segment generation** — each shot runs the full orchestrator loop
   (patch → validate → submit → judge) with a per-shot seed offset
   (`base + i*17`).
3. **Stitch** — ffmpeg concat into `outputs/<run_id>/`.
4. **Full-video judge** — the local LLM reviews the stitched video against the
   whole brief + storyboard; weak shots (`shot:N` in its issues) are
   selectively re-generated and re-stitched, up to `--max-full-judge-rounds`
   (default 2).

A pipeline-level record (storyboard, segment scores, full-judge history) is
written to `state/runs/<ts>_<id>_pipeline.json`. Use `--dry-run` to see the
plan and validate every segment's workflow without spending GPU.

## Fractal videos (procedural, CPU-only)

`master_agent/fractal/` renders Mandelbrot/Julia deep-zooms in pure numpy
(smooth coloring, 2x supersampling, float64 depth cap ~1e10) and pipes frames
into ffmpeg — no ComfyUI, no GPU, deterministic:

```bash
python -m master_agent fractal "title sequence" --duration 20 --target seahorse --palette fire
python -m master_agent fractal --julia --palette neon --duration 30
python -m master_agent fractal --audio track.mp3     # beat-reactive + muxed audio
```

Targets: `seahorse | elephant | minibrot | spiral`. Palettes: `fire | ocean |
monochrome | neon | sunset`. With `--audio`, the zoom eases into each beat,
the palette flashes on onsets, and (in `--julia` mode) the `c` parameter
wobbles with the energy envelope.

## Beat-synced music videos

`master_agent/music/` does dependency-free beat detection (ffmpeg decode →
numpy spectral-flux onset envelope → comb-filter tempo, 60–200 BPM →
phase-aligned beat grid → RMS energy sections) and cuts generation to the
grid:

```bash
python -m master_agent music "dreamy synthwave MV, chrome grids" --audio track.mp3
python -m master_agent music "..." --audio track.mp3 --visual fractal   # CPU-only
python -m master_agent run "music video for my song" --audio track.mp3  # auto-routes here
```

`--visual shots` (default): shots are planned on beat boundaries (shorter
shots on high-energy sections), the storyboard LLM gets the beat map as
context, each shot runs the full orchestrator loop, clips are trimmed to
their exact beat windows, concatenated, and the audio is muxed on top
(`kind: "music_video"` run record). `--visual fractal` renders a
beat-reactive fractal for the whole track instead — no GPU. Auto-routing:
`run --audio <file>` with music intent (keywords: "music video", "song",
"beat", "track", "mv") or audio longer than one segment goes to the music
pipeline unless `--variant` is given.

## Upscale post-stage

`--upscale rtx|seedvr2` on `run`/`fractal`/`music` (or
`master_agent.upscale.upscale_video(path, method=...)`) pushes the final
video through ComfyUI:

- `rtx` — NVIDIA RTX Video Super Resolution (fast), via the validated
  Mickmumpitz RTX-SR workflow (nodes 12/14 patched per input).
- `seedvr2` — SeedVR2 diffusion upscaler (slow, production-grade) via
  `workflows/upscale_seedvr2_api.json`, a minimal
  VHS_LoadVideo → SeedVR2 → VHS_VideoCombine chain on the `models/SEEDVR2/`
  weights.

## Persona & intake interview (default-on)

Before anything is generated, the agent's persona interviews you — audience,
tone, look, must-haves, deal-breakers — and synthesizes a creative brief that
becomes the pipeline request. Default persona is **Ara**: the warmth and wit
of Grok's Ara, recalibrated for business (she talks audience, brand and
deliverables, not just vibes).

- **CLI** — interactive runs (`run`, `music`, `fractal`) open the interview
  first; type `just go` / `skip` to bail out early, `--no-interview` to
  bypass entirely (`INTAKE_ENABLED=0` disables globally; non-TTY runs skip
  automatically). `python -m master_agent brief "rough idea"` runs the
  interview only (`--go` chains straight into generation);
  `python -m master_agent persona list|show` manages personas.
- **Web** — the Create and Music tabs open a chat with Ara before the first
  Generate; her refined brief lands in the Brief box (editable) when she's
  done. "skip interview" bypasses.
- **Personas are files** — `master_agent/persona/personas/<slug>.md`
  (bundled: `ara`, `exec`). Drop your own `<name>.md` in `state/personas/`
  (same slug overrides bundled) and select with `PERSONA=<slug>`.
- Every interview writes `state/runs/<ts>_<id>_intake.json` (`kind:
  "intake"`) so the knowledge base learns your preferences over time.
- No LLM available? The intake degrades to a canned opener and folds your
  answers into the request — never fatal.

## Knowledge base (local RAG)

ChromaDB (`state/chroma/`) with Ollama `nomic-embed-text` embeddings — fully
local. Two collections: `workflows` (digests of the templates) and `runs`
(every orchestrator/pipeline run record, auto-ingested after each run).

- Before storyboarding, the pipeline recalls similar past runs (request,
  scores, the judge's own words) and injects them into the prompt — the
  start of the self-learning loop.
- CLI: `python -m master_agent kb ingest` (bulk load), `kb search "query"
  [-k N] [--workflows]`, `kb stats`.
- `KB_ENABLED=0` disables it; `KB_RECALL_K` tunes hit count (default 3).

## Characters & LoRAs (CCC stage)

Mickmumpitz-style consistent characters: a local-LLM **character bible**
(name, rare-token trigger word, fixed appearance block, 12-16 shot prompts)
-> **Flux character sheet** rendered through ComfyUI (`workflows/flux_t2i.json`,
fp8 weights, 1024x1024) -> **vision-curated** for identity consistency
(qwen3-vl scores each shot against the hero image, threshold 0.78) ->
**captioned dataset** (`[trigger], <appearance>` .txt files) under
`state/characters/<name>/` -> **Flux LoRA training** with Ostris ai-toolkit
-> **vision-judged validation** on 4 held-out scenes, with a retry ladder
(+500 steps -> lr 2e-4 -> rank 32 -> +1024 res, max 5 attempts).

```bash
python -m master_agent download-flux            # one-time ~17GB fp8 weights
python -m master_agent lora setup               # one-time ai-toolkit clone + venv (long)
python -m master_agent character create "a grizzled dwarven smith, braided beard"
python -m master_agent character create "..." --train   # sheet + train + validate loop
python -m master_agent character list
python -m master_agent lora train <name> [--steps N --lr X --rank N --validate]
python -m master_agent lora validate <name> [--attempt N]
```

Notes:

- Flux weights come from the non-gated `Comfy-Org/flux1-dev` +
  `comfyanonymous/flux_text_encoders` HF repos into `models/` (shared by
  ComfyUI inference and ai-toolkit training; `ae.safetensors` already ships).
- ai-toolkit lives in `ai-toolkit/` with its **own** venv (`ai-toolkit/venv`)
  — the project `.venv` is untouched. Training is resumable: re-run the same
  `lora train` command after an interrupt.
- 16GB tuning: `quantize` + `low_vram` + `gradient_checkpointing`, rank 16,
  1500 steps, resolutions [512, 768]. Expect hours per LoRA on the 5060 Ti.
  The trainer frees ComfyUI's VRAM before starting.
- Finished LoRAs are copied to `models/loras/<name>_r<rank>.safetensors` so
  any ComfyUI workflow can load them; characters and training attempts are
  logged to the KB (`characters` / `lora_runs` collections).

## Web UI (studio dashboard)

```bash
python -m master_agent ui --port 8189    # http://127.0.0.1:8189
```

FastAPI + single-file dashboard (`master_agent/web/`). Tabs: **Create**
(brief → storyboard preview or full GPU run, variant/panel pickers),
**Fractal** (CPU deep-zoom renders, optional beat-reactive audio + upscale),
**Music** (upload a track → beat-synced music video, AI shots or fractal
visual), **Jobs** (live logs, 2.5s polling, video playback of results),
**Runs** (history + judge scores), **Knowledge** (KB search), **Models**
(inventory). Health strip: ComfyUI/GPU/Ollama/KB at a glance. Jobs run
in-process; one GPU job at a time (`run` + music-shots), fractal/dry-run
jobs may overlap. Audio uploads land in `state/uploads/`. Localhost
single-user — no auth.

## Hermes MCP server

`master_agent/mcp_server.py` exposes the agent to Hermes over stdio MCP
(registered as `master-agent` in `~/.hermes/config.yaml`). Tools:

- `health` — ComfyUI/GPU + Ollama + KB counts
- `create_video` — full pipeline (or `dry_run=True` to plan/validate only)
- `plan_storyboard` — segment split + panel storyboard with KB recall, no GPU
- `judge_asset` — grade a video file (heuristics + LLM + vision)
- `search_workflows` / `search_runs` — KB semantic search
- `kb_ingest` — bulk-load the knowledge base
- `list_models` — local LTX weight inventory
- `validate_workflow` — check a workflow JSON before queuing
- `create_character` — CCC stage: bible -> Flux sheet -> captioned dataset
  (`train=True` chains into LoRA training)
- `train_lora` — Flux LoRA training for an existing character (+ validation)

Pipeline chatter is redirected to stderr inside tools so the stdio protocol
channel stays clean. Heavy native deps are pre-imported at server startup
(main thread) — without that warmup, first-use imports inside anyio worker
threads stall ~30s per DLL on Windows.

Latency note for MCP clients: `health`/`search_*`/`validate_workflow`/
`kb_ingest` answer in ~2s. `judge_asset` and `create_video` cold-load large
Ollama models (19-27GB) and can take several minutes — allow long timeouts.

## Status (2026-08-04, third pass)

- **Persona & intake interview** (`master_agent/persona/`): the agent now
  interviews the user before generating anything and turns the conversation
  into a structured creative brief. Default persona **Ara** (Grok-Ara
  warmth, business-calibrated) + strict `exec`; personas are markdown files
  with user overrides in `state/personas/` (`PERSONA` env). Default-on in
  CLI (`--no-interview` to skip) and web (chat panels on Create/Music,
  `POST /api/intake`, `/api/persona`); new `brief` and `persona` CLI
  subcommands; interviews recorded as `kind: "intake"` run records.
  104 pytest green.

## Status (2026-08-04, second pass)

- **Fractal pipeline** (`master_agent/fractal/`): pure-numpy Mandelbrot/Julia
  deep-zoom renderer (smooth coloring, 2x supersampling, active-subset
  iteration, float64 depth cap) → ffmpeg H.264 pipe. Beat-reactive mode:
  zoom eases into beats, palette flashes on onsets, Julia `c` wobbles with
  energy. `python -m master_agent fractal`, `kind: "fractal"` run records.
- **Beat-synced music videos** (`master_agent/music/`): dependency-free beat
  detection (numpy spectral flux + comb-filter tempo 60–200 BPM + phase-
  aligned grid + RMS energy sections). `run_music_video()` plans shots on
  beat boundaries (shorter on drops/choruses), storyboards with beat context,
  reuses the orchestrator per shot, trims clips to exact beat windows
  (`cut_to_windows`), muxes the track (`mux_audio` in `video_concat.py`).
  `--visual fractal` skips ComfyUI entirely. `run --audio` auto-routes music
  intent (keywords or audio longer than one segment) here.
- **Upscale post-stage** (`master_agent/upscale.py`, `--upscale`): `rtx` via
  the validated RTX-SR workflow (verified live: 320x240 → 1920x1080),
  `seedvr2` via new `workflows/upscale_seedvr2_api.json` (minimal chain on
  the SEEDVR2 weights, validates PASS live).
- **Web UI**: Fractal + Music tabs with `POST /api/fractal`, `/api/music`,
  `/api/upload` (state/uploads/); GPU gate covers music-shots jobs, fractal
  jobs overlap. 91 pytest green.

## Status (2026-08-05)

- **LTX 2.3 Movie Builder integrated (batch 4).**
  `260507_VIDEO-BUDDY_MOVIE-BUILDER_1-1_ADV_api.json` (189 nodes) converted and
  **validates PASS live**; manifest entry `vb_movie_builder` (29 entries).
  Shot-by-shot movie pipeline: Flux 2 Klein start-frames, LTX 2.3 video+audio
  with voice cloning (5s reference), optional 2x spatial upscale, 360°
  environment generation + shot-reverse-shot crops, ShotAssembler final cut.
  Full usage guide: `workflows/260507_VIDEO-BUDDY_MOVIE-BUILDER_GUIDE.md`
  (KB-ingested — guides in `workflows/*.md` are now indexed alongside the
  JSON digests).
- **Reusable converter:** `state/convert_ui_to_api.py` converts any UI graph
  export to API format (reroute/GetNode-SetNode unwrapping, subgraph inline
  expansion, schema-accurate widget mapping); regression-checked against all
  9 existing UI↔API pairs. Batch-4 repairs in `state/repair_batch4_api.py`
  (3 private author assets repointed to placeholders).
- **New packs:** ComfyUI-Olm-DragCrop, ComfyUI_preview360panorama
  (ProGamerGov), comfyui-LatLong, fresh comfyui-mickmumpitz-nodes clone, and
  local `movie_builder_shims` (no-op stand-ins for the unreleased
  MickmumpitzShotOrder/ShotDuplicator organizational nodes).
- **~34GB new weights** via `state/download_models.py` BATCH4 section:
  flux-2-klein 360-erp outpaint LoRA, ltx-2.3-id-lora-talkvid-3k (voice
  cloning), gemma_3_12B_it_fp8_scaled, LTX23 video VAE bf16, x2 spatial
  upscaler (new `latent_upscale_models` folder mapped in
  `extra_model_paths.yaml`), and the LTX-2.3-dev-Q4_K_S GGUF low-VRAM option.
- **Code on GitHub:** initial push to the private repo
  `thcabq352/ltx-video-buddy` (weights, ComfyUI portable, Blender, example
  media, secrets excluded — see `.gitignore` at the repo root).

## Status (2026-08-04)

- **Mickmumpitz batches 2+3 integrated.** 15 more UI workflows converted to
  API format and **validate clean live + offline** (28 manifest entries total):
  Qwen-Image-Edit-360, AI-VFX startimage/preprocess/1.0/1.3, LTX-2.3
  3D-RENDERING LIP-SYNC v08, CCC 4.1 Krea2-Edit BETA, the three AI-RENDERING
  example projects (030/050/BusinessWoman), Z-Image-Turbo-CN, AI-RENDERER
  SMPL/ADV 2.1 + ADV 2.0, and NVIDIA-RTX-SUPER-RESOLUTION.
- **New packs:** gguf/ComfyUI-GGUF, ComfyUI_essentials, comfyui_controlnet_aux,
  DepthCrafter, cotracker, ComfyUI-RMBG (SAM3 — downloads `sam3.pt` on first
  run), ComfyUI-WanVaceAdvanced, ComfyUI-Krea2-Ostris-Edit, comfyui-krea2edit,
  ComfyUI-GIMM-VFI, mickmumpitz's ComfyUI-WanVideoWrapper fork,
  ComfyUI-DepthAnythingV3, Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.
  RES4LYF + comfyui-mickmumpitz-nodes updated (1.4.0).
  `triton-windows` + `nvidia-vfx` (from `pypi.nvidia.com`, per Mickmumpitz's
  RTX guide) installed into the portable python.
- **~100GB new weights** via `state/download_models.py` (see `models/README.md`:
  Qwen-Edit GGUFs, VACE e4m3fn + Q4_K_M, LTX ic-loras/OmniNFT, Krea2 fp8 +
  identity-edit LoRA, Z-Image + Fun-Controlnet). The script now moves HF cache
  blobs instead of copying (cache symlinks were doubling disk usage).
  `extra_model_paths.yaml` gained `model_patches` for the Z-Image controlnet.
- **Blender 5.3 alpha** lives in the project root
  (`blender-5.3.0-alpha+main.*/`) and `Example_Blender_Files.zip` is unzipped
  under `workflows/AI-RENDERING-EXAMPLE FILES/` — clay/depth/line passes for
  the AI-RENDERER + LIP-SYNC pipelines; bundled assets were copied into
  ComfyUI `input/`.
- **Scene/style LoRA training** scaffold in `training/` (ai-toolkit configs
  for LTX-2.3 and Wan 2.2 tuned to 16GB, dataset convention + docs) — see
  `training/README.md`.
- Converter repairs for this batch live in `state/repair_batch2_api.py`
  (dead subgraph-output links, SetNode rewires, dropped VAEEncode restoration,
  widget-shift fixes, private-asset repoints). 68 pytest green.

## Status (2026-07-31)

- **Mickmumpitz library integrated.** All 7 UI workflows (`2606*`/`2607*`) were
  converted to API format (`*_api.json`, via the locally installed
  [workflow-to-api converter endpoint](https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint))
  and **validate clean live + offline**. Originals are kept for frontend
  editing + KB RAG; re-convert after editing. Newly installed packs:
  VideoHelperSuite, RES4LYF (Film Grain), Comfyroll, AutoCropFaces,
  ComfyUI-QwenVL (dataset tagger downloads Qwen3-VL weights on first run).
  Fixes along the way: `ltx_windows_fix` infinite flush recursion (latent —
  fired on any `logging` flush), LTXVideo kornia 0.8 `pad` import,
  impact-pack/seedvr2/was-ns missing deps (skimage, rotary-embedding-torch,
  numba…), converter widget-misalignment repairs (`state/repair_vb_api.py`).
- **New pipeline variant `wan22`** (Wan 2.2 two-stage T2V, 16fps, 4n+1 frames):
  full routing (director keywords + LLM prompt, CLI, web UI), dual-UNET
  weights via `MODEL_FILES["wan22"]`, per-variant frame math
  (`VARIANT_GEN` in config.py). Private LoRAs in the original graph
  (CCC4-0 character, Krea Multiple_realistic) are disabled/bypassed — they
  were never published; the four public Wan LoRAs stay active.
- `krea2_img` is a manifest/patcher template variant (like `flux`) — usable
  via `load_and_patch_workflow`, not video-pipeline routed (image graph).
  `vb_ideogram` validates but needs an Ideogram API key to actually run.
- Validator now understands prefix-style autogrow inputs
  (`COMFY_AUTOGROW_V3`, e.g. BatchImagesNode `images.image0`).

## Status (2026-07-30)

- All four templates (`base_t2v_i2v`, `directors`, `eros_t2v_i2v`,
  `lipsync_ia2v`) **validate clean** live and offline, and all four
  patcher outputs (`load_and_patch_workflow`) validate clean end-to-end.
- `lipsync_ia2v.json` was repaired against this install: stale UI-export
  widget keys replaced with real input names, `GetImageSizeAndCount` → core
  `GetImageSize`, checkpoint refs → `ltx-2.3-22b-dev-fp8.safetensors`
  (downloaded; official `ltx-2.3-22b-dev.safetensors` remains the optional
  upgrade). Remaining warnings: `ResizeImageMaskNode.width/height` are V3
  dynamic widgets not statically declared in `/object_info` (expected).
  `LoadVideo.file` defaults to `warehouse_src_30fps.mp4` (present in ComfyUI
  `input/`); the patcher swaps it via `video_name=` at runtime.
- `extra_model_paths.yaml` in the portable ComfyUI now points at this repo's
  `models/` (it previously pointed at the sibling `LTX Project/models`, which
  no longer exists — the server saw zero checkpoints before the fix).

Partial downloads under `models/` (`*.part`) are flagged by `scan-models`.
