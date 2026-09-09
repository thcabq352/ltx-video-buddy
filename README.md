# Video Buddy

**An autonomous AI video production studio that runs entirely on your own hardware.**

Video Buddy turns a plain-language request — *"a premium cold-brew commercial,"*
*"a music video for this track,"* *"a five-shot short film with my lead
actor's face and voice"* — into a finished, judged, upscaled video. It
interviews you before it generates anything, plans the production like a
director, renders on a curated library of state-of-the-art open video models,
grades its own work shot by shot, and learns from every run.

No cloud render farm. No per-minute pricing. No footage leaving your machine.

---

## Why it exists

Generative video models are extraordinarily capable and extraordinarily hard to
operate. Getting a production-grade result out of ComfyUI today means knowing
which of hundreds of node graphs to use, which weights each one needs, how to
keep a character's face consistent across shots, how to clone a voice, how to
fix a bad render — and doing it all by hand, per shot.

Video Buddy is the missing layer: **an agent that operates the studio for
you.** You bring the intent; it brings the craft.

## What it does

- **Creative intake, like a real agency.** Before anything renders, the
  agent's persona (default: **Ara**) interviews you — audience, tone, look,
  must-haves, deal-breakers — and distills a structured creative brief.
- **Directs, not just generates.** A local LLM director routes your brief to
  the right pipeline, a storyboard panel plans shot-by-shot cards (camera,
  action, continuity), and a state machine executes: patch → validate →
  submit → judge → retry.
- **Fail fast, zero wasted GPU.** Every workflow is validated against the live
  node registry and local model inventory *before* anything is queued. CUDA
  OOM triggers an automatic downscale-and-retry ladder instead of a crash.
- **It grades its own work.** A three-legged judge — heuristics, a text LLM,
  and a vision model reviewing actual frames — scores every clip for temporal
  consistency, subject lock, and artifacts. Weak shots get rewritten and
  re-rendered, automatically.
- **Movie Builder.** Shot-by-shot film production on LTX 2.3: Flux 2 Klein
  start-frames, per-shot video + audio, **voice cloning from a 5-second
  sample**, 360° environment generation for matched shot-reverse-shot, color
  matching across cuts, and a final assembler that stitches the movie.
- **Beat-synced music videos.** Dependency-free beat detection (spectral flux
  + comb-filter tempo) builds a phase-aligned beat grid; shots are planned,
  cut, and trimmed to exact beat windows, with the track muxed on top.
- **A character factory.** Describe a character once — the agent writes a
  character bible, renders a reference sheet, vision-curates a dataset, trains
  a Flux LoRA on a single 16GB GPU, and validates it on held-out scenes.
- **It remembers.** A fully local knowledge base (ChromaDB + Ollama
  embeddings) ingests every workflow, guide, and run record. Past successes
  and judge feedback shape future prompts — a self-learning loop.
- **Production finishing.** One flag upscales any result via NVIDIA RTX Video
  Super Resolution (fast) or SeedVR2 diffusion upscaling (production-grade).

## Interfaces

| Interface | Use |
|---|---|
| **CLI** | `python -m master_agent run "..."` — full pipeline, flags for everything |
| **Web studio** | `python -m master_agent ui` — VIDEO BUDDY dashboard (Create / Comfy / Voice / Fractal / Music / Jobs) |
| **MCP server** | Exposes the agent as tools (`create_video`, `plan_storyboard`, `judge_asset`, …) to MCP clients |

## Architecture

```
you ──▶ persona intake ──▶ creative brief
                             │
                    director LLM ──▶ variant routing
                             │
                    storyboard panel (multi-LLM, judged)
                             │
              ┌──────── orchestrator state machine ────────┐
              │ patch → validate → submit → poll → judge   │  ◀── KB recall
              └──────────────┬─────────────────────────────┘
                             ▼
                     ComfyUI (:8188) — 29 curated workflows
                     LTX 2.3 · Flux 2 Klein · Wan 2.2 · Qwen · Z-Image
                             │
                     stitch → full-video judge → upscale → delivery
```

**Local-first by design:** the director, storyboard, judge, and embeddings run
on Ollama (`qwen3-vl-heretic`, 9B-class VL). Cloud LLMs (Grok, Claude) are optional fallbacks
and panel members, never a requirement.

## Quickstart

Need **Python 3.10+**. From `video_buddy/` run the installer — it creates `.venv`, installs pip deps, Playwright Chromium, copies `.env`, and tries to install ffmpeg + Ollama models.

**Windows:** `install.bat`  
**macOS / Linux:** `./install.sh`  
**Any OS:** `python install.py`

```bash
cd video_buddy
python install.py
python -m master_agent setup          # re-check anytime
python -m master_agent ui --port 8189
```

Start ComfyUI on `:8188` (Windows portable: `ComfyUI_windows_portable\run_api_8188.bat`). Then:

```bash
python -m master_agent run "cinematic close-up of rain on a window"
python -m master_agent music "dreamy synthwave MV" --audio track.mp3
```

Model weights (~450GB curated) are reproduced on any machine with
`python state/download_models.py`. Full operator docs live in
[`video_buddy/README.md`](video_buddy/README.md).

## Repository layout

```
video_buddy/
├── master_agent/      # the agent: orchestrator, director, judge, persona,
│                      # KB, characters/LoRA, music, fractal, upscale, web, MCP
├── workflows/         # curated ComfyUI workflow library + manifests + guides
├── training/          # LoRA training configs (16GB-tuned)
├── tests/             # 104 pytest cases
├── state/             # tooling: model downloader, UI→API converter, repairs
└── docs/              # white paper & investor materials
```

## Documentation

- [Operator manual](video_buddy/README.md) — full CLI/web/MCP reference and changelog
- [Movie Builder guide](video_buddy/workflows/260507_VIDEO-BUDDY_MOVIE-BUILDER_GUIDE.md) — shot-by-shot film production
- White paper & investor materials — `docs/` (not published in this repo; available on request)

## Status

Actively developed. 16 API workflows validate clean against a live server;
104 tests green. Latest milestone: **LTX 2.3 Movie Builder** — shot-by-shot
film production with voice cloning and 360° environments — integrated,
live-validated, and documented. See the
[technical README](video_buddy/README.md#status-2026-08-05) for the full
changelog.

## Legal & licensing

Video Buddy is an orchestration layer. The models it drives — LTX 2.3
(Lightricks), Flux (Black Forest Labs), Wan, Qwen, and others — are
third-party works under their own licenses, several of which restrict
commercial use. Selected workflow designs credit
[Mickmumpitz](https://mickmumpitz.ai). **Review each model's license before
commercial deployment.** This repository contains no model weights.

---

*Built for creators and businesses who want a production studio, not a
science project.*
