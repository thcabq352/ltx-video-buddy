# Video Buddy

**An autonomous AI video production studio that runs entirely on your own hardware.**

Video Buddy turns a plain-language request — *"a premium cold-brew commercial,"*
*"a music video for this track,"* *"a five-shot short film with my lead
actor's face and voice"* — into a finished, judged, upscaled video. It
interviews you before it generates anything, plans the production like a
director, renders on a curated library of open video models, grades its own
work shot by shot, and learns from every run.

No cloud render farm. No per-minute pricing. No footage leaving your machine.

## Demo — MiniMax H3 / photoreal

A public still from an 8-second MiniMax H3 clip Video Buddy queued through
Comfy (`h3_t2v` / fl2va). Files live in this repo so github.com viewers do
not need Tailscale or a private share.

<p align="center">
  <a href="docs/demo/H3-SHOWCASE-BMX-8s-720p.mp4">
    <img src="docs/demo/H3-SHOWCASE-BMX-8s-hero.png" alt="MiniMax H3 via Video Buddy — golden-hour BMX berm, backflip, dusty stick (~8s)" width="100%" />
  </a>
</p>

<p align="center"><em>Golden-hour BMX berm → backflip → dusty stick. ~8s. MiniMax H3 via Video Buddy / Comfy.</em></p>

Play in the GitHub blob viewer:
[`docs/demo/H3-SHOWCASE-BMX-8s-720p.mp4`](docs/demo/H3-SHOWCASE-BMX-8s-720p.mp4)
(8.0 s, 1280×720, 4 197 204 bytes). Poster + provenance live at
[`docs/demo/`](docs/demo/). [`docs/showcase/h3-bmx/`](docs/showcase/h3-bmx/)
is a pointer to that folder (no second copy). The 1080p encode is omitted
from git (~10 MB).

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
- **LTX 2.5 in the default catalog.** Seven distilled API graphs
  (`ltx25_t2v_i2v`, `ltx25_t2v_i2v_two_stage`, `ltx25_flf2v`, `ltx25_msr`,
  `ltx25_v2v_ic_lora`, `ltx25_a2v`, `ltx25_t2a`) are first-class variants —
  no experimental flags. **MiniMax H3** (`h3_t2v` / `h3_i2v` / `h3_flf` /
  `h3_r2v`, aliases `fl2va` / `ref2va`) is the same: GGUF-first omni
  video+stereo audio, CFG 1.0. LTX 2.3 (`base` / `eros` / `directors` /
  `lipsync`), Wan 2.2 T2V, music videos, and LTX TeaCache
  inject-when-registered (soft-bypass if the pack is missing).
- **Scan-local-first weights.** Buddy inventories `MODELS_DIR`, Comfy
  `models/`, extra volumes, `extra_model_paths.yaml`, and the Hugging Face
  hub cache. It asks before downloading. Official bf16 Gemma is not required
  if a heretic / int8 text encoder is already on disk.
- **Fail fast, zero wasted GPU.** Every workflow is validated against the live
  node registry and local model inventory *before* anything is queued. CUDA
  OOM triggers an automatic downscale-and-retry ladder instead of a crash.
- **It grades its own work.** A three-legged judge — heuristics, a text LLM,
  and a vision model reviewing actual frames — scores every clip for temporal
  consistency, subject lock, and artifacts. Weak shots get rewritten and
  re-rendered, automatically.
- **Movie Builder.** Shot-by-shot film production on **LTX 2.3**: Flux 2 Klein
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
| **CLI (first)** | Drive Comfy with `python -m master_agent comfy run`. Full director pipeline is `run "..."`. |
| **Doctor** | `python -m master_agent doctor` — deps + LTX 2.5 / H3 inventory. **Does not fetch weights.** |
| **Download** | `python -m master_agent download-models --ltx25` or `--h3` lists confirmed-missing slots; `--yes` only after you agree. |
| **Capabilities** | `python -m master_agent capabilities --offline` — Comfy pack vs Buddy wiring matrix. |
| **Web studio** | `python -m master_agent ui` — optional human dashboard (Create / Comfy / Voice / Fractal / Music / Jobs) |
| **MCP server** | Hermes tools (`create_video`, `plan_storyboard`, `judge_asset`, …) after the CLI path works |

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
                     ComfyUI (:8188) — default catalog
                     LTX 2.5 · MiniMax H3 · LTX 2.3 · Flux 2 Klein · Wan 2.2 · Qwen · Z-Image
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
python -m master_agent doctor         # deps + LTX 2.5 / H3 scan; no download
python -m master_agent workflows      # default catalog (includes ltx25_* and h3_*)
python -m master_agent ui --port 8189
```

Start ComfyUI on `:8188` (Windows portable: `ComfyUI_windows_portable\run_api_8188.bat`). Then:

```bash
python -m master_agent health
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain"
python -m master_agent comfy attach --recipe previs.json --json workflow.json   # dry-run attach
python -m master_agent run "cinematic close-up of rain on a window"
python -m master_agent music "dreamy synthwave MV" --audio track.mp3
python -m master_agent mv render "I'm in love with a bot" --audio track.mp3 --out out/MV-FIXED.mp4
python -m master_agent mv render --audio track.mp3 --dry-run   # plan + unique + Remotion wiring, no GPU
```

**Do not assume a weight download is needed.** `doctor` reports the 16GB-class
loader pick (GGUF Q4 → NVFP4 if `VRAM_GB` ≥ 14 → int8-convrot → bf16). If a
slot is confirmed missing (zero-byte files count as missing), review the ask
then:

```bash
python -m master_agent download-models --ltx25        # list only
python -m master_agent download-models --ltx25 --yes  # fetch that missing set
python -m master_agent download-models --h3           # MiniMax H3 list only
python -m master_agent download-models --h3 --yes
```

Legacy Mickmumpitz / LTX 2.3 packs still use `python state/download_models.py`
when you need those older filenames. Full operator docs:
[`video_buddy/README.md`](video_buddy/README.md) ·
[`video_buddy/docs/QUICKSTART.md`](video_buddy/docs/QUICKSTART.md) ·
[`video_buddy/REQUIRED-FILES.md`](video_buddy/REQUIRED-FILES.md).

## Repository layout

```
docs/demo/             # H3 BMX poster + 720p + provenance
docs/showcase/h3-bmx/  # pointer README to docs/demo/
video_buddy/
├── master_agent/      # the agent: orchestrator, director, judge, persona,
│                      # KB, characters/LoRA, music, fractal, upscale, web, MCP
├── workflows/         # curated ComfyUI workflow library + manifests + guides
│   ├── ltx-2.5/       # seven default-catalog LTX 2.5 API graphs
│   └── minimax-h3/    # four default-catalog MiniMax H3 API graphs (fl2va / ref2va)
├── training/          # LoRA training configs (16GB-tuned, LTX 2.3 / Wan 2.2)
├── tests/             # pytest (catalog, weights, capabilities, Rainey stop-lines)
├── state/             # tooling: legacy model downloader, UI→API converter
└── docs/              # operator quickstart (CLI / doctor / download-models)
```

## Documentation

- [Operator manual](video_buddy/README.md) — full CLI/web/MCP reference and changelog
- [Doctor / download-models / catalog](video_buddy/docs/QUICKSTART.md) — inventory-first LTX 2.5 flow
- [Required files](video_buddy/REQUIRED-FILES.md) — accepted local names + Hub catalog
- [MiniMax H3 demo](docs/demo/) — photoreal BMX still + run sidecar
- [MiniMax H3 showcase](docs/showcase/h3-bmx/) — pointer to `docs/demo/`
- [LTX 2.5 workflows](video_buddy/workflows/ltx-2.5/README.md) — ids and invoke examples
- [MiniMax H3 workflows](video_buddy/workflows/minimax-h3/README.md) — fl2va / ref2va ids
- [Self-improvement loop](video_buddy/docs/SELF_IMPROVEMENT_LOOP.md) — live judge → revise → re-run map
- [ClipProvenance](video_buddy/docs/CLIP_PROVENANCE.md) — sidecar + run-row prompt/seed/judge contract
- [Music-video mode](video_buddy/docs/MUSIC_VIDEO.md) — Comfy/LTX burns → Remotion MTV stitch (not Grok)
- [Capability audit](video_buddy/AUDIT.md) — what Buddy drives vs the live tower
- [Merge notes (historical)](MERGE-LTX25.md) — PR #6, already merged
- [Movie Builder guide](video_buddy/workflows/260507_VIDEO-BUDDY_MOVIE-BUILDER_GUIDE.md) — LTX 2.3 shot-by-shot film
- [Agent notes](video_buddy/AGENTS.md) — Rainey fleet stop-lines
- [Hermes skill](video_buddy/skills/video-buddy/SKILL.md) — install into `~/.hermes/skills/video-buddy/`
- White paper & investor materials — not published in this repo; available on request

## Hermes install

Hermes agents forget Video Buddy if they only see MCP tools. **MCP ≠ skills.**
The installer copies the skill **and** seats Hermes profile `ltx` (primary
discovery). A2A on studio `:8189` stays as the fallback. Buddy never binds 8642.

```bash
cd video_buddy
python install_hermes_skill.py
# copies video_buddy/skills/video-buddy/ → ~/.hermes/skills/video-buddy/
# seats ~/.hermes/profiles/ltx/  (SOUL + MCP, no .env)
python -m master_agent hermes status
```

Or copy that folder by hand. Confirm `SKILL.md` is at `~/.hermes/skills/video-buddy/SKILL.md`.
Confirm `~/.hermes/profiles/ltx/SOUL.md`. Custom SOUL is not overwritten unless `--force`.

`~/.hermes/config.yaml` fragment (**no secrets** — local stdio only):

```yaml
mcp_servers:
  master-agent:
    command: "<VIDEO_BUDDY>/.venv/bin/python"   # Windows: .venv\Scripts\python.exe
    args:
      - "<VIDEO_BUDDY>/master_agent/mcp_server.py"
```

Prefer `hermes mcp add master-agent -- <venv-python> <VIDEO_BUDDY>/master_agent/mcp_server.py`
then `hermes mcp test master-agent`. Hermes usually spawns the server; to start
by hand from `video_buddy/`: `<venv python> master_agent/mcp_server.py`.

## Status

Actively developed. **LTX 2.5 is on `main`** (merge [PR #6](https://github.com/thcabq352/ltx-video-buddy/pull/6), 2026-09-13) with the Comfy capability audit ([PR #5](https://github.com/thcabq352/ltx-video-buddy/pull/5)).

What landed:

- Seven LTX 2.5 graphs in the **default** catalog — usable with `--variant`, Create-tab, Comfy-tab, `GET /api/variants`. No env flags.
- Inventory-first weight scan. `doctor` never fetches. `download-models --ltx25` lists confirmed-missing files and fetches only after `--yes`.
- 16GB-class loader preference: **GGUF Q4 → NVFP4 (VRAM ≥ 14) → int8-convrot → bf16**.
- WAN / K3NK paths unchanged. LTX TeaCache is inject-when-registered (PR #4); missing pack still soft-bypasses.
- **Not** ported from [ltx2.5-research-agent](https://github.com/thcabq352/ltx2.5-research-agent): LangGraph research/scrape/A2A/Gradio harness, secrets, and the `ltx_research_agent` package.

See the [operator changelog](video_buddy/README.md#status-2026-09-13) for the full history.

## Legal & licensing

Video Buddy is an orchestration layer. The models it drives — LTX 2.5 and
LTX 2.3 (Lightricks), Flux (Black Forest Labs), Wan, Qwen, and others — are
third-party works under their own licenses, several of which restrict
commercial use. LTX 2.5 Hub packs are gated
([`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5), LTX-2.x
Community License). Selected workflow designs credit
[Mickmumpitz](https://mickmumpitz.ai). **Review each model's license before
commercial deployment.** This repository contains no model weights.

---

*Built for creators and businesses who want a production studio, not a
science project.*
