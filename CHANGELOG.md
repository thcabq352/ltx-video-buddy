# Changelog

All notable changes to Video Buddy are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/) once 1.0 ships.

> **Proposed versions.** No release has been tagged yet. The version numbers and dates below are a proposed retroactive grouping of the history on `main`, by the date each change merged, and each version links to the last merge in its group. Entries are grouped by theme rather than by Added / Changed / Fixed. How to add to this file: [`docs/RELEASING.md`](docs/RELEASING.md).

## [Unreleased]

### Documentation

- Changelog, release conventions, and a pull request template.
- Operator manual brought in line with `main`: workflow ingest, managed ComfyUI, SageAttention, the tiled-decode VAE guard, the Rainey1 pack, and the full CLI surface.
- Status note on the agent gateway consolidation that is in progress.

## [0.7.0] - 2026-10-08 (proposed)

### Workflow Ingest

- Ingest a one-off ComfyUI graph from API JSON, from UI JSON (converted by a running Comfy), or from a Comfy history entry. Buddy stores it locally with its provenance and never queues on ingest.
- Learn the tunable fields (prompt, negative, seed, size, frames, checkpoint, output prefix) with confidence levels, and report missing nodes, likely node packs, and missing models.
- Dry-run prints the resolved graph without queueing. `comfy run --ingested` queues through the same patcher, frame law, and VAE guard as catalog variants.
- Family routing sends recognized in/outpaint, lipsync, and Sulphur graphs through their existing helpers. `--no-family-route` keeps them generic.
- `comfy promote` writes a local draft manifest entry. It never replaces a catalog default and never runs git.
- Optional `--llm-assist` names low-confidence widgets with a local model (llama.cpp, then Ollama). No cloud call.
- Drop zone on the studio Comfy tab, with queue gated behind an explicit confirm.

### Managed ComfyUI

- Managed launches request SageAttention (`--use-sage-attention`) whenever the `sageattention` package is importable, and log a warning instead of crashing when it is not.

### Rendering Quality

- Tiny preview VAEs (`taeltx*` / `tae*`) are swapped out of tiled decode, or refused when requested explicitly, before anything is queued.

## [0.6.0] - 2026-10-05 (proposed)

### Managed ComfyUI

- `comfy start` / `stop` / `status` / `restart` run a loopback ComfyUI through a pinned comfy-cli, with a crash-restart watchdog.
- Buddy writes its own model-paths YAML so Comfy sees Buddy's model folder plus read-only views of existing installs.
- External mode for a server you already run: Buddy refuses lifecycle commands and never writes into that tree.
- Hardware routing sentence for NVIDIA and AMD (ROCm) cards on doctor and managed launches.
- `comfy update` reports stale packs and only updates after explicit consent, saving a node snapshot first.

### Models and Weights

- LTX 2.3 / 2.5 model selector on the CLI and in a new studio Models tab, with size totals, download ETA, a disk check, and keep-or-wipe when switching versions.

### Workflows

- Sulphur LTX 2.3 studio graphs (image-to-video and text-to-video, base and distilled). LoRA weights stay local and are never downloaded.
- Ingredients IC-LoRA remake recipes for vertical concert and pier shots.

### Rainey1

- Director preset and recipe pack.
- Opt-in judge rubric with look dimensions and hard-fail gates.
- Batch top-cut: one recipe over several seeds, junk filtered before the judge, top-K keepers copied with provenance. Dry-run queues nothing.

### Local LLM

- llama.cpp is the preferred local backend. Buddy starts `llama-server` when it is available and stops it on shutdown, falling back to Ollama.

### Music

- HeartMuLa sizes its attention cache to the clip and the card so typical tracks fit on 16GB, and saves the wav through a fallback when torchaudio cannot.

### Reliability

- The multi-segment pipeline wiring is restored after a merge regression.

### Documentation

- Operator manual rewritten with one home per fact. The pre-slim manuals are kept in `docs/archive/`.
- The repository stands alone as the complete Python product.
- Acknowledgment for Jason Rainey.

## [0.5.0] - 2026-09-30 (proposed)

### Lip Sync and Voice

- A photo plus a voice file routes to LTX 2.5 audio-to-video. MiniMax H3 takes reference audio for a spoken line in the sample's voice (coarse mouth sync).
- Long audio-to-video lipdub is segmented on pauses and stitched under one provenance record, with idle renders for silences, a previous-frame anchor, an opt-in pause reset, and cuts snapped to word edges.

### Workflows

- LTX 2.3 and LTX 2.5 in/outpaint workflows.

### Models and Weights

- Generate resolves weights from the local inventory. GGUF files already on disk are preferred at any VRAM.
- 12GB-class machines get inventory-first weights and local-only defaults.

### Music

- Optional local HeartMuLa: generate a track from lyrics and tags, or transcribe word timestamps for lipdub.

### Local-Only

- Seedance 2.5 Draft requests fail closed onto the local LTX 2.5 catalog. Partner graphs are recorded as field shapes and are never queued.
- LTX graphs no longer include hosted API nodes. Every queued node runs on your ComfyUI.

### Knowledge

- A git-synced `knowledge/` folder shares lessons between machines and is indexed into local Chroma on startup.

### Reliability

- Judge sampling, audio caps, and attempt lineage fixes. LTX 2.3 graphs keep their authored prompt, sampler, and output wiring. The base bundle no longer overwrites unrelated weights. Budget holds resume cleanly.

## [0.4.0] - 2026-09-21 (proposed)

### Judge and Self-Improvement

- Closed judge → revise → re-run loop, with a dry mode that needs no ComfyUI.
- Every clip carries a `buddy.clip.provenance/v1` sidecar that the loop reads before a revise.

### Workflows

- Live Comfy attach applies a previs `buddy.comfy.attach/v1` patch plan, dry-run by default.
- LTX 2.5 graphs converted from the official examples, with last-frame chaining for long stories.
- Brain / Hands capability contract: the director ranks stories, Hands checks what the machine can render now.
- Duration ladder lesson from 1s to 10s.
- Wan Fun Inpaint template wired to a mask.

### Music

- Music-video mode: one unique LTX clip per beat window, stitched with Remotion.

### Rendering Quality

- TeaCache is injected on LTX graphs when the node is registered, and bypassed when it is not.

### Models and Weights

- Local model files are reused instead of re-downloaded. GGUF names match Comfy's loader choices.

### Reliability

- One pooled HTTP client for ComfyUI calls. Best-effort uploads, VRAM frees, and knowledge ingests log failures instead of hiding them. Wider no-GPU CI coverage.

## [0.3.0] - 2026-09-15 (proposed)

### Models and Weights

- LTX 2.5 default catalog with inventory-first, scan-before-ask weights. Doctor reports and never downloads.
- MiniMax H3 catalog (text, image, first-last frame, and reference to audio-video) with GGUF-first weights.
- Shared 16GB pack policy across Hands, doctor, and loaders.

### Agent Integration

- Hermes skill package and profile `ltx`, with A2A as a fallback.

### Local LLM

- llama.cpp as a first-class local LLM backend beside Ollama.

### Reliability

- Capability audit against live Comfy classes. Missing optional accelerators bypass instead of failing lint. Retired paths are marked retired.

## [0.2.0] - 2026-09-11 (proposed)

### Studio

- Control layer, Zod persona, and one-command setup (`install.py`).
- Comfy CLI-first workflow and a studio About card.

### Rendering Quality

- LTX `8n+1` frame law with safe defaults, and a 9-frame diagnose that records seconds per step before scaling.

### Judge and Self-Improvement

- Judge split into look versus health, with a human veto when the look is strong but the brief is missed.
- LoRA A/B comparisons lock the encoder and seed family.

### Reliability

- Render shift budget ledger with a non-destructive shift reset.
- L0→L5 operator curriculum.

## [0.1.0] - 2026-08-17 (proposed)

### Studio

- First release of the local LTX video agent over ComfyUI, with a workflow library, a landing README, and a validated Movie Builder graph.
- Studio UI with power mode, voice chat, fractal paint, and media views.
- Hermes skills hub package.

[Unreleased]: https://github.com/thcabq352/ltx-video-buddy/commits/main
[0.7.0]: https://github.com/thcabq352/ltx-video-buddy/commit/729fdd0
[0.6.0]: https://github.com/thcabq352/ltx-video-buddy/commit/c51b5b8
[0.5.0]: https://github.com/thcabq352/ltx-video-buddy/commit/90d2bdb
[0.4.0]: https://github.com/thcabq352/ltx-video-buddy/commit/fbc6029
[0.3.0]: https://github.com/thcabq352/ltx-video-buddy/commit/9302785
[0.2.0]: https://github.com/thcabq352/ltx-video-buddy/commit/bf8f4fa
[0.1.0]: https://github.com/thcabq352/ltx-video-buddy/commit/195e828
