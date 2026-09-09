---
name: video-buddy
description: "Use when running Video Buddy local video on this machine: LTX, Wan, lipsync, music video, fractal, ComfyUI API graphs, storyboard/render/judge, or the master-agent MCP."
version: 1.1.0
author: Ara / Ringmaster
license: MIT
platforms: [windows, macos, linux]
metadata:
  hermes:
    tags: [video, comfyui, ltx, wan, mcp, video-buddy, master-agent, cli]
    category: media
    related_skills: [ltx-director-hermes, route-to-specialist-profiles, native-mcp]
    homepage: https://github.com/thcabq352/ltx-video-buddy
---

# Video Buddy

## About

VIDEO BUDDY is the local ComfyUI video studio in this tree. The Python package stays `master_agent`; MCP id is `master-agent`. Print the card with `python -m master_agent about`. Comfy is `:8188`. Studio `:8189` is an optional human dashboard (About tab included). Drive graphs with `comfy run`; the director pipeline is `run`. Local VL is `qwen3-vl-heretic`. Personas: `ara`, `exec`, `zod` (Zod always says Admarel). Souls: `studio`, `play`.

## Drive Comfy from the CLI first

Queue this studio's Comfy through `python -m master_agent`, not the studio Comfy tab, not Comfy's own UI, and not generic comfyui MCP.

1. `health` — Comfy on `:8188` must be up.
2. `comfy run` to drive a graph, or `run` for the director pipeline.
3. Confirm a real file under `video_buddy/outputs/` before claiming done.

Studio `:8189` is a human dashboard on the same prepare/lint/queue path. MCP is for Hermes after the CLI path works. A live studio tab is not proof Comfy is up.

## Identity

- **Name:** Video Buddy
- **Code home:** `C:\Users\thcab\Documents\ltx2.3_agent\video_buddy`
- **Repo:** https://github.com/thcabq352/ltx-video-buddy
- **Not** a Hermes profile. External friend next to hardline/forge/coder.
- **Package:** `python -m master_agent` (MCP id `master-agent`)
- **Comfy API:** portable `ComfyUI_windows_portable\run_api_8188.bat` — `:8188`
- **Studio UI:** `python -m master_agent ui --port 8189` (optional)
- Personas: `ara`, `exec`, `zod` (Zod always addresses the user as Admarel)
- Souls: `studio`, `play`
- Local VL: `qwen3-vl-heretic`. Auto LLM: `ollama → grok`. No Kimi.

## When to use

- Local brief → storyboard → validate → render → judge → stitch
- LTX 2.3 / Wan 2.2 / lipsync / music video / fractal / Movie Builder
- Drive or lint a Comfy API graph (`comfy run`)
- Character → Flux sheet → LoRA (CCC)
- KB search over **this** studio's workflows/runs

## When NOT to use

| Signal | Route |
|---|---|
| Pixie Forge brand / scotty.fyi / HyperFrames | `hermes -p forge` |
| Grok cloud ≤15s clip | default Grok video skills |
| Ocala SMB / Agent OS | hardline |
| Generic node poke on a **different** Comfy (AraSM) | forge + that install's MCP |

## Preflight

```bash
cd "C:/Users/thcab/Documents/ltx2.3_agent/video_buddy"
PY=./.venv/Scripts/python.exe
$PY -m master_agent health
```

If Comfy is down: start `ComfyUI_windows_portable\run_api_8188.bat` and leave it open. VRAM is 16GB — do not stack a heavy Video Buddy job with another Comfy or a giant Ollama VL.

## Drive Comfy (CLI)

Default `comfy run` prepares, lints, queues, polls, and copies into `outputs/`. `--prepare` stops before the GPU queue.

```bash
$PY -m master_agent about
$PY -m master_agent comfy run --mode generate --prompt "BRIEF" --variant base
$PY -m master_agent comfy run --mode template --template base --set 12.steps=8
$PY -m master_agent comfy run --mode template --template lipsync --prepare --out prepared.json
$PY -m master_agent comfy run --mode raw --json workflow.json
```

Templates: slugs `base`, `eros`, `directors`, `lipsync`, `wan22`, or a path under `workflows/`. `--set NODE.FIELD=VALUE` is repeatable.

## Director and other CLI

```bash
$PY -m master_agent run "BRIEF" --quality draft --duration 3 --no-interview
$PY -m master_agent run "BRIEF" --duration 8 --dry-run --no-interview
$PY -m master_agent run "talking head" --video input.mp4 --variant lipsync --no-interview
$PY -m master_agent music "synthwave MV" --audio track.mp3 --no-interview
$PY -m master_agent fractal "title" --duration 20 --target seahorse
$PY -m master_agent persona set zod
$PY -m master_agent kb search "lipsync" -k 5
```

Long GPU jobs: background the terminal. Intake is on unless `--no-interview`.

## Hermes MCP (secondary)

Server id **`master-agent`**. Register with `hermes mcp add` / `hermes mcp test master-agent` — do not hand-edit Hermes config from this skill. Tools: `health`, `create_video`, `plan_storyboard`, `judge_asset`, `search_workflows`, `search_runs`, `kb_ingest`, `list_models`, `validate_workflow`, `create_character`, `train_lora`. Prefer ~900s tool timeout.

Canonical skill: `skills/video-buddy/SKILL.md`. Also `~/.agents/skills/video-buddy` and `~/.hermes/skills/media/video-buddy`.

## Pitfalls

| Excuse | Reality |
|---|---|
| "I'll just click Queue in the studio" | Agents use `comfy run` / `run`. UI is for humans. |
| "Studio tab is up" | That is `:8189`. Comfy is `:8188`. `health` first. |
| "comfyui MCP is faster" | Wrong box unless the job is on a different Comfy install. |
| "Interview will block me" | Unattended runs pass `--no-interview`. |
| Old path `kimi ltx/` | Gone. Never point MCP there. |

## Verification

- [ ] `python -m master_agent health` reports Comfy up before GPU work
- [ ] Graph jobs used `comfy run` (or `--prepare` when only linting)
- [ ] Output path exists under `video_buddy/outputs/` before claiming done
