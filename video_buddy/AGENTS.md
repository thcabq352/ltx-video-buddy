# Video Buddy — agent notes

Local ComfyUI video studio. Code lives in this directory. Git root is the parent `ltx2.3_agent` repo (`thcabq352/ltx-video-buddy`).

Authoritative framing: **Briefing for LTX Video Buddy (`master_agent`) — fleet curriculum & field notes** (2026-09-10). Stop-lines in §6 of that briefing are implemented first-class here. Do not skip the curriculum path.

## Install

From this folder, one command on any OS:

```powershell
python install.py
```

Windows: `install.bat`. macOS / Linux: `./install.sh`. Re-check with `python -m master_agent setup`; install missing pieces with `--fix`.

## About

`python -m master_agent about` prints the studio card (also `GET /api/about` and the studio About tab). VIDEO BUDDY is the local ComfyUI studio; package `master_agent`; MCP `master-agent`.

## Curriculum

Print the card anytime: `python -m master_agent curriculum`.

### Part 1 — `LESSON_BUDDY_WORKS_HERE` (L0→L5)

Do these in order. Do not jump to overnight work.

| Lesson | Name | Do |
|---|---|---|
| L0 | tree | Know this tree: `video_buddy` / `master_agent`. Not sibling studios (`ltx_director/`, `SOS/`, `lot/`). |
| L1 | about | `python -m master_agent about` — studio card before any GPU claim. |
| L2 | health | `python -m master_agent health` — Comfy **:8188** up. Studio `:8189` is not proof. |
| L3 | dry-run | `python -m master_agent run "BRIEF" --dry-run` — plan/lint only. No queue, no shift-budget spend. |
| L4 | speed diagnose | `python -m master_agent diagnose --variant base --prompt "garden proof"` — 9-frame hull, print `sec/step`. |
| L5 | short proof | A real file in `outputs/`. `ffprobe` frames + size. Junk `<100KB` or `<3` frames is FAIL. |

### Part 2 — `LESSON_BUDDY_PART2_OVERNIGHT`

Gated on **Part 1 L5 + human OK**. Do not start overnight/long-run work until a 9-frame proof exists and a human says OK.

## Run

Drive Comfy from the CLI first. Studio `:8189` is optional.

```powershell
.\.venv\Scripts\python.exe -m master_agent curriculum
.\.venv\Scripts\python.exe -m master_agent about
.\.venv\Scripts\python.exe -m master_agent health
.\.venv\Scripts\python.exe -m master_agent diagnose --variant base --prompt "garden proof"
.\.venv\Scripts\python.exe -m master_agent comfy run --mode generate --prompt "BRIEF" --variant base
.\.venv\Scripts\python.exe -m master_agent run "BRIEF" --quality draft --duration 3 --no-interview
.\.venv\Scripts\python.exe -m master_agent ui --port 8189
```

`comfy run` prepares, lints, queues, and copies into `outputs/`. `--prepare` stops before the GPU queue. Comfy portable is `:8188` (`ComfyUI_windows_portable\run_api_8188.bat`). Studio is `:8189`. Do not treat a live studio tab as proof Comfy is up.

Diagnose the hull before any scale. This is a 9-frame short fire (steps 6–8, fixed seed), not a 121-frame burn:

```powershell
.\.venv\Scripts\python.exe -m master_agent diagnose --variant base --prompt "garden proof"
```

It always prepares + lints first, then queues the short fire, prints wall time and `sec/step`, copies into `outputs/`, and ffprobes the file. Junk (`<100KB` or `<3` frames) is FAIL. Diagnose does **not** spend shift budget. `--prepare` stops after lint. Do not walk `DOWNSCALE_LADDER` or raise resolution until the hull has recorded `sec/step`. `--dry-run` on `run` stays plan-only (no GPU, no used increment).

```powershell
.\.venv\Scripts\python.exe -m master_agent budget status
.\.venv\Scripts\python.exe -m master_agent budget reset-shift
```

## Field lessons

- **`length=8` is junk.** LTX wants `8n+1` (min 9). Length 8 collapses to a 1-frame file. Never queue 8; snap to 9.
- **Know `sec/step` before scale.** Ada field note: 285 s/step dropped to 66.5 s/step after a lowvram / hull-sized fire. Do not climb `DOWNSCALE_LADDER` or raise frames until diagnose recorded `sec/step`.
- **Audio field is `frames_number`** on `LTXVEmptyLatentAudio`, paired with video `length`.
- **Missing TeaCache → bypass**, not a hard-fail. Do not auto-install packs. Live Comfy may only expose `WanVideoTeaCache` / `WanVideoTeaCacheKJ`.
- **LTX 2.5 default catalog.** `ltx25_*` variants are listed with no env flag. Scan weights first (GGUF Q4 / NVFP4 / int8 / heretic TE all count); ask before download (`python -m master_agent download-models --ltx25`). Do not auto-download gated `Lightricks/LTX-2.5` files. Zero-byte files are missing. Stub ckpt remaps to the official split pack; local GGUF wins when present.
- **Tracker `DONE` can lie.** Confirm with history + `ffprobe` + size/frames. Junk `<100KB` or `<3` frames is FAIL even if the tracker says done.
- **Port-in-use ≠ kill the cook.** If `:8188` / `:8189` is already bound, do not kill a running render. Attach or wait.
- **Budget ~80 VRAM-min HOLD** with a shift-reset ledger. HOLD is `input-required`, not failed. `budget reset-shift` archives `previous_used` / `previous_shift_id` and never wipes history.
- **Windows folder-prefixed weights** stay backslash style (`wan\file.safetensors`).
- **LoRA A/B locks the encoder.** `gemma_3_12B_it_fp8_scaled` is the proven TE family. Do not swap the default TE for Heretic mid-A/B. Only LoRA name/strength may change.

## Judge ≠ taste

The judge is a **coherent-take / retry ladder**, not an album curator.

- Admiral / human eyes beat the judge for **aesthetic album lock**. `album_lock` from the judge is always false.
- Split **look** (craft/motion) from **health** (probe/metadata: size, frames, duration). Discount metadata noise; retries use `look_score` only.
- Low `brief_adherence` + high look → `human_veto` / `HUMAN_VETO`. Still no album-lock.

## Stop-lines (briefing §6)

- **LTX frame law.** Valid lengths are `8n+1`, minimum **9**. `snap_ltx_frames()` never returns 8. The patcher writes `EmptyLTXVLatentVideo.length` and `LTXVEmptyLatentAudio.frames_number` together. Validator WARNs and auto-corrects unless `--strict` (ERROR). `DEFAULT_FRAMES`, diagnose, and draft start at 9. Long-run templates keep their length but still snap illegal counts. `DOWNSCALE_LADDER` first rungs are 9/17/25/33 — not 121.
- **Diagnose before scale.** Measure `sec/step` on a 9-frame hull before raising res/frames. Scale is refused until `state/control/diagnose_hull.json` has `sec_per_step`.
- **TeaCache bypass.** Missing optional accelerators (`TeaCache`, `WanVideoTeaCache`, `WanVideoTeaCacheKJ`, …) are WARNING + bypass: MODEL (or the typed upstream link) is rewired past the node and the node is dropped. Validation still PASSes. Do not auto-install packs.
- **Judge look vs health.** Payload has `look_score`, `health_score`, and `combined_score` (back-compat). Retry ladder uses **look only**. Low `brief_adherence` + high look is `human_veto`. Probe `min_frames=3`; junk `<100KB` is a health fail.
- **Shift reset.** `RenderBudget.reset_shift()` archives `{event:shift_reset, previous_used, previous_shift_id}` then `used=0` `paused=False`. Never wipe the ledger. Diagnose/dry-run do not increment `used`. HOLD is A2A `input-required`, not `failed`.
- **LoRA A/B.** When comparing LoRAs, lock encoder + seed family; only LoRA name/strength may change. Do not swap `gemma_3_12B_it_fp8_scaled` / the default TE for Heretic mid-A/B. Windows folder-prefixed checkpoint names stay backslash style.

## Identity

Persona is the interview voice. Soul is standing studio values.

- Bundled personas: `ara`, `exec`, `zod` — override in `state/personas/<slug>.md`
- Bundled souls: `studio`, `play` — override in `state/souls/<slug>.md`
- Switch: `python -m master_agent persona set <slug>` / `soul set <slug>`, Create-tab selects, or `PERSONA` / `SOUL` env
- Changes go through versioned config (`state/control/config.json` + history)

## Local model

Default Ollama / vision model is `qwen3-vl-heretic`. Auto LLM chain is `ollama → grok`. Do not add a Kimi provider. That Heretic VL is **not** the LTX text encoder — do not swap it onto LoRA A/B graphs.

## Control layer

Render budget (VRAM-minutes, cap ~80) can pause the queue over cap. Each shift has `shift_id`; `budget reset-shift` archives the previous used total and zeros the counter without wiping history. `/api/control` exposes knobs, budget, and last-10 config history. A2A lives at `GET /.well-known/agent.json` and `POST /a2a` (studio port **8189**). Budget-held pipeline jobs map to A2A `input-required`, not `failed`.

## Tests

Use the project venv. Prefer one file at a time for GPU-adjacent work:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_curriculum.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_ltx_frames.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_accelerator_bypass.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_judge_split.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_shift_budget.py -q
```

Do not commit models, `state/runs`, Comfy portable, or sibling trees (`ltx_director/`, `SOS/`, `lot/`).
