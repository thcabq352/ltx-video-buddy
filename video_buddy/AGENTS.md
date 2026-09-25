# Video Buddy — agent notes

Local ComfyUI video studio. Code lives in this directory. Git root is the parent `ltx2.3_agent` repo (`thcabq352/ltx-video-buddy`).

Authoritative framing: **Briefing for LTX Video Buddy (`master_agent`) — fleet curriculum & field notes** (2026-09-10). Stop-lines in §6 of that briefing are implemented first-class here. Do not skip the curriculum path.

## Install

From this folder, one command on any OS:

```powershell
python install.py
```

Windows: `install.bat`. macOS / Linux: `./install.sh`. Re-check with `python -m master_agent doctor` (alias of `setup`; **does not fetch weights**). Install missing *deps* with `--fix`. Confirmed-missing LTX 2.5 weights: `python -m master_agent download-models --ltx25` then `--yes` after you agree. MiniMax H3: `python -m master_agent download-models --h3` then `--yes`.

## Hermes install

MCP tools alone are not enough — Hermes loads capabilities from a real skill
folder. Source of truth: [`skills/video-buddy/`](skills/video-buddy/SKILL.md).

```powershell
python install_hermes_skill.py
```

That copies `skills/video-buddy/` to `~/.hermes/skills/video-buddy/` and seats
Hermes profile `ltx` at `~/.hermes/profiles/ltx/` (Windows / macOS / Linux;
`HERMES_HOME` overrides `~/.hermes`). No `.env` is written. Confirm server id
`master-agent` in the profile (or default) `config.yaml` (no secrets):

```yaml
mcp_servers:
  master-agent:
    command: "<VIDEO_BUDDY>/.venv/bin/python"   # Windows: .venv\Scripts\python.exe
    args:
      - "<VIDEO_BUDDY>/master_agent/mcp_server.py"
```

Start MCP (cwd = this folder): `<venv python> master_agent/mcp_server.py`.
Prefer `hermes -p ltx` / `hermes mcp add` / `hermes mcp test master-agent`.
Primary discovery is the `ltx` seat (or the studio facade on `:8189`).
A2A (`GET /.well-known/agent.json`, `POST /a2a`) is the fallback. Never bind 8642.

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
.\.venv\Scripts\python.exe -m master_agent doctor
.\.venv\Scripts\python.exe -m master_agent workflows
.\.venv\Scripts\python.exe -m master_agent capabilities --offline
.\.venv\Scripts\python.exe -m master_agent health
.\.venv\Scripts\python.exe -m master_agent diagnose --variant base --prompt "garden proof"
.\.venv\Scripts\python.exe -m master_agent comfy run --mode generate --prompt "BRIEF" --variant base
.\.venv\Scripts\python.exe -m master_agent comfy run --mode generate --prompt "BRIEF" --variant ltx25_t2v_i2v
.\.venv\Scripts\python.exe -m master_agent comfy run --mode generate --prompt "BRIEF" --variant h3_t2v
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
- **TeaCache:** `graph_ops.ensure_teacache` injects welltop-cn `TeaCache` on LTX MODEL after LoRA / before guider when `/object_info` has the class. LTX defaults: `model_type=ltxv`, `rel_l1_thresh=0.06`, `start_percent=0`, `end_percent=1`, `cache_device=cuda`; `max_skip_steps=3` only if the node exposes it. Pack: https://github.com/welltop-cn/ComfyUI-TeaCache (~1.7x). Templates stay clean — the patcher / `comfy run` inject. If the class is missing, do **not** insert; leftover nodes still **bypass** (WARN + rewire). Do not auto-install packs. Wan-only aliases (`WanVideoTeaCache`, `WanVideoTeaCacheKJ`) are not LTX TeaCache.
- **LTX 2.5 default catalog (PR #6, merged).** `ltx25_*` variants (and research aliases `t2v_i2v` / `flf2v` / …) are listed with no env flag. Inventory first. 16GB-class loader pick: GGUF Q4 → NVFP4 (if `VRAM_GB` ≥ 14) → int8-convrot → bf16. Heretic/int8 TE counts; official bf16 Gemma is not required. The duration head is optional (no shipped 2.5 graph loads it). A zero-byte duration-head is not present and does not fail `ltx25-weights`. `doctor` reports the pick and does **not** fetch. `download-models --ltx25` lists confirmed-missing; `--yes` only after the ask.
- **MiniMax H3 default catalog.** `h3_t2v` / `h3_i2v` / `h3_flf` (fl2va) and `h3_r2v` (ref2va). Aliases `fl2va` / `ref2va`. GGUF Q4_K DiT first, then NVFP4/int8. Prefer Comfy Qwen3-VL TE (NVFP4 AWQ / int8 / int4) — GGUF TE Q4_K_M is too heavy as default. CFG stays 1.0. 16GB sweet spot: 0.6–0.8 MP, ≤12 s, 4 steps. `download-models --h3` lists confirmed-missing; `--yes` only after the ask. Do not break LTX 2.5 / WAN paths.
- **Brain / Hands (sibling PR #9).** Director ranks stories only. Hands (`can_fulfill`) answers possible-right-now from a live/object_info/inventory snapshot (`FitResult`). `CapabilityContract` is `buddy.capability.contract/v1` — model/family/variant, resolution, duration_s, story_duration_s, audio, control_layers. **No** VRAM/slot/weight-path on the contract. Long stories: Hands `plan_last_frame_chain` (20s → 3×8s). Music-video beat windows stay on the MTV path. See [`docs/BRAIN_HANDS.md`](docs/BRAIN_HANDS.md).
- **Shared 16GB policy** (`master_agent.models.vram_policy`). Hands / doctor / loaders inherit GGUF Q4/Q5 → NVFP4 → int8/fp8. Doctor row `vram-policy`. `workflows --vram` prints the family table. The **director brain does not** pick cheaper-GPU graphs from this table. Heavy graphs stay available when the story names them; Hands rejects if the snapshot cannot fulfill. K3NK AIO is not a default (no attested pack). Lightx2v / turbo LoRAs when present; LTX TeaCache inject-when-registered (soft-bypass if the pack is missing); `DOWNSCALE_LADDER` after diagnose.
- **Missing TeaCache / LanPaint_KSampler / GetWarpedNoiseFromVideo / MMAudio* → bypass**, not a hard-fail. Do not auto-install packs. Live tower (2026-09-13, 4114 classes) registers exact `TeaCache` + `WanVideoTeaCache`. There is no `VideoNoiseWarp` class — use `GetWarpedNoiseFromVideo`. Do not bypass structural nodes (`WanFunInpaintToVideo`, `Wan22FunControlToVideo`, `IPAdapterFaceID`). WAN / K3NK / TeaCache paths were **not** rewritten by the 2.5 merge.
- **Tracker `DONE` can lie.** Confirm with history + `ffprobe` + size/frames. Junk `<100KB` or `<3` frames is FAIL even if the tracker says done.
- **Port-in-use ≠ kill the cook.** If `:8188` / `:8189` is already bound, do not kill a running render. Attach or wait.
- **Budget ~80 VRAM-min HOLD** with a shift-reset ledger. HOLD is `input-required`, not failed. `budget reset-shift` archives `previous_used` / `previous_shift_id` and never wipes history.
- **Windows folder-prefixed weights** stay backslash style (`wan\file.safetensors`).
- **LoRA A/B locks the encoder.** `gemma_3_12B_it_fp8_scaled` is the proven TE family. Do not swap the default TE for Heretic mid-A/B. Only LoRA name/strength may change.

## TeaCache

Buddy-native path is **inject-when-registered**, not baked JSON:

1. `master_agent.comfy.graph_ops.ensure_teacache` is the only writer.
2. `load_and_patch_workflow` (diagnose / draft / generate) and `prepare_run` (`comfy run` raw/template) call it on LTX graphs (`base`, `eros`, `directors`, `lipsync`).
3. If `TeaCache` is in `/object_info` (or `state/object_info.json` cache), a welltop-cn node is inserted on MODEL after the last LoRA (`LoraLoaderModelOnly` / `LTXICLoRALoaderModelOnly`) and before `MultimodalGuider` / `CFGGuider` / sampler.
4. If the class is missing, inject is a no-op. Any leftover TeaCache-style node is still **soft-bypassed** (WARN + rewire). Validation PASSes. Packs are never auto-installed.

| Widget | LTX default |
|---|---|
| `class_type` | `TeaCache` |
| `model_type` | `ltxv` (or `LTX-Video` if that is the only enum) |
| `rel_l1_thresh` | `0.06` |
| `start_percent` | `0` |
| `end_percent` | `1` |
| `cache_device` | `cuda` (omitted if the node has no such input) |
| `max_skip_steps` | `3` **only if** the pack exposes the input |

Install on Scott’s Comfy: https://github.com/welltop-cn/ComfyUI-TeaCache (LTX-Video, ~1.7x). Wan-only `WanVideoTeaCache` / `WanVideoTeaCacheKJ` aliases are **not** this node; they still bypass when missing.

## Judge ≠ taste

The judge is a **coherent-take / retry ladder**, not an album curator.

- Admiral / human eyes beat the judge for **aesthetic album lock**. `album_lock` from the judge is always false.
- Split **look** (craft/motion) from **health** (probe/metadata: size, frames, duration). Discount metadata noise; retries use `look_score` only.
- Low `brief_adherence` + high look → `human_veto` / `HUMAN_VETO`. Still no album-lock.
- Cheap Quality Bar (buddy-core ids, no GPU): **a** missing music bed, **c**
  thin still→I2V / missing previs, **d** unused control pack. **b** face
  scores are not ported. Fail → `RevisePlan` → re-run → re-judge. Terminal
  `loop_status` is `passed` / `exhausted` / `human_veto` / `error`.
  `--self-improve-dry` closes the loop without Comfy. See
  [`docs/SELF_IMPROVEMENT_LOOP.md`](docs/SELF_IMPROVEMENT_LOOP.md).
  Every clip stores ClipProvenance (`buddy.clip.provenance/v1`) as
  `{output_dir}/shot-N.buddy.json` plus the same object on the run JSON
  (`output_path` / `hash`). Sidecar is the source of truth before revise.
  See [`docs/CLIP_PROVENANCE.md`](docs/CLIP_PROVENANCE.md).

## Stop-lines (briefing §6)

- **LTX frame law.** Valid lengths are `8n+1`, minimum **9**. `snap_ltx_frames()` never returns 8. The patcher writes `EmptyLTXVLatentVideo.length` and `LTXVEmptyLatentAudio.frames_number` together. Validator WARNs and auto-corrects unless `--strict` (ERROR). `DEFAULT_FRAMES`, diagnose, and draft start at 9. Long-run templates keep their length but still snap illegal counts. `DOWNSCALE_LADDER` first rungs are 9/17/25/33 — not 121.
- **Diagnose before scale.** Measure `sec/step` on a 9-frame hull before raising res/frames. Scale is refused until `state/control/diagnose_hull.json` has `sec_per_step`.
- **TeaCache.** Single path: `ensure_teacache` in `graph_ops` (called from `load_and_patch_workflow` and `prepare_run`). When `TeaCache` exists in object_info, insert/refresh welltop-cn node after LoRA before sampler/guider (`ltxv`, `rel_l1_thresh=0.06`, `start_percent=0`, `end_percent=1`; `max_skip_steps=3` if exposed). When missing, skip insert; leftover `TeaCache` / `WanVideoTeaCache*` still WARNING + bypass and validate PASSes. Install on Scott’s Comfy: `https://github.com/welltop-cn/ComfyUI-TeaCache`. Do not auto-install packs. Do not bake TeaCache into JSON templates.
- **Optional-node bypass.** Missing optional nodes (`TeaCache`, `WanVideoTeaCache`, `LanPaint_KSampler`, `GetWarpedNoiseFromVideo`, `MMAudioSampler`, …) are WARNING + bypass: the typed upstream link is rewired past the node and the node is dropped. Validation still PASSes. Do not auto-install packs. Do not bypass Fun Inpaint / Fun Control / FaceID / ControlNet / Voronoi — those are payload nodes.
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

Default local / vision model is `qwen3-vl-heretic`. Auto LLM chain is
`ollama → llamacpp → grok`. Do not add a Kimi provider. That Heretic VL is
**not** the LTX text encoder — do not swap it onto LoRA A/B graphs.

### Ollama vs llama.cpp

| | Ollama | llama.cpp |
|---|---|---|
| Env | `OLLAMA_URL` (default `http://127.0.0.1:11434`), `OLLAMA_MODEL` | `LLAMACPP_URL` (default `http://127.0.0.1:8080`), `LLAMACPP_MODEL` |
| Spec | `ollama[:model]` | `llamacpp` / `llama.cpp` / `llama-cpp` / `llamacpp[:model]` |
| Chat | `{url}/v1` OpenAI-compat | `{url}/v1` OpenAI-compat (`llama-server --api`) |
| Embeddings | `POST /api/embed` | `POST /v1/embeddings` (disable KB with a warning if missing) |
| Vision | `POST /api/chat` + images | multimodal `/v1/chat/completions`; skip vision if the server is text-only |

Pin with `LLM_PROVIDER=llamacpp` when Ollama is not installed. Health /
MCP `health` / studio dots report each backend separately — never treat
llama.cpp as Ollama. Do not bind 8642 (Hermes) or 8189 (studio).

## Control layer

Render budget (VRAM-minutes, cap ~80) can pause the queue over cap. Each shift has `shift_id`; `budget reset-shift` archives the previous used total and zeros the counter without wiping history. `/api/control` exposes knobs, budget, and last-10 config history. Hermes profile `ltx` is the primary seat (`python -m master_agent hermes status`). The studio facade is `POST /p/ltx/v1/chat/completions` on **8189** (never 8642). A2A fallback lives at `GET /.well-known/agent.json` (also `agent-card.json`) and `POST /a2a`. Budget-held pipeline jobs map to A2A `input-required`, not `failed`. `done_with_warnings` maps to `completed`.

## Tests

Use the project venv. Prefer one file at a time for GPU-adjacent work:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_curriculum.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_ltx_frames.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_accelerator_bypass.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_judge_split.py tests/test_self_improvement_loop.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_shift_budget.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_brain_hands.py tests/test_ltx25_catalog.py tests/test_ltx25_weights.py tests/test_h3_catalog.py tests/test_h3_weights.py tests/test_capabilities.py tests/test_vram_policy.py -q
```

Do not commit models, `state/runs`, Comfy portable, or sibling trees (`ltx_director/`, `SOS/`, `lot/`).
