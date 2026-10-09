---
name: video-buddy
description: "Use when generating or judging local ComfyUI video on this machine with Video Buddy (LTX 2.5, MiniMax H3, WAN, lipsync, music video, fractal, storyboard) through the LTX bot gateway and `python -m master_agent agent`. Use when an agent is about to treat Grok Imagine / cloud-only video as the local studio, or reaches for a Buddy MCP server, A2A endpoint, or studio port that no longer exists."
---

# Video Buddy

Local ComfyUI video studio in this repo (`video_buddy/`). Package is `master_agent`. Not cloud Imagine.

**One gateway.** The LTX bot gateway is the only way in. It loads this skill and runs `python -m master_agent agent <tool> --args '<json>'` from `video_buddy/` with the project venv. Buddy runs no server of its own: no MCP server, no A2A endpoint, no studio port, no Hermes seat registration.

## When to use

- Local brief → storyboard → validate → render → judge → stitch on **this** machine
- LTX 2.5 (`ltx25_*` / aliases `t2v_i2v`, `flf2v`, …), MiniMax H3 (`h3_*` / `fl2va` / `ref2va`), LTX 2.3 (`base` / `eros` / `directors` / `lipsync`), Wan 2.2, music video, fractal, Movie Builder
- Seedance 2.5 Draft → Final (Pack C). **Local-only is a hard requirement.** Generate on `http://127.0.0.1:8188` with `ltx25_t2v_i2v`, `ltx25_flf2v`, or `ltx25_msr`. Partner ids are a field-shape record and are not queued. See `references/local-only.md` and repo `docs/FEATURES.md`
- Drive or lint a Comfy API graph (`comfy run`)
- Inventory-first weights: `inventory`, then `doctor` (no fetch), then `download-models --ltx25` or `--h3` if a slot is confirmed missing. `--heartmula` lists HeartMuLa slots and does not fetch until `--yes` (check free space first; prefer `HeartCodec-oss-20260123`). `--scan-only` never fetches. GGUF is the loader whenever a compatible file is on disk; fp8/bf16/EROS are fallbacks. 12GB: `VRAM_GB=12` and `FORCE_LOADER=gguf` (hides NVFP4/bf16 suggestions; does not block a lone bf16). Fully local LLM: `LLM_PROVIDER=ollama`. HeartMuLa is sequential with LTX on 16GB. On 16GB the backbone KV window is sized under heartlib's 8192 (that window OOMs during the GQA expand); `--max-seq-len 512` is the 5s smoke. Wav save falls back to soundfile when torchaudio/torchcodec fails.
- Character → Flux sheet → LoRA (CCC)

## When NOT to use

| Signal | Route elsewhere |
|---|---|
| Grok / Imagine cloud clip, no local Comfy | default cloud video skills — not Buddy |
| Pixie Forge brand / scotty.fyi / HyperFrames | `hermes -p forge` |
| Generic node poke on a **different** Comfy install | that install's own tooling |
| A checkout other than this `video_buddy/` tree | that checkout's own docs — L0 is this package only |

## Calling Buddy

```bash
# cwd = video_buddy/, project venv
python -m master_agent agent list
python -m master_agent agent health
python -m master_agent agent create_video --args '{"request": "rain on a pier at dawn", "dry_run": true}'
python -m master_agent agent create_video --args @job.json
```

- stdout is exactly one JSON document; logs and the config banner go to stderr.
- Exit 0 = ok. Exit 1 = the result has `status` `error` or `busy`. Exit 2 = unknown tool or bad `--args`.
- `busy` means another render holds the GPU (`state/gpu.lock`). Nothing was queued; retry when it finishes.
- `paused` means the render budget held the run. `control_set` with `reset_budget: true` (or `budget_reset_shift`) resumes it.
- Media paths must sit under `outputs/`, `state/uploads/`, the ComfyUI input/output folders, or `MEDIA_EXTRA_ROOTS`.
- Allow ~900s per call. `create_video` / `judge_asset` / `train_lora` are long.

Python callers can `from master_agent import agent_api` and call the same functions.

**Do not invent tools.** The tools are the ones `agent list` prints (`master_agent/agent_api.py`). Diagnose, curriculum, `comfy run`, doctor, ingest/promote are **CLI commands**, not agent tools.

## Curriculum stop-lines (before overnight)

Do L0→L5 in order. Print the card: `python -m master_agent curriculum`. Part 2 overnight is gated on **L5 + human OK**.

| Lesson | Do |
|---|---|
| L0 tree | This tree: `video_buddy` / `master_agent`. |
| L1 about | `python -m master_agent about` — studio card before any GPU claim. |
| L2 health | `python -m master_agent health` — Comfy **:8188** up. A gateway reply is not proof. |
| L3 dry-run | `python -m master_agent run "BRIEF" --dry-run` — plan/lint only. No queue, no shift-budget spend. |
| L4 diagnose | `python -m master_agent diagnose --variant base --prompt "garden proof"` — 9-frame hull, print `sec/step`. |
| L5 proof | Real file in `outputs/`. `ffprobe` frames + size. **Junk <100KB or <3 frames = FAIL.** |

Hard rules:

- **Diagnose before scale.** Do not climb `DOWNSCALE_LADDER` or raise res/frames until `state/control/diagnose_hull.json` has `sec_per_step`.
- **LTX frames are `8n+1`, minimum 9.** Never queue length **8** (collapses to 1 frame). Snap to 9.
- Tracker `DONE` can lie. Confirm path + `ffprobe` + size. Junk still FAIL.
- `doctor` never downloads. `--yes` only after the missing-slot ask.

## Ports

| Port | What | Proof |
|---|---|---|
| **:8188** | ComfyUI API | `health` / `agent health` shows Comfy up |
| **:8080** | llama.cpp `llama-server` (preferred local LLM) | `agent health` → `local_llm.llamacpp` |
| **:11434** | Ollama (second local LLM) | `agent health` → `ollama` — not implied by llama.cpp |

Buddy binds no gateway port. Do not bind **8642** (Hermes API) or treat llama.cpp as Ollama.

## llama.cpp vs Ollama

llama.cpp is the preferred local backend. Buddy starts `llama-server` (`LLAMACPP_BIN` or `PATH`) against `MODELS_DIR` on first use; `LLAMACPP_AUTOSTART=0` uses one that is already listening. If that binary is missing, auto warns and uses Ollama. Pin `llamacpp` or `ollama` to force one backend.

| | llama.cpp | Ollama |
|---|---|---|
| Env | `LLAMACPP_URL`, `LLAMACPP_MODEL`, `LLAMACPP_BIN` | `OLLAMA_URL`, `OLLAMA_MODEL` |
| Provider | `LLM_PROVIDER=llamacpp` (aliases `llama.cpp`, `llama-cpp`) or `auto` | `LLM_PROVIDER=ollama` |
| Chat / storyboard | `{url}/v1/chat/completions` | `{url}/v1/chat/completions` |
| Embeddings | `/v1/embeddings` (KB no-ops if missing) | `/api/embed` |
| Vision judge | multimodal `/v1/chat/completions`; heuristic-only if the GGUF is text-only | `/api/chat` + images |

`auto` order: llamacpp → ollama. It never calls a cloud model; Grok is opt-in (`LLM_PROVIDER=grok` or a panel naming it). Panels accept `llamacpp[:model]`.
`health` reports llama.cpp and Ollama, and whether opt-in Grok credentials exist (read-only).

## Agent tools ↔ CLI

Full signatures: [TOOLS.md](TOOLS.md).

| Agent tool | CLI equivalent |
|---|---|
| `health` | `python -m master_agent health` (Comfy only) |
| `about` | `python -m master_agent about --json` |
| `create_video` | `python -m master_agent run "BRIEF" [--duration N --quality draft --variant SLUG --seed N --storyboard MODE --upscale rtx --dry-run --no-interview]` |
| `plan_storyboard` | No GPU plan preview. Closest: `run "BRIEF" --dry-run` (also lints). |
| `judge_asset` | Agent tool only. |
| `search_workflows` / `search_runs` / `search_knowledge` | `python -m master_agent kb search "QUERY" [--workflows / --knowledge] -k 3` |
| `kb_ingest` | `python -m master_agent kb ingest` |
| `list_runs` | Agent tool only (reads `state/runs/`). |
| `list_models` | `python -m master_agent scan-models` |
| `validate_workflow` | `python -m master_agent validate path.json` |
| `create_character` | `python -m master_agent character create "DESC" [--name N --shots N --train]` |
| `train_lora` | `python -m master_agent lora train NAME [--steps N --lr X --rank N --validate]` |
| `control_get` / `control_set` | `persona set` / `soul set` cover two knobs; judge, VRAM and budget-cap knobs are agent tools only. |
| `budget_status` / `budget_reset_shift` | `python -m master_agent budget status --json` / `budget reset-shift` |

CLI-only: `curriculum`, `inventory`, `doctor`/`setup`, `workflows`, `capabilities`, `download-models`, `download-flux`, `models select`, `comfy run|ingest|learn|dry-run|promote|attach|start|stop|status|restart|update`, `diagnose`, `rainey1-batch`, `fetch-object-info`, `power-tune`, `persona`, `soul`, `brief`, `fractal`, `music`, `mv plan`/`mv render`, `heartmula generate`/`heartmula transcribe`, `lora setup`/`validate`, `character list`.

Drive graphs with CLI first: `comfy run`. Director pipeline: `run` or `agent create_video`. Unattended: `--no-interview`.

```bash
python -m master_agent health
python -m master_agent comfy run --mode generate --prompt "BRIEF" --variant ltx25_t2v_i2v
python -m master_agent comfy run --mode generate --prompt "BRIEF" --variant h3_t2v
python -m master_agent run "BRIEF" --variant ltx25_t2v_i2v --no-interview
python -m master_agent diagnose --variant base --prompt "garden proof"
python -m master_agent mv render --audio track.mp3 --out out/MV-FIXED.mp4 --dry-run
```

## Install (Jason / Scott)

Source of truth in this repo: `video_buddy/skills/video-buddy/`.

```bash
# from video_buddy/
python install_hermes_skill.py
# copies skill → ~/.hermes/skills/video-buddy/  (skill only: no profile, no gateway config, no .env)
```

Windows / macOS / Linux all use `~/.hermes/skills/video-buddy/` (`HERMES_HOME` overrides `~/.hermes`). Confirm `SKILL.md` is in that folder. The LTX bot gateway's terminal must run from `video_buddy/` with the project venv. See [PROFILE.md](PROFILE.md) for the gateway setup and the one-time cleanup of the old MCP entry.

## Common mistakes

| Excuse | Reality |
|---|---|
| "I'll call the master-agent MCP tools" | Removed. Use `python -m master_agent agent <tool>`. |
| "I'll use Imagine / Grok video" | Wrong box for this local studio. |
| "I'll post to the studio on 8189 / A2A" | Removed. One gateway, one CLI entry. |
| "Queue length 8" | Illegal. LTX wants `8n+1`, min 9. |
| "Skip to overnight" | L0→L5 + human OK first. Diagnose before scale. |
| "Tracker says DONE" | Junk `<100KB` / `<3` frames is FAIL. |
| "It said busy, I'll start a second render" | One GPU. Wait for the first render. |
| "I'll add a diagnose agent tool" | Not registered. Use the CLI. |
