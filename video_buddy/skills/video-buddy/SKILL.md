---
name: video-buddy
description: "Use when generating or judging local ComfyUI video on this machine with Video Buddy (LTX 2.5, MiniMax H3, WAN, lipsync, music video, fractal, storyboard, or the master-agent MCP). Use when an agent is about to treat Grok Imagine / cloud-only video as the local studio, or when MCP tools exist but Hermes has not loaded a real skill folder."
---

# Video Buddy

Local ComfyUI video studio in this repo (`video_buddy/`). Package is `master_agent`. MCP server id is **`master-agent`**. Not cloud Imagine. Primary seat is Hermes profile **`ltx`** (`hermes -p ltx`). A2A on studio `:8189` is the fallback.

**MCP ≠ skills.** Connecting `master-agent` exposes tools. Hermes only reliably uses Buddy when this folder is installed at `~/.hermes/skills/video-buddy/` (this `SKILL.md`). Tools without the skill = agents forget Buddy.

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
| Generic node poke on a **different** Comfy install | that install's MCP, not `master-agent` |
| A checkout other than this `video_buddy/` tree | that checkout's own docs — L0 is this package only |

## Hermes profile (primary)

1. Skill installed at `~/.hermes/skills/video-buddy/` (this file).
2. Profile `ltx` seated at `~/.hermes/profiles/ltx/` (`python install_hermes_skill.py` or `python -m master_agent hermes register`).
3. Server id `master-agent` present in the profile (and/or default) `config.yaml`.
4. Prefer `hermes -p ltx` or the studio facade `http://127.0.0.1:8189/p/ltx/v1/chat/completions`.
5. A2A-only peers: `GET /.well-known/agent.json` + `POST /a2a` on `:8189`.

Check: `python -m master_agent hermes status`. Buddy never binds **8642**.

## MCP must be running

1. Skill installed at `~/.hermes/skills/video-buddy/` (this file).
2. Server id `master-agent` present in `~/.hermes/config.yaml` (or `profiles/ltx/config.yaml`).
3. Hermes connected to that server (or start it: `<venv python> master_agent/mcp_server.py` from `video_buddy/`).

Skill without MCP → no tools. MCP without skill → tools exist, agents still forget Buddy. Need both.

Prefer ~900s tool timeout. `create_video` / `judge_asset` / `train_lora` are long.

**Do not invent tools.** The only MCP tools are the 11 below (from `master_agent/mcp_server.py`). Diagnose, curriculum, `comfy run`, doctor, budget, hermes, ui, about are **CLI-only**.

## Curriculum stop-lines (before overnight)

Do L0→L5 in order. Print the card: `python -m master_agent curriculum`. Part 2 overnight is gated on **L5 + human OK**.

| Lesson | Do |
|---|---|
| L0 tree | This tree: `video_buddy` / `master_agent`. |
| L1 about | `python -m master_agent about` — studio card before any GPU claim. |
| L2 health | `python -m master_agent health` — Comfy **:8188** up. Studio `:8189` is not proof. |
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
| **:8188** | ComfyUI API | `health` / MCP `health` shows Comfy up |
| **:8189** | Optional studio dashboard (`python -m master_agent ui`) | Human UI only. A live tab is **not** Comfy. |
| **:8080** | llama.cpp `llama-server` (preferred local LLM) | MCP `health.llamacpp` / `local_llm.llamacpp` |
| **:11434** | Ollama (second local LLM) | MCP `health.ollama` — not implied by llama.cpp |

Do not bind **8642** (Hermes API) or treat llama.cpp as Ollama.

## llama.cpp vs Ollama

llama.cpp is the preferred local backend. Buddy starts `llama-server` (`LLAMACPP_BIN` or `PATH`) against `MODELS_DIR` and stops it on shutdown. If that binary is missing, auto warns and uses Ollama. Pin `llamacpp` or `ollama` to force one backend.

| | llama.cpp | Ollama |
|---|---|---|
| Env | `LLAMACPP_URL`, `LLAMACPP_MODEL`, `LLAMACPP_BIN` | `OLLAMA_URL`, `OLLAMA_MODEL` |
| Provider | `LLM_PROVIDER=llamacpp` (aliases `llama.cpp`, `llama-cpp`) or `auto` | `LLM_PROVIDER=ollama` |
| Chat / storyboard | `{url}/v1/chat/completions` | `{url}/v1/chat/completions` |
| Embeddings | `/v1/embeddings` (KB no-ops if missing) | `/api/embed` |
| Vision judge | multimodal `/v1/chat/completions`; heuristic-only if the GGUF is text-only | `/api/chat` + images |

`auto` order: llamacpp → ollama → grok. Panels accept `llamacpp[:model]`.
`health` reports llama.cpp, Ollama, and Grok separately.

## MCP tools ↔ CLI

Full signatures: [TOOLS.md](TOOLS.md). Invoke MCP as `master-agent.<tool>`. CLI from `video_buddy/` with the project venv: `python -m master_agent …`.

| MCP (`master-agent`) | CLI equivalent |
|---|---|
| `health()` | `python -m master_agent health` |
| `create_video(request, duration_s=5, quality="draft", variant=None, llm_panel=None, dry_run=False)` | `python -m master_agent run "BRIEF" [--duration N --quality draft --variant SLUG --llm-panel … --dry-run --self-improve-dry --no-interview]` |
| `plan_storyboard(request, duration_s=8, quality="draft", llm_panel=None)` | No GPU plan preview. Closest: `run "BRIEF" --dry-run` (also lints). |
| `judge_asset(video_path, request="", full_video=False)` | No standalone judge CLI. Judge runs inside `run` / `music`. |
| `search_workflows(query, k=3)` | `python -m master_agent kb search "QUERY" --workflows -k 3` |
| `search_runs(query, k=3)` | `python -m master_agent kb search "QUERY" -k 3` |
| `kb_ingest()` | `python -m master_agent kb ingest` |
| `list_models()` | `python -m master_agent scan-models` |
| `validate_workflow(path)` | `python -m master_agent validate path.json` |
| `create_character(description, name="", shots=0, train=False)` | `python -m master_agent character create "DESC" [--name N --shots N --train]` |
| `train_lora(character_name, steps=0, lr=0, rank=0, validate=True)` | `python -m master_agent lora train NAME [--steps N --lr X --rank N --validate]` |

CLI-only (no MCP tool): `about`, `curriculum`, `inventory`, `doctor`/`setup`, `workflows`, `capabilities`, `download-models`, `download-flux`, `comfy run`, `diagnose`, `budget`, `hermes`, `ui`, `fetch-object-info`, `power-tune`, `persona`, `soul`, `brief`, `fractal`, `music`, `mv plan`/`mv render`, `heartmula generate`/`heartmula transcribe`, `lora setup`/`validate`, `character list`.

Drive graphs with CLI first: `comfy run`. Director pipeline: `run`. Unattended: `--no-interview`.

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
# copies skill → ~/.hermes/skills/video-buddy/
# seats profile ltx → ~/.hermes/profiles/ltx/  (no .env, no secrets)
# or: python -m master_agent hermes register
```

Windows / macOS / Linux all use `~/.hermes/skills/video-buddy/` (`HERMES_HOME` overrides `~/.hermes`). Confirm `SKILL.md` is in that folder. Confirm `profiles/ltx/SOUL.md` exists. A custom SOUL is not overwritten unless `--force`.

Confirm MCP in `~/.hermes/config.yaml` (fragment, **no secrets**):

```yaml
mcp_servers:
  master-agent:
    command: "<VIDEO_BUDDY>/.venv/bin/python"   # Windows: .venv\Scripts\python.exe
    args:
      - "<VIDEO_BUDDY>/master_agent/mcp_server.py"
```

`<VIDEO_BUDDY>` is this repo's `video_buddy/` directory. Exact YAML key may be whatever `hermes mcp add` writes — the **server id must be `master-agent`**. Prefer:

```bash
hermes mcp add master-agent -- <venv-python> <VIDEO_BUDDY>/master_agent/mcp_server.py
hermes mcp test master-agent
```

Hermes usually spawns the stdio server. To start by hand (cwd = `video_buddy/`):

```bash
.venv/bin/python master_agent/mcp_server.py          # macOS / Linux
.venv\Scripts\python.exe master_agent\mcp_server.py  # Windows
```

## Common mistakes

| Excuse | Reality |
|---|---|
| "MCP is connected, I know the tools" | Skill folder is what Hermes loads. Install it. |
| "I'll use Imagine / Grok video" | Wrong box for this local studio. |
| "Studio tab is up" | That is `:8189`. Comfy is `:8188`. `health` first. |
| "Queue length 8" | Illegal. LTX wants `8n+1`, min 9. |
| "Skip to overnight" | L0→L5 + human OK first. Diagnose before scale. |
| "Tracker says DONE" | Junk `<100KB` / `<3` frames is FAIL. |
| "I'll click Queue in the studio" | Agents use `comfy run` / `run`. UI is for humans. |
| "I'll add a diagnose MCP tool" | Not registered. Use the CLI. |
