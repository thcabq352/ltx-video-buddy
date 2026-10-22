# Architecture

Video Buddy is the local operator in front of ComfyUI. Package `master_agent`. One gateway: the LTX bot gateway. No cloud render farm is required. Cloud LLMs are optional panel members.

```
you ──▶ persona intake ──▶ creative brief
                             │
                    director LLM ──▶ variant ranking
                             │
                    Hands.can_fulfill (live snapshot)
                             │
                    storyboard panel (optional)
                             │
              patch → validate → submit → poll → judge
                             │
                     ComfyUI :8188
                             │
                     stitch → full-video judge → upscale
```

Nothing is queued until validation passes. CUDA OOM walks `DOWNSCALE_LADDER` (and `vram_policy.downscale_ladder_for`) instead of crashing the process.

## Intake, persona, soul

Default persona is **Ara**. Default soul is **studio** (`play` is the other bundled soul). Personas: `ara`, `exec`, `zod` in `master_agent/persona/personas/`. Override with `state/personas/<slug>.md` or `state/souls/<slug>.md`.

`run`, `music`, and `fractal` interview first unless `--no-interview`, `INTAKE_ENABLED=0`, a non-TTY, or the user types `just go` / `skip`. `brief` is interview-only; `--go` then generates. Records land in `state/runs/*_intake.json`.

## Brain and Hands

Schema `buddy.capability.contract/v1`. Code: `master_agent/capability.py`, `master_agent/hands.py`.

The director ranks stories across every `workflows/manifests.yaml` slug (`WORKFLOW_FILES`). Hands answers whether the live tower can fulfill that story now (`FitResult`: `ok`, `insufficient_vram`, `missing_weights`, `missing_nodes`). Keyword rules are the fallback (`DIRECTOR_LLM=0`). `--variant` forces. A source video implies lipsync.

The contract carries model, family, variant, resolution, `duration_s`, `story_duration_s`, audio, and `control_layers`. It does not carry VRAM, slot, or a weight path. The director does not pick a cheaper-GPU graph from the pack table. Heavy graphs stay available when the story names them. Hands rejects when the snapshot cannot fulfill.

Long stories: `plan_last_frame_chain` (20s → 3×8s). Music-video beat windows stay on the MTV path and do not use that chain.

```json
{
  "schema": "buddy.capability.contract/v1",
  "model": "ltx-2.5",
  "family": "ltx25",
  "variant": "ltx25_t2v_i2v",
  "resolution": [768, 512],
  "duration_s": 8.0,
  "story_duration_s": 20.0,
  "audio": true,
  "control_layers": ["openpose"]
}
```

Tests: `python -m pytest tests/test_brain_hands.py -q` from `video_buddy/`.

## State machine and pipeline

`run` states: `SELECT_VARIANT → PATCH → VALIDATE → SUBMIT → POLL → RESOLVE → JUDGE → DONE/ERROR`.

Requests longer than one Hands clip go through `orchestrator/pipeline.py`: storyboard, per-segment orchestrator (`seed = base + i*17`), ffmpeg stitch, full-video judge, weak-shot regen up to `--max-full-judge-rounds` (default 2). Pipeline JSON is `state/runs/<ts>_<id>_pipeline.json`.

Storyboard modes: `smart` (default), `always`, `multi_only`, `off`. Panel presets (`--llm-panel` / `LLM_PANEL`): `local` (default), `grok`, `grok+local`, `grok+claude`. `--panel-judge` (default `ollama`) picks among multi-member panels. Unavailable members are skipped.

Power mode (`--power-mode` or `POWER_MODE=1`) lets the LLM propose graph ops after the heuristic patch. Ops are validate-gated. `power-tune` is the dry-run. Ops: `set_widget`, `set_widget_by_class`, `rewire`, `add_node`, `remove_node`, `delete_input`.

Judge, quality bar, and provenance: [Judge](JUDGE.md), [Provenance](PROVENANCE.md).

## Local model

llama.cpp is the preferred local backend. Buddy starts `llama-server` on the first LLM call when `LLM_PROVIDER` is `auto` (the default) or `llamacpp`. The process is `LLAMACPP_BIN` or `llama-server` on `PATH`, bound to `LLAMACPP_URL` (default `:8080`), with `--models-dir` set to `MODELS_DIR` (default `video_buddy/models/`). Automatic install downloads the public GGUFs into `MODELS_DIR/<model-name>/` (`qwen3-vl-heretic/`, `nomic-embed-text/`) unless a vision pair is already under `MODELS_DIR` (checked at `gguf_vl_heretic/` first) or an Ollama `nomic-embed-text` blob can be referenced read-only. When those files are on disk, it writes `state/llamacpp-models.ini` (with `mmproj` when a projector is in use) and passes it as `--models-preset`. Buddy stops that process on shutdown. A server that was already listening is left alone. `LLAMACPP_AUTOSTART=0` turns off the start; Buddy then only uses a server that is already listening. If the binary is missing, auto warns and continues to Ollama. It does not crash.

`LLM_PROVIDER=auto` (and an unset provider) tries llama.cpp, then Ollama, and stops there. Grok (cloud) is opt-in only: `LLM_PROVIDER=grok` or a panel that names it. Pin `llamacpp` or `ollama` to force that backend. A pin does not hop to the other local server.

Default vision and text model name is `qwen3-vl-heretic` (`LLAMACPP_MODEL` / `OLLAMA_MODEL`). That model is not the LTX text encoder. Do not swap it onto a LoRA A/B graph.

| | llama.cpp | Ollama | Grok |
|---|---|---|---|
| Role | preferred local | second local | last resort |
| URL | `LLAMACPP_URL` default `:8080` | `OLLAMA_URL` default `:11434` | xAI API |
| Model | `LLAMACPP_MODEL` | `OLLAMA_MODEL` | `SPACEXAI_MODEL` |
| Spec | `llamacpp` / `llama.cpp` / `llamacpp[:model]` | `ollama[:model]` | `grok` |
| Embeddings | `POST /v1/embeddings` | `POST /api/embed` | — |
| Vision | multimodal `/v1/chat/completions` | `POST /api/chat` + images | — |

Health reports llama.cpp and Ollama reachability, and whether opt-in Grok credentials exist (read-only; health never refreshes tokens or writes `~/.hermes/auth.json`). The active backend follows the order above. Do not bind 8642 for llama.cpp. If embeddings or vision are missing, KB calls no-op with a warning and the judge stays heuristic-only.

## Knowledge

ChromaDB at `video_buddy/state/chroma/` (gitignored) holds `workflows`, `runs`, and `knowledge`. The git folder [`knowledge/`](../knowledge/) is what the other machine can pull. Schema files and `EXAMPLE` entries are not indexed. `kb ingest` embeds recorded entries. Recall prefers `failures/` (`priority=high`).

## Ports

| Port | Role |
|---|---|
| 8188 | ComfyUI HTTP |
| 11434 | Ollama |
| 8080 | llama.cpp |
| 8642 | Hermes. Buddy does not bind it. |
| 8501 | Optional Streamlit dashboard, only while `streamlit run streamlit_app.py` runs ([Browser dashboard](GETTING_STARTED.md#browser-dashboard)). Bind it to 127.0.0.1. |

Shift budget is about 80 VRAM-minutes. Over cap, the queue HOLDs and the run reports `paused`, not `error`. Diagnose and dry-run do not increment `used`.

`health`, `agent health` (`budget` key), and `doctor` (`budget` row) print one read-only line from the ledger (`state/control/budget.json`):

```text
budget: 34.2 / 80.0 VRAM-min used, 45.8 remaining; paused runs: 2 (run_ab12, run_cd34)
```

The current shift is the ledger's `shift_id`: `used` counts since `shift_started_at`, and `budget reset-shift` starts a new one. Paused runs are run ids whose newest `*_pipeline.json` is `paused`, the same set `reset-shift` continues. `(HOLD)` follows `remaining` while the queue is held. A missing or empty ledger reads as 0 used. The JSON has `used`, `cap`, `remaining`, `paused`, `paused_run_ids`, `shift_id`, `shift_started_at`, `ledger` (`ok` / `missing` / `empty` / `unreadable`), and `line`.

## One gateway

The **LTX bot gateway** is the only agent entry point. It loads the
`video-buddy` skill and runs `python -m master_agent agent <tool> --args JSON`
from `video_buddy/` (or imports `master_agent.agent_api`). Buddy listens on no
port of its own. The optional Streamlit dashboard is for people, not agents: it
calls the same `agent_api` functions in-process and is not a gateway.

"LTX bot gateway" means the Hermes gateway that serves the LTX bot (profile
`ltx`, standalone or mux `/p/ltx/` on 8642), driving Buddy through its
terminal. Status 2026-10-19: confirmed in configuration, not yet in production.
The tower's Hermes configs dropped the `master-agent` MCP entry on 2026-10-18,
so the bot reaches Buddy only through `agent`. Until that path has run in
production for two weeks, a dormant MCP fallback exists (see
[Revert / fallback path](#revert--fallback-path)).

Removed in [PR #81](https://github.com/thcabq352/ltx-video-buddy/pull/81)
(merge `2b03703`): the `master-agent` MCP server (`master_agent/mcp_server.py`,
`1ada3a9`), the Hermes `ltx` seat machinery and profile writers
(`master_agent/hermes/`, `hermes status|register`, `4b3628b`), the studio on
`:8189` (UI, Comfy-tab ingest drop zone, job manager, `ui`, `run_ui_8189.bat`,
`1cb01be`), the Hermes chat facade and A2A (`master_agent/a2a/`, `7997511`).
Docs and skill followed in `c67e909`.

Concurrency: `agent_api.create_video` holds `state/gpu.lock` for a render and
returns `status=busy` when another render has it. A budget HOLD returns
`status=paused`; `control_set(reset_budget=true)` or `budget_reset_shift`
resumes paused runs in-process.

## Revert / fallback path

**Temporary.** `python -m master_agent fallback-mcp` (`master_agent/fallback_mcp.py`)
is an MCP stdio server with id `master-agent` whose tools are exactly
`agent_api.TOOLS`: same names, signatures, and docstrings, each call forwarded
to the same function. It has no tool logic of its own. It is dormant: nothing
starts or imports it, it needs the optional `mcp` package, and it refuses to
serve unless `BUDDY_FALLBACK_MCP=1`. It binds no port; Hermes owns the process.
The skill package does not mention it, so the bot keeps using `agent` unless an
operator wires the fallback in.

MCP stdio was chosen over an HTTP surface on `:8189` because it opens no
network listener (no Host/Origin guard to rebuild for an API that can queue
GPU renders), needs no web framework, and is the config Hermes used before
PR #81, so the snippet below is the known-good shape.

### Enable it

From `video_buddy/`, with the project venv:

```bash
pip install -r requirements-fallback-mcp.txt
BUDDY_FALLBACK_MCP=1 python -m master_agent fallback-mcp   # smoke: waits on stdin; Ctrl-C
```

Without the flag it prints why it is off and exits 2.

### Point Hermes at it

In `~/.hermes/profiles/ltx/config.yaml` (or `~/.hermes/config.yaml`), no
secrets. `<VIDEO_BUDDY>` is this repo's `video_buddy/` directory, as an
absolute path:

```yaml
mcp_servers:
  master-agent:
    command: "<VIDEO_BUDDY>/.venv/bin/python"   # Windows: <VIDEO_BUDDY>\.venv\Scripts\python.exe
    args: ["-m", "master_agent", "fallback-mcp"]
    env:
      BUDDY_FALLBACK_MCP: "1"
      PYTHONPATH: "<VIDEO_BUDDY>"
      LLAMACPP_AUTOSTART: "0"   # reuse a running llama-server; do not spawn one per MCP child
```

Then restart that gateway and run `hermes mcp test master-agent`. Buddy reads
its own settings from `video_buddy/.env`. An MCP child may not inherit the
gateway's shell environment, so put anything Buddy needs in that file or in
`env:` above. To turn the fallback off, delete the `master-agent` block and
restart the gateway.

### Full revert to the pre-PR #81 surfaces

Use this only if the shim is not enough. Work on a branch. Revert the newest
commit first:

```bash
git revert 1ada3a9            # MCP server only: applies cleanly on 133cfc7 and the suite passes
git revert 1cb01be            # :8189 studio
git revert 4b3628b            # Hermes ltx seat machinery: conflicts in install_hermes_skill.py and tests/test_hermes_skill.py
git revert 7997511            # A2A + Hermes chat facade
git revert c67e909            # docs and skill: conflicts across README, docs/, and the skill files
pip install -r requirements.txt
```

The restored `mcp_server.py` exposes the 11 tools it had on 2026-10-09. It
does not have `comfy_status`, `comfy_start`, `comfy_restart`, or later agent
tools, which is one reason to prefer the shim. Remove the shim (below) when
reverting `1ada3a9`, because both register as `master-agent`. Restore the
pre-PR #81 Hermes config (script path
`<VIDEO_BUDDY>/master_agent/mcp_server.py`, no flag).

### Remove the shim

Remove after **2026-11-02**, when both hold: the Hermes `ltx` bot has served
real requests through `python -m master_agent agent` for 14 consecutive days,
and no gateway config in that window enabled `BUDDY_FALLBACK_MCP`. If the
fallback was turned on, the 14 days restart after it is turned off and the
cause is fixed.

Delete `video_buddy/master_agent/fallback_mcp.py`,
`video_buddy/requirements-fallback-mcp.txt`, and
`video_buddy/tests/test_fallback_mcp.py`, and the `requirements-fallback-mcp.txt`
line in `.github/workflows/pytest.yml`. Drop `cmd_fallback_mcp` and the
`fallback-mcp` parser entry (and its `banner_stream` case) from
`master_agent/cli/parser.py`, and the line in the `master_agent/__main__.py`
docstring. Drop this section and the fallback sentence under
[One gateway](#one-gateway). `pip uninstall mcp` on the tower.
