# Architecture

Video Buddy is the local operator in front of ComfyUI. Package `master_agent`. MCP id `master-agent`. No cloud render farm is required. Cloud LLMs are optional panel members.

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

llama.cpp is the preferred local backend. Buddy starts `llama-server` for the studio, the MCP server, and LLM calls when `LLM_PROVIDER` is `auto` (the default) or `llamacpp`. The process is `LLAMACPP_BIN` or `llama-server` on `PATH`, bound to `LLAMACPP_URL` (default `:8080`), with `--models-dir` set to `MODELS_DIR` (default `video_buddy/models/`). Buddy stops that process on shutdown. A server that was already listening is left alone. If the binary is missing, auto warns and continues to Ollama. It does not crash.

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

Health reports llama.cpp and Ollama reachability, and whether opt-in Grok credentials exist (read-only; health never refreshes tokens or writes `~/.hermes/auth.json`). The active backend follows the order above. Do not bind 8642 or 8189 for llama.cpp. If embeddings or vision are missing, KB calls no-op with a warning and the judge stays heuristic-only.

## Knowledge

ChromaDB at `video_buddy/state/chroma/` (gitignored) holds `workflows`, `runs`, and `knowledge`. The git folder [`knowledge/`](../knowledge/) is what the other machine can pull. Schema files and `EXAMPLE` entries are not indexed. `kb ingest` embeds recorded entries. Recall prefers `failures/` (`priority=high`).

## Ports

| Port | Role |
|---|---|
| 8188 | ComfyUI HTTP |
| 8189 | Studio UI, Hermes facade `POST /p/ltx/v1/chat/completions`, A2A |
| 11434 | Ollama |
| 8080 | llama.cpp |
| 8642 | Hermes. Buddy does not bind it. |

Shift budget is about 80 VRAM-minutes. Over cap, the queue HOLDs. HOLD is A2A `input-required`, not `failed`. `done_with_warnings` maps to `completed`. Diagnose and dry-run do not increment `used`.
