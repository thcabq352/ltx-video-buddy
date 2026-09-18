# Hermes profile `ltx`

Video Buddy seats itself as Hermes profile **`ltx`**. That is the primary
discovery path. A2A on studio `:8189` is the fallback for A2A-only peers.

## What the installer writes

`python install_hermes_skill.py` (or `python -m master_agent hermes register`):

- Skill folder `~/.hermes/skills/video-buddy/` (`SKILL.md`, `TOOLS.md`, this file)
- Profile dir `~/.hermes/profiles/ltx/`
  - `SOUL.md` — `LTX_RESEARCH_SYSTEM` (Ltx research seat)
  - `profile.yaml` — name + description
  - `config.yaml` — `terminal.cwd` + MCP `master-agent` (merged if the file exists)

Never written: `.env`, API keys, `API_SERVER_KEY`. Do not invent them.

SOUL refresh is merge-safe: a custom `SOUL.md` is left alone unless it still
equals the stock `LTX_RESEARCH_SYSTEM` text or you pass `--force`.

## Discovery order

1. Healthy real Hermes `ltx` gateway (standalone or mux `/p/ltx/` on **8642**).
2. Buddy facade on studio **8189**: `http://127.0.0.1:8189/p/ltx/v1/chat/completions`
   (`source=buddy-adapter`). Studio must be up (`python -m master_agent ui`).
3. A2A fallback: `GET /.well-known/agent.json` (also `agent-card.json`) and
   `POST /a2a` (`message/send`, `tasks/get`).

Buddy **never binds 8642**. That port belongs to the default Hermes API server.

## Pitch

The facade runs `hermes_pitch` only when the user explicitly asks for a pitch
(or `metadata.pitch=true`). A raw brief goes straight to the director pipeline.

## Check

```bash
python -m master_agent hermes status
python -m master_agent hermes register --hermes-home "$HERMES_HOME"
```
