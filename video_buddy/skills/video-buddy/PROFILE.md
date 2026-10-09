# LTX bot gateway

Video Buddy has exactly one gateway: the **LTX bot gateway**, the Hermes
gateway that serves the LTX bot (historically profile `ltx`, standalone or
mux `/p/ltx/` on **8642**). Buddy does not host, register, discover, or
probe it. The gateway calls Buddy; Buddy never listens.

## What the gateway needs

- This skill at `~/.hermes/skills/video-buddy/` (`python install_hermes_skill.py`).
- A terminal whose cwd is this repo's `video_buddy/` and whose Python is the
  project venv.
- Nothing else: no MCP server entry, no API key, no `.env` from Buddy.

Suggested system prompt for the bot (optional, set by the operator):

> You are Ltx, the LTX research bot. Specialize in LTX Video on the Video
> Buddy portable ComfyUI. Stay short. Never print tokens, API keys, or bearers.

## Calls

```bash
python -m master_agent agent list
python -m master_agent agent create_video --args '{"request": "BRIEF", "dry_run": true}'
```

Exit codes, `busy`, `paused` and media-path rules: [SKILL.md](SKILL.md#calling-buddy).

## One-time cleanup on an existing install (operator step)

Older installs wrote `~/.hermes/profiles/ltx/config.yaml` with an
`mcp_servers.master-agent` entry pointing at `master_agent/mcp_server.py`.
That file no longer exists. Before deploying this version, delete the
`master-agent` entry under `mcp_servers` in that profile (and in
`~/.hermes/config.yaml` if it was added there). Keep `terminal.cwd`.
Buddy no longer edits Hermes config, so it will not do this for you.

Removed, with no replacement port: the studio dashboard and Comfy-tab drop
zone on **8189**, the `/p/ltx/v1/chat/completions` facade, A2A
(`/.well-known/agent.json`, `POST /a2a`), `python -m master_agent hermes
status|register`, and `python -m master_agent ui`.

Buddy **never binds 8642**. That port belongs to the default Hermes API server.
