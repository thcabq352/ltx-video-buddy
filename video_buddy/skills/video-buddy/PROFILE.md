# LTX bot gateway

Video Buddy has exactly one gateway: the **LTX bot gateway**, the Hermes
gateway that serves the LTX bot (Hermes profile `ltx`). Buddy does not host,
register, discover, or probe it. The gateway calls Buddy; Buddy never listens.

## What the gateway needs

- This skill in the `ltx` profile. From `video_buddy/`:

  ```bash
  python install_hermes_skill.py --profile ltx
  # → ~/.hermes/profiles/ltx/skills/media/video-buddy/
  ```

  `--dest DIR` installs into any other folder; no flag installs into
  `~/.hermes/skills/video-buddy/`. `HERMES_HOME` overrides `~/.hermes`.
- A terminal whose cwd is this repo's `video_buddy/` and whose Python is the
  project venv.
- Nothing else: no server entry, no API key, no `.env` from Buddy.

Suggested system prompt for the bot (optional, set by the operator):

> You are Ltx, the LTX research bot. Specialize in LTX Video on the Video
> Buddy portable ComfyUI. Stay short. Never print tokens, API keys, or bearers.
> Start ComfyUI only with `python -m master_agent agent comfy_start`; never run
> ComfyUI main.py or run_*.bat in a foreground terminal.

## Calls

```bash
python -m master_agent agent list
python -m master_agent agent comfy_status
python -m master_agent agent comfy_start
python -m master_agent agent create_video --args '{"request": "BRIEF", "dry_run": true}'
```

`comfy_start` is detached and returns within `timeout_s` (default 180 s), well
inside the gateway's 420 s tool limit. `starting` means poll `comfy_status`;
`error` carries the launch log tail. Do not repeat a failed start unchanged.
Exit codes, `busy`, `paused` and media-path rules:
[SKILL.md](SKILL.md#calling-buddy).

## Old profile configs (operator step)

If `~/.hermes/profiles/ltx/config.yaml` (or `~/.hermes/config.yaml`) still has
a `master-agent` entry under its server list from an older install, delete
that entry. The program it pointed at no longer exists, so the entry only
produces startup errors. Keep `terminal.cwd`. Buddy does not edit Hermes
config, so it will not do this for you.

Buddy **never binds 8642**. That port belongs to the default Hermes API server.
