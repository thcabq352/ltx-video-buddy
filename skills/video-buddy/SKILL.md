---
name: video-buddy
description: "Use when generating or judging local ComfyUI video on this machine with Video Buddy (LTX 2.5, MiniMax H3, WAN, lipsync, music video, fractal, storyboard, or the master-agent MCP). Use when an agent is about to treat Grok Imagine / cloud-only video as the local studio, or when MCP tools exist but Hermes has not loaded a real skill folder."
---

# Video Buddy (tap pointer)

**Canonical skill (install this):** [`video_buddy/skills/video-buddy/`](../../video_buddy/skills/video-buddy/SKILL.md)

This repo-root `skills/` entry is a Hermes tap pointer only. MCP tools without
that skill folder = agents forget Buddy.

```bash
cd video_buddy
python install_hermes_skill.py
# → ~/.hermes/skills/video-buddy/
```

Confirm `master-agent` in `~/.hermes/config.yaml`, then start
`<venv python> master_agent/mcp_server.py` from `video_buddy/`.
