---
name: video-buddy
description: "Use when generating or judging local ComfyUI video on this machine with Video Buddy (LTX 2.5, MiniMax H3, WAN, lipsync, music video, fractal, storyboard) through the LTX bot gateway and `python -m master_agent agent`. Use when an agent is about to treat Grok Imagine / cloud-only video as the local studio."
---

# Video Buddy (tap pointer)

**Canonical skill (install this):** [`video_buddy/skills/video-buddy/`](../../video_buddy/skills/video-buddy/SKILL.md)

This repo-root `skills/` entry is a Hermes tap pointer only.

```bash
cd video_buddy
python install_hermes_skill.py
# → ~/.hermes/skills/video-buddy/  (skill only)
python -m master_agent agent list
```
