---
name: video-buddy
description: "Use when generating or judging local ComfyUI video on this machine with Video Buddy (LTX 2.5, MiniMax H3, WAN, lipsync, music video, fractal, storyboard) through the LTX bot gateway and `python -m master_agent agent`. Use when ComfyUI needs starting (always `agent comfy_start`, never a foreground main.py). Use when an agent is about to treat Grok Imagine / cloud-only video as the local studio."
---

# Video Buddy (tap pointer)

**Canonical skill (install this):** [`video_buddy/skills/video-buddy/`](../../video_buddy/skills/video-buddy/SKILL.md)

This repo-root `skills/` entry is a Hermes tap pointer only.

```bash
cd video_buddy
python install_hermes_skill.py --profile ltx
# → ~/.hermes/profiles/ltx/skills/media/video-buddy/  (SKILL.md, TOOLS.md, PROFILE.md, references/)
python -m master_agent agent list
```

**Starting ComfyUI:** only `python -m master_agent agent comfy_start` (or `python -m master_agent comfy start`). It runs ComfyUI detached and returns when `:8188` answers or the launch fails. NEVER run ComfyUI `main.py` or `run_*.bat` in a foreground terminal: it blocks until the tool limit kills it. If a raw launch is unavoidable, use `background=true` and poll `agent comfy_status` or `GET http://127.0.0.1:8188/system_stats`. After a failed start, read the returned `launch.log_tail` and do not retry the identical command.
