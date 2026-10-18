# Skills (Hermes / agentskills)

Canonical agent skill is shipped under **`video_buddy/skills/video-buddy/`**
(install source of truth). Install it with `python video_buddy/install_hermes_skill.py`:
`--profile ltx` puts it in the LTX bot's profile
(`~/.hermes/profiles/ltx/skills/media/video-buddy/`), no flag puts it in
`~/.hermes/skills/video-buddy/`, `--dest DIR` anywhere else. It copies the skill
files only.

| Skill | Purpose |
|---|---|
| `video-buddy` | How the LTX bot gateway calls this studio (`python -m master_agent agent <tool>` + CLI). Starting ComfyUI only through `agent comfy_start` (detached), never a foreground `main.py`. Curriculum stop-lines, ports, tool↔CLI map. LTX 2.5 + MiniMax H3 + WAN. |

Repo-root `skills/video-buddy/` is a tap pointer only — do not treat it as the
full skill. Hub: `hermes skills tap add thcabq352/ltx-video-buddy`
