# Skills (Hermes / agentskills)

Canonical agent skill is shipped under **`video_buddy/skills/video-buddy/`**
(install source of truth). Copy that folder to `~/.hermes/skills/video-buddy/`
or run `python video_buddy/install_hermes_skill.py`.

| Skill | Purpose |
|---|---|
| `video-buddy` | How Hermes calls this studio (CLI + MCP `master-agent`). Curriculum stop-lines, ports, tool↔CLI map. LTX 2.5 + MiniMax H3 + WAN. |

Repo-root `skills/video-buddy/` is a tap pointer only — do not treat it as the
full skill. Hub: `hermes skills tap add thcabq352/ltx-video-buddy`
