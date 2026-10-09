# Skills (Hermes / agentskills)

Canonical agent skill is shipped under **`video_buddy/skills/video-buddy/`**
(install source of truth). Copy that folder to `~/.hermes/skills/video-buddy/`
or run `python video_buddy/install_hermes_skill.py` (also seats profile `ltx`).

| Skill | Purpose |
|---|---|
| `video-buddy` | How the LTX bot gateway calls this studio (`python -m master_agent agent <tool>` + CLI). One gateway; no MCP, A2A or studio port. Curriculum stop-lines, ports, tool↔CLI map. LTX 2.5 + MiniMax H3 + WAN. |

Repo-root `skills/video-buddy/` is a tap pointer only — do not treat it as the
full skill. Hub: `hermes skills tap add thcabq352/ltx-video-buddy`
