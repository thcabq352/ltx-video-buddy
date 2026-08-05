---
name: video-buddy
description: "Use for Video Buddy local video agent (MCP master-agent)."
version: 1.0.1
author: Ara / Ringmaster
license: MIT
platforms: [windows]
metadata:
  hermes:
    tags: [video, comfyui, ltx, wan, mcp, video-buddy, master-agent]
    category: media
    related_skills: [ltx-director-hermes, route-to-specialist-profiles, native-mcp]
    homepage: https://github.com/thcabq352/ltx-video-buddy
---

# Video Buddy (Master Agent)

## Identity
- **Name:** Video Buddy
- **Code home:** `C:\Users\thcab\Documents\ltx2.3_agent\video_buddy`
- **Repo:** https://github.com/thcabq352/ltx-video-buddy (private)
- **Not** a Hermes profile. External friend next to hardline/forge/coder.
- **Hermes MCP server name:** `master-agent`
- **Studio UI:** `http://127.0.0.1:8189` (`python -m master_agent ui --port 8189` or `run_ui_8189.bat`)
- **Comfy:** portable `video_buddy/ComfyUI_windows_portable` → `run_api_8188.bat` → `http://127.0.0.1:8188`
- **Vault card:** `Documents/Obsidian Vault/Agentic OS/ops/agent-roster/video_buddy.md`

## When to use
- Autonomous local video: brief → storyboard → validate → render → judge → stitch
- LTX 2.3 / Wan 2.2 / lipsync / music video / fractal / Movie Builder packs in this tree
- Character → Flux sheet → LoRA (CCC)
- KB search over **this** studio's workflows/runs

## When NOT to use
| Signal | Route |
|---|---|
| Pixie Forge brand / scotty.fyi marketing / HyperFrames package | `hermes -p forge` |
| Grok cloud ≤15s clip | default Grok video skills |
| Ocala SMB / Agent OS | hardline |
| Only generic Comfy node poke on AraSM install | forge + comfyui MCP (confirm which Comfy is up) |

## Preflight (always)
```bash
curl -sS -m 3 http://127.0.0.1:8188/system_stats
curl -sS -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8189/
cd "C:/Users/thcab/Documents/ltx2.3_agent/video_buddy"
./.venv/Scripts/python.exe -m master_agent health
```
If Comfy down: start `ComfyUI_windows_portable\run_api_8188.bat` (leave window open).
VRAM: 16GB — don't stack Video Buddy heavy job + other Comfy + giant Ollama VL blindly.

## CLI (unattended)
```bash
cd "C:/Users/thcab/Documents/ltx2.3_agent/video_buddy"
PY=./.venv/Scripts/python.exe

$PY -m master_agent health
$PY -m master_agent run "BRIEF" --quality draft --duration 3 --no-interview
$PY -m master_agent run "BRIEF" --duration 8 --dry-run --no-interview
$PY -m master_agent run "talking head" --video input.mp4 --variant lipsync --no-interview
$PY -m master_agent music "synthwave MV" --audio track.mp3 --no-interview
$PY -m master_agent fractal "title" --duration 20 --target seahorse
$PY -m master_agent kb search "lipsync" -k 5
```
Long GPU jobs: terminal background + `notify_on_complete=true`. Deliverable under `outputs/`.

## Hermes MCP
Config (`~/.hermes/config.yaml` → `mcp_servers.master-agent`):
```yaml
master-agent:
  command: C:/Users/thcab/Documents/ltx2.3_agent/video_buddy/.venv/Scripts/python.exe
  args:
    - C:/Users/thcab/Documents/ltx2.3_agent/video_buddy/master_agent/mcp_server.py
  enabled: true
  timeout: 900
  connect_timeout: 120
```
Verify: `hermes mcp test master-agent` → 11 tools.
Tools: `health`, `create_video`, `plan_storyboard`, `judge_asset`, `search_workflows`, `search_runs`, `kb_ingest`, `list_models`, `validate_workflow`, `create_character`, `train_lora`.
After config change: restart gateway / new session so tools re-bind.

## Skills Hub wiring
- Canonical skill path in repo: `skills/video-buddy/SKILL.md`
- Tap: `hermes skills tap add thcabq352/ltx-video-buddy`
- Install: `hermes skills install thcabq352/ltx-video-buddy/video-buddy -y` (or hub identifier after index)
- Agents junction: `~/.agents/skills/video-buddy` → this directory
- Hermes local: `~/.hermes/skills/media/video-buddy` junction or hub install path

## Ringmaster vs forge
- **Video Buddy** = this agent's director stack + **its** portable Comfy + judge/KB
- **forge** = Hermes specialist profile (bots, Pixie, HyperFrames, often AraSM Comfy MCP)
Same machine GPU: pick **one** render owner per job.

## Pitfalls
1. Old path `…/ltx2.3_agent/kimi ltx/` is **gone** — never point MCP there.
2. UI `:8189` up ≠ Comfy `:8188` up.
3. `ltx-director-hermes` may still mention `LTX Project` :8765 — if that tree missing, use Video Buddy.
4. Intake interview default-on in interactive CLI — use `--no-interview` for unattended agent runs.
5. Private tap installs need GitHub auth (`gh` / credential helper).

## Verification
- [ ] `hermes skills tap list` includes `thcabq352/ltx-video-buddy`
- [ ] `hermes skills list` shows `video-buddy` enabled
- [ ] `hermes mcp test master-agent` → 11 tools
- [ ] `python -m master_agent health` Comfy up before GPU
- [ ] Output path exists under `video_buddy/outputs/` before claiming done
