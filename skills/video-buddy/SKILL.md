---
name: video-buddy
description: "Use for Video Buddy local video agent (MCP master-agent)."
version: 1.0.2
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
- **MCP server name:** `master-agent` (Hermes native MCP client; register once via `hermes mcp`)
- **Studio UI:** loopback port **8189** — `python -m master_agent ui --port 8189` or `run_ui_8189.bat`
- **Comfy API:** portable under `video_buddy/ComfyUI_windows_portable` — `run_api_8188.bat` — loopback port **8188**
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
cd "C:/Users/thcab/Documents/ltx2.3_agent/video_buddy"
./.venv/Scripts/python.exe -m master_agent health
# optional: studio dashboard
# ./.venv/Scripts/python.exe -m master_agent ui --port 8189
```
If Comfy is down: start `ComfyUI_windows_portable\run_api_8188.bat` (leave window open).
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
Server id: **`master-agent`**.

Command (stdio):
- Python: `C:/Users/thcab/Documents/ltx2.3_agent/video_buddy/.venv/Scripts/python.exe`
- Script: `C:/Users/thcab/Documents/ltx2.3_agent/video_buddy/master_agent/mcp_server.py`
- Prefer long tool timeout (≈900s) and connect timeout (≈120s)

Register/update with Hermes MCP tooling (`hermes mcp add` / `hermes mcp list` / `hermes mcp test master-agent`) — do **not** hand-edit Hermes config files from this skill.

Verify: `hermes mcp test master-agent` → 11 tools:
`health`, `create_video`, `plan_storyboard`, `judge_asset`, `search_workflows`, `search_runs`, `kb_ingest`, `list_models`, `validate_workflow`, `create_character`, `train_lora`.

After MCP registration changes: new session / gateway refresh so tools re-bind.

## Skills Hub wiring
- Canonical path in repo: `skills/video-buddy/SKILL.md`
- Tap: `hermes skills tap add thcabq352/ltx-video-buddy`
- Install from private raw URL (via `gh api …/contents/… --jq .download_url`) or keep local/agents copies
- Agents path: `~/.agents/skills/video-buddy`
- Hermes path: `~/.hermes/skills/media/video-buddy`

## Ringmaster vs forge
- **Video Buddy** = this agent's director stack + **its** portable Comfy + judge/KB
- **forge** = Hermes specialist profile (bots, Pixie, HyperFrames, often AraSM Comfy MCP)
Same machine GPU: pick **one** render owner per job.

## Pitfalls
1. Old path `…/ltx2.3_agent/kimi ltx/` is **gone** — never point MCP there.
2. Studio UI up ≠ Comfy API up.
3. `ltx-director-hermes` may still mention `LTX Project` A2A harness — if that tree is missing, use Video Buddy.
4. Intake interview default-on in interactive CLI — use `--no-interview` for unattended agent runs.
5. Private hub installs need GitHub auth; community scan may flag localhost ports — prefer local skill + tap for discovery.

## Verification
- [ ] `hermes skills tap list` includes `thcabq352/ltx-video-buddy`
- [ ] `hermes skills list` shows `video-buddy` enabled
- [ ] `hermes mcp test master-agent` → 11 tools
- [ ] `python -m master_agent health` reports Comfy up before GPU work
- [ ] Output path exists under `video_buddy/outputs/` before claiming done
