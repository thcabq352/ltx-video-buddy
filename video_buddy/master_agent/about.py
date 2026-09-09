"""Studio identity card — CLI `about`, GET /api/about, and the About tab."""

from __future__ import annotations

from typing import Any

from master_agent import __version__

STUDIO_PORT = 8189
REPO = "https://github.com/thcabq352/ltx-video-buddy"
DRIVE_GRAPH = (
    'python -m master_agent comfy run --mode generate --prompt "BRIEF" --variant base'
)
DRIVE_DIRECTOR = (
    'python -m master_agent run "BRIEF" --quality draft --duration 3 --no-interview'
)
DRIVE_PREPARE = (
    "python -m master_agent comfy run --mode template --template base --prepare"
)


def studio_about() -> dict[str, Any]:
    import master_agent.config as cfg

    persona = {"slug": cfg.PERSONA, "name": cfg.PERSONA}
    soul = {"slug": cfg.SOUL, "name": cfg.SOUL}
    try:
        from master_agent.persona.persona import load_persona

        p = load_persona()
        persona = {"slug": p.slug, "name": p.name}
    except Exception:
        pass
    try:
        from master_agent.persona.soul import load_soul

        s = load_soul()
        soul = {"slug": s.slug, "name": s.name}
    except Exception:
        pass
    return {
        "name": "VIDEO BUDDY",
        "tagline": "Local ComfyUI video studio. Drive Comfy from the CLI first.",
        "version": __version__,
        "package": "master_agent",
        "mcp_id": "master-agent",
        "repo": REPO,
        "comfy": {"url": cfg.COMFYUI_URL, "port": cfg.COMFYUI_PORT},
        "studio": {"url": f"http://127.0.0.1:{STUDIO_PORT}", "port": STUDIO_PORT},
        "drive": {
            "first": "cli",
            "health": "python -m master_agent health",
            "graph": DRIVE_GRAPH,
            "director": DRIVE_DIRECTOR,
            "prepare": DRIVE_PREPARE,
        },
        "identity": {"persona": persona, "soul": soul},
        "models": {
            "ollama": cfg.OLLAMA_MODEL,
            "vision": cfg.VISION_MODEL,
            "llm_provider": cfg.LLM_PROVIDER,
        },
        "personas": ["ara", "exec", "zod"],
        "souls": ["studio", "play"],
    }


def format_about(card: dict[str, Any] | None = None) -> str:
    data = card or studio_about()
    ident = data.get("identity") or {}
    persona = ident.get("persona") or {}
    soul = ident.get("soul") or {}
    models = data.get("models") or {}
    drive = data.get("drive") or {}
    comfy = data.get("comfy") or {}
    studio = data.get("studio") or {}
    lines = [
        f"{data.get('name', 'VIDEO BUDDY')}  v{data.get('version', '')}",
        data.get("tagline") or "",
        "",
        f"Comfy    {comfy.get('url')}",
        f"Studio   {studio.get('url')}  (optional dashboard)",
        f"Persona  {persona.get('slug')} ({persona.get('name')})   soul {soul.get('slug')} ({soul.get('name')})",
        f"Local    {models.get('ollama')}   provider {models.get('llm_provider')}",
        f"Package  {data.get('package')}   MCP {data.get('mcp_id')}",
        "",
        "Drive a graph:",
        f"  {drive.get('health')}",
        f"  {drive.get('graph')}",
        "",
        "Director pipeline:",
        f"  {drive.get('director')}",
        "",
        "Lint only:",
        f"  {drive.get('prepare')}",
    ]
    return "\n".join(lines) + "\n"
