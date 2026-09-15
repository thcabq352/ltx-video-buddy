"""Write / merge Hermes profile `ltx` (no secrets, never creates .env)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from master_agent.hermes.gateways import LTX_RESEARCH_SYSTEM, hermes_home

PROFILE_ID = "ltx"
PROFILE_DESCRIPTION = (
    "Video Buddy / LTX research seat. Local ComfyUI video on this machine."
)
MANAGED_COMMENT = "# managed-by: video-buddy"


@dataclass
class RegisterResult:
    profile_dir: Path
    soul_written: bool = False
    soul_skipped: bool = False
    config_written: bool = False
    env_created: bool = False
    notes: list[str] = field(default_factory=list)


def default_python(video_buddy_root: Path) -> str:
    win = video_buddy_root / ".venv" / "Scripts" / "python.exe"
    nix = video_buddy_root / ".venv" / "bin" / "python"
    if win.is_file():
        return str(win)
    if nix.is_file():
        return str(nix)
    return sys.executable


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: dict[str, Any], *, header: str = MANAGED_COMMENT) -> None:
    try:
        import yaml

        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except Exception:
        body = _fallback_yaml(data)
    path.write_text(f"{header}\n{body}", encoding="utf-8")


def _fallback_yaml(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = "  " * indent
    for key, val in data.items():
        if isinstance(val, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_fallback_yaml(val, indent + 1))
        elif isinstance(val, list):
            lines.append(f"{pad}{key}:")
            for item in val:
                lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{key}: {val}")
    return "\n".join(lines) + ("\n" if indent == 0 else "")


def _should_write_soul(path: Path, *, force: bool) -> bool:
    if force or not path.is_file():
        return True
    current = path.read_text(encoding="utf-8").strip()
    return current == LTX_RESEARCH_SYSTEM.strip()


def _merge_config(existing: dict[str, Any], *, video_buddy_root: Path, python: str) -> dict[str, Any]:
    out = dict(existing)
    term = out.get("terminal")
    if not isinstance(term, dict):
        term = {}
        out["terminal"] = term
    term["cwd"] = str(video_buddy_root)
    servers = out.get("mcp_servers")
    if not isinstance(servers, dict):
        servers = {}
        out["mcp_servers"] = servers
    servers["master-agent"] = {
        "command": python,
        "args": [str(video_buddy_root / "master_agent" / "mcp_server.py")],
    }
    return out


def register_ltx_profile(
    home: Path | None = None,
    *,
    video_buddy_root: Path,
    python: str | None = None,
    force: bool = False,
) -> RegisterResult:
    """Merge-if-exists. Refresh SOUL only when it still equals LTX_RESEARCH_SYSTEM or force."""
    root = Path(home) if home is not None else hermes_home()
    profile_dir = root / "profiles" / PROFILE_ID
    profile_dir.mkdir(parents=True, exist_ok=True)
    result = RegisterResult(profile_dir=profile_dir)
    py = python or default_python(Path(video_buddy_root))

    soul_path = profile_dir / "SOUL.md"
    if _should_write_soul(soul_path, force=force):
        soul_path.write_text(LTX_RESEARCH_SYSTEM.strip() + "\n", encoding="utf-8")
        result.soul_written = True
    else:
        result.soul_skipped = True
        result.notes.append("left custom SOUL.md in place (pass --force to overwrite)")

    meta_path = profile_dir / "profile.yaml"
    if force or not meta_path.is_file():
        _write_yaml(
            meta_path,
            {"name": PROFILE_ID, "description": PROFILE_DESCRIPTION},
        )

    config_path = profile_dir / "config.yaml"
    existing = _read_yaml(config_path)
    merged = _merge_config(existing, video_buddy_root=Path(video_buddy_root), python=py)
    _write_yaml(config_path, merged)
    result.config_written = True

    env_path = profile_dir / ".env"
    if env_path.exists():
        result.notes.append("existing .env left untouched")
    return result
