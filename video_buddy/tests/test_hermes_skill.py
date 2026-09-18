"""Hermes skill package: renders, matches mcp_server + CLI, install copies.

Run: python -m pytest tests/test_hermes_skill.py -q
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

VIDEO_BUDDY = Path(__file__).resolve().parents[1]
if str(VIDEO_BUDDY) not in sys.path:
    sys.path.insert(0, str(VIDEO_BUDDY))

from install_hermes_skill import SKILL_FILES, install_skill  # noqa: E402
SKILL_DIR = VIDEO_BUDDY / "skills" / "video-buddy"
SKILL_MD = SKILL_DIR / "SKILL.md"
TOOLS_MD = SKILL_DIR / "TOOLS.md"
MCP_SERVER = VIDEO_BUDDY / "master_agent" / "mcp_server.py"
MAIN_PY = VIDEO_BUDDY / "master_agent" / "__main__.py"


def _skill_corpus() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in (SKILL_MD, TOOLS_MD) if p.is_file())


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        raise AssertionError("SKILL.md must open with YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise AssertionError("SKILL.md frontmatter is not closed")
    block = text[4:end]
    out: dict[str, str] = {}
    for raw in block.splitlines():
        if not raw.strip() or raw.strip().startswith("#") or ":" not in raw:
            continue
        key, val = raw.split(":", 1)
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def registered_mcp_tools() -> list[str]:
    tree = ast.parse(MCP_SERVER.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            func = call.func if call else dec
            if isinstance(func, ast.Attribute) and func.attr == "tool":
                names.append(node.name)
                break
    return names


def registered_cli_commands() -> set[str]:
    src = MAIN_PY.read_text(encoding="utf-8")
    names = set(re.findall(r"sub\.add_parser\(\s*[\"']([^\"']+)[\"']", src))
    for aliases in re.findall(r"aliases\s*=\s*\[([^\]]+)\]", src):
        names.update(re.findall(r"[\"']([^\"']+)[\"']", aliases))
    return names


def test_skill_renders_frontmatter_and_sections():
    assert SKILL_MD.is_file()
    assert TOOLS_MD.is_file()
    assert (SKILL_DIR / "PROFILE.md").is_file()
    text = SKILL_MD.read_text(encoding="utf-8")
    meta = _frontmatter(text)
    assert meta.get("name") == "video-buddy"
    desc = meta.get("description") or ""
    assert desc.startswith("Use when")
    assert "Video Buddy" in desc or "video buddy" in desc.lower()
    assert "master-agent" in desc
    body = text.split("\n---\n", 1)[1]
    assert "# Video Buddy" in body
    for needle in (
        "MCP ≠ skills",
        "~/.hermes/skills/video-buddy/",
        "master-agent",
        ":8188",
        ":8189",
        "8n+1",
        "L0",
        "L5",
        "Imagine",
        "ltx",
        "A2A",
    ):
        assert needle in body, f"missing {needle!r}"
    assert "Not a Hermes profile" not in body


def test_stop_lines_match_curriculum():
    body = _skill_corpus()
    assert "junk" in body.lower()
    assert "<100KB" in body or "<100kb" in body.lower()
    assert "<3" in body
    assert "never" in body.lower() and "8" in body
    assert "diagnose" in body.lower()
    assert "LESSON_BUDDY" in body or "L0→L5" in body or "L0->L5" in body


def test_mcp_tools_match_mcp_server():
    tools = registered_mcp_tools()
    assert tools == [
        "health",
        "create_video",
        "plan_storyboard",
        "judge_asset",
        "search_workflows",
        "search_runs",
        "kb_ingest",
        "list_models",
        "validate_workflow",
        "create_character",
        "train_lora",
    ]
    corpus = _skill_corpus()
    for name in tools:
        assert name in corpus, f"MCP tool {name} missing from skill package"
    # Do not document invented tools as MCP.
    for fake in ("diagnose", "comfy_run", "download_models", "curriculum"):
        assert f"def {fake}" not in MCP_SERVER.read_text(encoding="utf-8")


def test_cli_commands_documented():
    commands = registered_cli_commands()
    expected = {
        "curriculum",
        "about",
        "setup",
        "doctor",
        "workflows",
        "health",
        "fetch-object-info",
        "scan-models",
        "validate",
        "kb",
        "ui",
        "run",
        "power-tune",
        "persona",
        "soul",
        "brief",
        "fractal",
        "music",
        "download-flux",
        "download-models",
        "character",
        "lora",
        "comfy",
        "diagnose",
        "budget",
        "hermes",
        "capabilities",
    }
    assert expected <= commands
    corpus = _skill_corpus()
    for name in expected:
        assert name in corpus, f"CLI {name} missing from skill package"


def test_install_hermes_skill_copies(tmp_path: Path):
    dest = tmp_path / "skills" / "video-buddy"
    out = install_skill(dest=dest)
    assert out == dest
    for name in SKILL_FILES:
        copied = dest / name
        src = SKILL_DIR / name
        assert copied.is_file()
        assert copied.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_install_hermes_skill_uses_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    home = tmp_path / "hermes-home"
    out = install_skill(home=home)
    assert out == home / "skills" / "video-buddy"
    assert (out / "SKILL.md").is_file()


def test_install_hermes_skill_seats_ltx_profile(tmp_path: Path):
    from install_hermes_skill import main

    home = tmp_path / "hermes-home"
    rc = main(["--hermes-home", str(home)])
    assert rc == 0
    assert (home / "skills" / "video-buddy" / "SKILL.md").is_file()
    assert (home / "skills" / "video-buddy" / "PROFILE.md").is_file()
    assert (home / "profiles" / "ltx" / "SOUL.md").is_file()
    assert not (home / "profiles" / "ltx" / ".env").exists()
