"""Hermes skill package: renders, matches agent tools + CLI, install copies.

Run: python -m pytest tests/test_hermes_skill.py -q
"""

from __future__ import annotations

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


def registered_agent_tools() -> list[str]:
    from master_agent.agent_api import TOOLS

    return list(TOOLS)


def registered_cli_commands() -> set[str]:
    """Commands registered by the CLI dispatcher (parser lives in cli/parser.py)."""
    sources = [
        MAIN_PY.read_text(encoding="utf-8"),
        (VIDEO_BUDDY / "master_agent" / "cli" / "parser.py").read_text(encoding="utf-8"),
    ]
    names: set[str] = set()
    for src in sources:
        names.update(re.findall(r"sub\.add_parser\(\s*[\"']([^\"']+)[\"']", src))
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
    assert "master_agent agent" in desc
    body = text.split("\n---\n", 1)[1]
    assert "# Video Buddy" in body
    for needle in (
        "One gateway",
        "LTX bot gateway",
        "~/.hermes/skills/video-buddy/",
        "python -m master_agent agent",
        "agent list",
        ":8188",
        "8n+1",
        "L0",
        "L5",
        "Imagine",
    ):
        assert needle in body, f"missing {needle!r}"
    for gone in ("mcp_server.py", "mcp_servers", "hermes register", "master_agent ui", "/a2a"):
        assert gone not in _skill_corpus(), f"removed surface still documented: {gone!r}"
    assert "Not a Hermes profile" not in body


def test_stop_lines_match_curriculum():
    body = _skill_corpus()
    assert "junk" in body.lower()
    assert "<100KB" in body or "<100kb" in body.lower()
    assert "<3" in body
    assert "never" in body.lower() and "8" in body
    assert "diagnose" in body.lower()
    assert "LESSON_BUDDY" in body or "L0→L5" in body or "L0->L5" in body


def test_agent_tools_documented_and_no_mcp_server():
    corpus = _skill_corpus()
    for name in registered_agent_tools():
        assert name in corpus, f"agent tool {name} missing from skill package"
    assert not (VIDEO_BUDDY / "master_agent" / "mcp_server.py").exists()


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
        "run",
        "power-tune",
        "persona",
        "soul",
        "brief",
        "fractal",
        "music",
        "mv",
        "download-flux",
        "download-models",
        "character",
        "lora",
        "comfy",
        "diagnose",
        "budget",
        "agent",
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


def test_install_hermes_skill_writes_skill_only(tmp_path: Path):
    from install_hermes_skill import main

    home = tmp_path / "hermes-home"
    rc = main(["--hermes-home", str(home)])
    assert rc == 0
    assert (home / "skills" / "video-buddy" / "SKILL.md").is_file()
    assert (home / "skills" / "video-buddy" / "PROFILE.md").is_file()
    assert not (home / "profiles").exists()


def test_install_hermes_skill_into_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    from install_hermes_skill import hermes_profile_skill_dir, main

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    expected = tmp_path / "hh" / "profiles" / "ltx" / "skills" / "media" / "video-buddy"
    assert hermes_profile_skill_dir("ltx") == expected
    assert main(["--profile", "ltx"]) == 0
    assert str(expected) in capsys.readouterr().out
    for name in ("SKILL.md", "TOOLS.md", "PROFILE.md", "references/local-only.md"):
        assert (expected / name).read_text(encoding="utf-8") == (SKILL_DIR / name).read_text(encoding="utf-8")
    assert not (tmp_path / "hh" / "skills").exists()
    assert sorted(p.name for p in (tmp_path / "hh" / "profiles" / "ltx").iterdir()) == ["skills"]


def test_install_hermes_skill_profile_defaults_to_home_dot_hermes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from install_hermes_skill import hermes_profile_skill_dir

    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert hermes_profile_skill_dir("ltx") == tmp_path / ".hermes" / "profiles" / "ltx" / "skills" / "media" / "video-buddy"


def test_install_hermes_skill_dest_flag(tmp_path: Path):
    from install_hermes_skill import main

    dest = tmp_path / "custom" / "video-buddy"
    assert main(["--dest", str(dest)]) == 0
    assert (dest / "SKILL.md").is_file() and (dest / "TOOLS.md").is_file() and (dest / "PROFILE.md").is_file()


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", ".."])
def test_install_hermes_skill_rejects_bad_profile_names(tmp_path: Path, bad: str):
    from install_hermes_skill import main

    assert main(["--hermes-home", str(tmp_path), "--profile", bad]) == 1
    assert not (tmp_path / "profiles").exists() or not any((tmp_path / "profiles").rglob("SKILL.md"))


def test_install_hermes_skill_dest_and_profile_are_exclusive(tmp_path: Path):
    from install_hermes_skill import main

    with pytest.raises(SystemExit) as exc:
        main(["--dest", str(tmp_path / "d"), "--profile", "ltx"])
    assert exc.value.code == 2


def test_skill_package_says_how_to_start_comfy():
    files = [SKILL_MD, TOOLS_MD, SKILL_DIR / "PROFILE.md", VIDEO_BUDDY.parent / "skills" / "video-buddy" / "SKILL.md"]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "comfy_start" in text, path
    skill = SKILL_MD.read_text(encoding="utf-8")
    for needle in (
        "python -m master_agent comfy start",
        "python -m master_agent agent comfy_start",
        "NEVER run ComfyUI `main.py`",
        "background=true",
        "/system_stats",
        "do not retry the identical command",
        "--profile ltx",
    ):
        assert needle in skill, needle
    agents = (VIDEO_BUDDY.parent / "docs" / "AGENTS.md").read_text(encoding="utf-8")
    assert "NEVER run ComfyUI `main.py`" in agents and "background=true" in agents


def test_skill_package_drops_removed_surfaces():
    files = [SKILL_MD, TOOLS_MD, SKILL_DIR / "PROFILE.md", VIDEO_BUDDY.parent / "skills" / "video-buddy" / "SKILL.md",
             VIDEO_BUDDY.parent / "skills" / "README.md"]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for gone in ("8189", "MCP", "mcp_server", "A2A", "/a2a"):
            assert gone not in text, (path.name, gone)
