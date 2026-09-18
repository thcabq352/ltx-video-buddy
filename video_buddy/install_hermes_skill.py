#!/usr/bin/env python3
"""Copy the Video Buddy Hermes skill and seat profile `ltx`.

Works on Windows, macOS, and Linux. No secrets. Does not start MCP.
Never creates a profile `.env`. Never binds port 8642.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "skills" / "video-buddy"
SKILL_FILES = ("SKILL.md", "TOOLS.md", "PROFILE.md")


def hermes_home(override: Path | None = None) -> Path:
    if override is not None:
        return override
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def hermes_skill_dir(home: Path | None = None) -> Path:
    return hermes_home(home) / "skills" / "video-buddy"


def install_skill(*, dest: Path | None = None, home: Path | None = None) -> Path:
    skill = SRC / "SKILL.md"
    if not skill.is_file():
        raise FileNotFoundError(f"missing skill source: {skill}")
    target = dest if dest is not None else hermes_skill_dir(home)
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in SKILL_FILES:
        src = SRC / name
        if src.is_file():
            shutil.copy2(src, target / name)
            copied.append(name)
    if "SKILL.md" not in copied:
        raise FileNotFoundError(f"SKILL.md not copied from {SRC}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install Video Buddy skill and seat Hermes profile ltx"
    )
    parser.add_argument("--dest", help="override skill destination directory")
    parser.add_argument("--hermes-home", help="override HERMES_HOME / ~/.hermes")
    parser.add_argument("--skip-profile", action="store_true", help="copy skill only")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite custom profiles/ltx/SOUL.md",
    )
    args = parser.parse_args(argv)
    dest = Path(args.dest).expanduser() if args.dest else None
    home = Path(args.hermes_home).expanduser() if args.hermes_home else None
    try:
        target = install_skill(dest=dest, home=home)
    except FileNotFoundError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    print(f"OK    Hermes skill installed at {target}")
    if args.skip_profile:
        print("Next  confirm server id master-agent in ~/.hermes/config.yaml")
        print("      start MCP (cwd=video_buddy/): <venv python> master_agent/mcp_server.py")
        return 0
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from master_agent.hermes.profile import default_python, register_ltx_profile

    result = register_ltx_profile(
        home=home,
        video_buddy_root=ROOT,
        python=default_python(ROOT),
        force=bool(args.force),
    )
    print(f"OK    Hermes profile ltx at {result.profile_dir}")
    if result.soul_skipped:
        print("      custom SOUL.md left in place (pass --force to overwrite)")
    print("Next  python -m master_agent hermes status")
    print("      A2A fallback remains on :8189  GET /.well-known/agent.json  POST /a2a")
    print("      do not bind 8642 from Buddy — that port is the default Hermes gateway")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
