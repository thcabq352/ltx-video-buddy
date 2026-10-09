#!/usr/bin/env python3
"""Copy the Video Buddy skill into the LTX bot gateway's Hermes home.

Works on Windows, macOS, and Linux. No secrets. Writes only the skill
folder: no profile, no gateway config, no `.env`. Never binds port 8642.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "skills" / "video-buddy"
SKILL_FILES = ("SKILL.md", "TOOLS.md", "PROFILE.md", "references/local-only.md")


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
            dest_path = target / name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_path)
            copied.append(name)
    if "SKILL.md" not in copied:
        raise FileNotFoundError(f"SKILL.md not copied from {SRC}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the Video Buddy skill")
    parser.add_argument("--dest", help="override skill destination directory")
    parser.add_argument("--hermes-home", help="override HERMES_HOME / ~/.hermes")
    args = parser.parse_args(argv)
    dest = Path(args.dest).expanduser() if args.dest else None
    home = Path(args.hermes_home).expanduser() if args.hermes_home else None
    try:
        target = install_skill(dest=dest, home=home)
    except FileNotFoundError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    print(f"OK    skill installed at {target}")
    print("Next  python -m master_agent agent list   (cwd=video_buddy/)")
    print("      do not bind 8642 from Buddy — that port is the default Hermes gateway")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
