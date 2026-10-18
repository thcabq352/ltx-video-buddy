#!/usr/bin/env python3
"""Copy the Video Buddy skill into a Hermes home or a Hermes profile.

Works on Windows, macOS, and Linux. No secrets. Writes only the skill
folder: no profile config, no gateway config, no `.env`. Never binds port 8642.

Destinations (first match wins):

  --dest DIR          exactly DIR
  --profile NAME      <hermes home>/profiles/NAME/skills/media/video-buddy
  (default)           <hermes home>/skills/video-buddy

<hermes home> is --hermes-home, else $HERMES_HOME, else ~/.hermes.

The LTX bot runs in the ``ltx`` profile:

  python install_hermes_skill.py --profile ltx
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "skills" / "video-buddy"
SKILL_FILES = ("SKILL.md", "TOOLS.md", "PROFILE.md", "references/local-only.md")
PROFILE_SKILL_CATEGORY = "media"
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def hermes_home(override: Path | None = None) -> Path:
    if override is not None:
        return override
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def hermes_skill_dir(home: Path | None = None) -> Path:
    return hermes_home(home) / "skills" / "video-buddy"


def hermes_profile_skill_dir(profile: str, home: Path | None = None) -> Path:
    if not _PROFILE_NAME.match(profile or "") or profile in {".", ".."}:
        raise ValueError(f"invalid Hermes profile name: {profile!r}")
    return hermes_home(home) / "profiles" / profile / "skills" / PROFILE_SKILL_CATEGORY / "video-buddy"


def install_skill(
    *,
    dest: Path | None = None,
    home: Path | None = None,
    profile: str | None = None,
) -> Path:
    skill = SRC / "SKILL.md"
    if not skill.is_file():
        raise FileNotFoundError(f"missing skill source: {skill}")
    if dest is not None:
        target = dest
    elif profile is not None:
        target = hermes_profile_skill_dir(profile, home)
    else:
        target = hermes_skill_dir(home)
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
    parser = argparse.ArgumentParser(
        description="Install the Video Buddy skill (SKILL.md, TOOLS.md, PROFILE.md, references/)",
        epilog="LTX bot: python install_hermes_skill.py --profile ltx",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--dest", help="install into exactly this directory")
    where.add_argument(
        "--profile",
        help="Hermes profile name; installs into <hermes home>/profiles/NAME/skills/media/video-buddy",
    )
    parser.add_argument("--hermes-home", help="override HERMES_HOME / ~/.hermes")
    args = parser.parse_args(argv)
    dest = Path(args.dest).expanduser() if args.dest else None
    home = Path(args.hermes_home).expanduser() if args.hermes_home else None
    try:
        target = install_skill(dest=dest, home=home, profile=args.profile)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    print(f"OK    skill installed at {target}")
    for name in SKILL_FILES:
        if (target / name).is_file():
            print(f"      {name}")
    print("Next  python -m master_agent agent list   (cwd=video_buddy/)")
    print("      start ComfyUI only with: python -m master_agent agent comfy_start")
    print("      do not bind 8642 from Buddy — that port is the default Hermes gateway")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
