"""Seed and repair ComfyUI-Manager's ``config.ini`` and ``channels.list``.

``comfy install`` drops ComfyUI-Manager into the managed ComfyUI, and Manager
writes its own ``config.ini`` on first launch with
``channel_url = https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main``.
On current Manager builds that URL raises ``[InvalidChannel]`` on every start
(Comfy-Org/ComfyUI-Manager #2709, #2561).

Manager only accepts a ``channel_url`` that is listed in the user's
``channels.list`` (``name::url`` lines; it validates against those URLs), so
both files are written: ``config.ini`` points at :data:`CHANNEL_URL` and
``channels.list`` maps the ``default`` channel to the same URL.

Edits are line-based and idempotent. Other keys, sections and channels are
kept. A ``channel_url`` or ``default::`` entry the user changed to some other
URL is left alone.

Manager keeps these files in ``<ComfyUI>/user/__manager`` on ComfyUI builds
with the System User API (``folder_paths.get_system_user_directory``), and in
``<ComfyUI>/user/default/ComfyUI-Manager`` on older ones. Existing files in
either place are repaired; new ones are only created where this ComfyUI's
Manager reads them, because a stray config.ini in the other folder triggers
Manager's migration path.

Manager has no ``channel`` or ``auto-updates`` key in ``config.ini``; the
"default" channel is whichever ``channels.list`` name maps to ``channel_url``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CHANNEL_URL = "https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main"
BROKEN_CHANNEL_URLS = frozenset({
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main",
    "https://raw.githubusercontent.com/ComfyOrg/ComfyUI-Manager/main",
})
DEFAULT_CHANNEL_NAME = "default"
EXTRA_CHANNEL_NAME = "comfy-org"

NEW_DIR = ("user", "__manager")
LEGACY_DIR = ("user", "default", "ComfyUI-Manager")

# Manager's channels.list.template, with ``default`` on the working URL.
CHANNELS_TEMPLATE = (
    f"{DEFAULT_CHANNEL_NAME}::{CHANNEL_URL}\n"
    "recent::https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/node_db/new\n"
    "legacy::https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/node_db/legacy\n"
    "forked::https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/node_db/forked\n"
    "dev::https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/node_db/dev\n"
    "tutorial::https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/node_db/tutorial\n"
)

CREATED, FIXED, UNCHANGED = "created", "fixed", "unchanged"

_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_CHANNEL_URL = re.compile(r"^(\s*channel_url\s*[=:]\s*)(.*?)\s*$")


def _norm(url: str) -> str:
    return url.strip().rstrip("/")


def _is_broken(url: str) -> bool:
    return _norm(url) in BROKEN_CHANNEL_URLS


@dataclass
class FileAction:
    path: Path
    action: str
    detail: str = ""


@dataclass
class ManagerConfigReport:
    actions: list[FileAction] = field(default_factory=list)

    @property
    def changed(self) -> list[FileAction]:
        return [a for a in self.actions if a.action != UNCHANGED]

    def lines(self) -> list[str]:
        return [f"{a.action} {a.path}" + (f" ({a.detail})" if a.detail else "") for a in self.changed]


def has_system_user_api(code_dir: Path) -> bool:
    try:
        text = (Path(code_dir) / "folder_paths.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "get_system_user_directory" in text


def manager_dirs(code_dir: Path) -> tuple[Path, list[Path]]:
    """``(primary, all)``: where this ComfyUI's Manager reads config, and every place it may live."""
    code_dir = Path(code_dir)
    new, legacy = code_dir.joinpath(*NEW_DIR), code_dir.joinpath(*LEGACY_DIR)
    primary = new if has_system_user_api(code_dir) else legacy
    return primary, [new, legacy]


def fix_config_text(text: str) -> tuple[str, str]:
    """``(new_text, detail)``; ``detail`` is empty when nothing had to change."""
    lines = text.splitlines()
    section = None
    default_at = None
    url_at = None
    for i, line in enumerate(lines):
        m = _SECTION.match(line)
        if m:
            section = m.group(1).strip()
            if section == "default" and default_at is None:
                default_at = i
            continue
        if section == "default" and url_at is None and _CHANNEL_URL.match(line):
            url_at = i
    if url_at is not None:
        m = _CHANNEL_URL.match(lines[url_at])
        current = m.group(2)
        if not _is_broken(current):
            return text, ""
        lines[url_at] = m.group(1) + CHANNEL_URL
        detail = f"channel_url {_norm(current)} -> {CHANNEL_URL}"
    elif default_at is not None:
        lines.insert(default_at + 1, f"channel_url = {CHANNEL_URL}")
        detail = f"channel_url set to {CHANNEL_URL}"
    else:
        lines[:0] = ["[default]", f"channel_url = {CHANNEL_URL}", ""] if lines else ["[default]", f"channel_url = {CHANNEL_URL}"]
        detail = f"added [default] channel_url = {CHANNEL_URL}"
    return "\n".join(lines) + "\n", detail


def fix_channels_text(text: str) -> tuple[str, str]:
    """``(new_text, detail)``: make sure :data:`CHANNEL_URL` is a listed channel, as ``default`` when possible."""
    lines = text.splitlines()
    entries = [(i, *line.split("::", 1)) for i, line in enumerate(lines) if line.count("::") == 1]
    if any(_norm(url) == CHANNEL_URL for _, _, url in entries):
        return text, ""
    default = next(((i, url) for i, name, url in entries if name.strip() == DEFAULT_CHANNEL_NAME), None)
    if default is None:
        lines.insert(0, f"{DEFAULT_CHANNEL_NAME}::{CHANNEL_URL}")
        detail = f"added {DEFAULT_CHANNEL_NAME}::{CHANNEL_URL}"
    elif _is_broken(default[1]):
        lines[default[0]] = f"{DEFAULT_CHANNEL_NAME}::{CHANNEL_URL}"
        detail = f"{DEFAULT_CHANNEL_NAME} {_norm(default[1])} -> {CHANNEL_URL}"
    else:
        lines.append(f"{EXTRA_CHANNEL_NAME}::{CHANNEL_URL}")
        detail = f"added {EXTRA_CHANNEL_NAME}::{CHANNEL_URL}"
    return "\n".join(lines) + "\n", detail


def _apply(path: Path, fixer, seed: str, *, create: bool, apply: bool) -> FileAction | None:
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
        new, detail = fixer(text)
        if new == text or not detail:
            return FileAction(path, UNCHANGED)
        if apply:
            path.write_text(new, encoding="utf-8")
        return FileAction(path, FIXED, detail)
    if not create:
        return None
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(seed, encoding="utf-8")
    return FileAction(path, CREATED, f"channel_url = {CHANNEL_URL}")


def ensure_manager_config(code_dir: Path, *, apply: bool = True) -> ManagerConfigReport:
    """Seed the primary Manager folder and repair any existing ``config.ini`` / ``channels.list``.

    With ``apply=False`` nothing is written; the report says what would change.
    """
    primary, dirs = manager_dirs(code_dir)
    # Manager migrates a legacy config.ini into __manager only while the new one is absent.
    legacy_config_pending = primary == dirs[0] and (dirs[1] / "config.ini").is_file()
    report = ManagerConfigReport()
    for d in dirs:
        for name, fixer, seed in (
            ("config.ini", fix_config_text, f"[default]\nchannel_url = {CHANNEL_URL}\n"),
            ("channels.list", fix_channels_text, CHANNELS_TEMPLATE),
        ):
            create = d == primary and not (name == "config.ini" and legacy_config_pending)
            action = _apply(d / name, fixer, seed, create=create, apply=apply)
            if action is not None:
                report.actions.append(action)
    return report
