"""``models register <file-or-dir>``: adopt LTX 2.3 weights the user already has.

Each file is matched by name against the ``ltx23_core`` bundle (official name,
accepted aliases, scan alternates) and, for the official name, by exact byte
size. A match is hard-linked (same volume) or copied into
``MODELS_DIR/<dest_folder>/``. Files already visible to Buddy (in ``MODELS_DIR``
or the resolved managed Comfy ``models/`` folder) are left where they are.
Writes go through the same destination guard as pack downloads.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from master_agent.models.deferred import BUNDLE, clear_deferred

MODES = ("auto", "hardlink", "symlink", "copy")
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".sft", ".ckpt", ".pt", ".pth", ".bin")

LINKED = "linked"
SYMLINKED = "symlinked"
COPIED = "copied"
IN_PLACE = "already-in-place"
WOULD = "would-register"
REJECTED = "rejected"


@dataclass
class RegisterItem:
    source: str
    action: str
    detail: str
    key: str = ""
    dest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegisterReport:
    items: list[RegisterItem]
    missing: list[str]
    complete: bool
    marker_cleared: bool = False

    @property
    def ok(self) -> bool:
        return not any(i.action == REJECTED for i in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "bundle": BUNDLE,
            "complete": self.complete,
            "missing": self.missing,
            "marker_cleared": self.marker_cleared,
            "items": [i.to_dict() for i in self.items],
        }


def _bundle_files():
    from master_agent.models.weights import files_for_bundle

    return files_for_bundle(BUNDLE)


def _names_for(weight) -> tuple[str, ...]:
    from master_agent.models.weights import LTX23_SCAN_ALTERNATES

    return (*weight.candidates, *LTX23_SCAN_ALTERNATES.get(weight.key, ()))


def match_weight(path: Path):
    """The ``ltx23_core`` slot whose accepted names include ``path.name`` (case-insensitive)."""
    name = path.name.lower()
    for weight in _bundle_files():
        if any(n.lower() == name for n in _names_for(weight)):
            return weight
    return None


def _collect(sources: Iterable[Path]) -> tuple[list[Path], list[RegisterItem]]:
    """Files to register. Folder scans keep only LTX 2.3 names; other weights there are ignored."""
    files: list[Path] = []
    problems: list[RegisterItem] = []
    for raw in sources:
        path = Path(raw).expanduser()
        if path.is_dir():
            found = sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES
            )
            matched = [p for p in found if match_weight(p) is not None]
            if not matched:
                problems.append(
                    RegisterItem(str(path), REJECTED, "no LTX 2.3 pack files in this folder (matched by file name)")
                )
            files.extend(matched)
        elif path.is_file():
            files.append(path)
        else:
            problems.append(RegisterItem(str(path), REJECTED, "no such file or folder"))
    return files, problems


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or os.path.samefile(a, b)
    except OSError:
        return False


def _visible_in_place(path: Path, dest_folder: str) -> bool:
    """True when ``path`` already sits in ``<searched models root>/<dest_folder>/``."""
    from master_agent.config import MODELS_DIR
    from master_agent.comfy.root import managed_comfy_root

    try:
        parent = path.resolve().parent
    except OSError:
        return False
    if parent.name.lower() != dest_folder.lower():
        return False
    roots = [Path(MODELS_DIR), managed_comfy_root() / "models"]
    return any(_same_file(parent.parent, root) for root in roots if root.exists())


def _place(src: Path, dest: Path, mode: str) -> str:
    if mode in {"auto", "hardlink"}:
        try:
            os.link(src, dest)
            return LINKED
        except OSError:
            if mode == "hardlink":
                raise
    if mode == "symlink":
        os.symlink(src.resolve(), dest)
        return SYMLINKED
    tmp = dest.with_name(dest.name + ".partial")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    return COPIED


def _register_one(src: Path, *, mode: str, force: bool, dry_run: bool) -> RegisterItem:
    from master_agent.config import MODELS_DIR
    from master_agent.models.download import DownloadDestinationError, assert_pack_download_destination

    weight = match_weight(src)
    if weight is None:
        expected = ", ".join(w.filename for w in _bundle_files())
        return RegisterItem(str(src), REJECTED, f"not an LTX 2.3 pack file name. Expected one of: {expected}")
    size = src.stat().st_size
    if size <= 0:
        return RegisterItem(str(src), REJECTED, "file is empty", key=weight.key)
    if src.name.lower() == weight.filename.lower() and size != weight.size_bytes and not force:
        return RegisterItem(
            str(src),
            REJECTED,
            f"size {size:,} bytes, expected {weight.size_bytes:,} ({weight.size_label}). "
            "Probably an unfinished download; re-download it, or pass --force to register it anyway",
            key=weight.key,
        )
    if _visible_in_place(src, weight.dest_folder):
        return RegisterItem(str(src), IN_PLACE, f"Buddy already finds it in {src.parent}", key=weight.key, dest=str(src))
    dest = Path(MODELS_DIR) / weight.dest_folder / src.name
    if dest.exists() or dest.is_symlink():
        if _same_file(src, dest):
            return RegisterItem(str(src), IN_PLACE, f"already registered at {dest}", key=weight.key, dest=str(dest))
        if dest.is_file() and dest.stat().st_size == size and not force:
            return RegisterItem(str(src), IN_PLACE, f"{dest} already exists with the same size", key=weight.key, dest=str(dest))
        if not force:
            return RegisterItem(
                str(src), REJECTED, f"{dest} already exists with a different size; remove it or pass --force",
                key=weight.key, dest=str(dest),
            )
    try:
        dest = assert_pack_download_destination(dest)
    except DownloadDestinationError as exc:
        return RegisterItem(str(src), REJECTED, str(exc), key=weight.key, dest=str(dest))
    if dry_run:
        return RegisterItem(str(src), WOULD, f"{mode} → {dest}", key=weight.key, dest=str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        action = _place(src, dest, mode)
    except OSError as exc:
        return RegisterItem(str(src), REJECTED, f"could not {mode} into {dest}: {exc}", key=weight.key, dest=str(dest))
    return RegisterItem(str(src), action, f"→ {dest}", key=weight.key, dest=str(dest))


def register_weights(
    sources: Iterable[Path | str],
    *,
    mode: str = "auto",
    force: bool = False,
    dry_run: bool = False,
) -> RegisterReport:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    from master_agent.models.weights import scan_bundle

    files, items = _collect(Path(s) for s in sources)
    seen: set[Path] = set()
    for src in files:
        key = src.resolve()
        if key in seen:
            continue
        seen.add(key)
        items.append(_register_one(src, mode=mode, force=force, dry_run=dry_run))
    status = scan_bundle(BUNDLE)
    cleared = False
    if status.ok and not dry_run:
        cleared = clear_deferred()
    return RegisterReport(
        items=items,
        missing=[w.filename for w in status.missing_mandatory],
        complete=status.ok,
        marker_cleared=cleared,
    )


def format_report(report: RegisterReport) -> str:
    tags = {REJECTED: "FAIL", WOULD: "PLAN", IN_PLACE: "OK  "}
    lines = [f"{tags.get(i.action, 'OK  ')}  {i.action:<16} {Path(i.source).name}: {i.detail}" for i in report.items]
    if not report.items:
        lines.append("Nothing to register.")
    if report.complete:
        lines.append("LTX 2.3 pack complete.")
        if report.marker_cleared:
            lines.append("Cleared the --skip-weights marker; doctor now checks the weights normally.")
    else:
        lines.append(f"LTX 2.3 pack still missing {len(report.missing)}: " + ", ".join(report.missing))
    return "\n".join(lines)
