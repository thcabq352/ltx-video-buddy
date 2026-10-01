"""Doctor rows and consent-gated HeartMuLa downloads.

``download-models --heartmula`` lists missing attested files.
``--yes`` is the only path that calls the Hub. Tests and CI must not pass it
against the network.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any, Callable

from master_agent.heartmula.config import (
    DEFAULT_CODEC_REPO,
    DEFAULT_MULA_REPO,
    DEFAULT_TRANSCRIPTOR_REPO,
    HeartMuLaConfigError,
    all_slots,
    models_dir,
    slots_for,
)
from master_agent.heartmula.generate import MissingHeartMuLaWeights

# Hugging Face ``usedStorage`` bytes retrieved 2026-09-30. Not a live quota.
HUB_USED_STORAGE_BYTES: dict[str, int] = {
    "HeartMuLa/HeartMuLa-oss-3B-happy-new-year": 15_752_486_168,
    "HeartMuLa/HeartMuLa-oss-3B": 15_752_486_168,
    "HeartMuLa/HeartMuLa-RL-oss-3B-20260123": 15_752_486_168,
    "HeartMuLa/HeartCodec-oss-20260123": 6_638_383_908,
    "HeartMuLa/HeartTranscriptor-oss": 3_055_544_304,
}

# Tower Comfy Desk, 2026-09-30: F: had about 28GB free. The default pull is
# happy-new-year + codec 20260123 + transcriptor (~25GB of those listings).
TOWER_FREE_NOTE = (
    "Tower disk check 2026-09-30: F: had about 28GB free. "
    "The default HeartMuLa pull is multi-GB "
    "(HeartMuLa-oss-3B-happy-new-year ~15.8GB, "
    "HeartCodec-oss-20260123 ~6.6GB, "
    "HeartTranscriptor-oss ~3.1GB, Hub usedStorage). "
    "HeartMuLa/HeartCodec-oss returned 401 on the tower and in this pack's "
    "Hub lookup — download HeartCodec-oss-20260123 into the HeartCodec-oss "
    "folder. Pair HeartMuLa-RL-oss-3B-20260123 with that same codec. "
    "Check free space before --yes. Nothing is fetched by doctor."
)

COMFY_CLASSES = ("HeartMuLa_Generate", "HeartMuLa_Transcribe")
COMFY_DISPLAY = {
    "HeartMuLa_Generate": "HeartMuLa Music Generator",
    "HeartMuLa_Transcribe": "HeartMuLa Lyrics Transcriber",
}
COMFY_COMMIT = "fdb53c4"


def _usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def scan_slots(group: str | None = None, root: Path | None = None) -> list[dict[str, Any]]:
    base = Path(root) if root is not None else models_dir()
    chosen = all_slots() if group is None else slots_for(group)
    rows: list[dict[str, Any]] = []
    for slot in chosen:
        path = base / slot.dest_rel
        rows.append(
            {
                "group": slot.group,
                "repo_id": slot.repo_id,
                "filename": slot.filename,
                "dest": str(path),
                "dest_rel": slot.dest_rel,
                "present": _usable(path),
            }
        )
    return rows


def missing_slots(group: str | None = None, root: Path | None = None) -> list[dict[str, Any]]:
    return [row for row in scan_slots(group, root) if not row["present"]]


def _missing_error(group: str) -> MissingHeartMuLaWeights:
    missing = missing_slots(group)
    names = ", ".join(row["dest_rel"] for row in missing[:8])
    extra = ""
    if len(missing) > 8:
        extra = f" (+{len(missing) - 8} more)"
    return MissingHeartMuLaWeights(
        f"HeartMuLa {group} weights missing ({len(missing)}): {names}{extra}. "
        "python -m master_agent download-models --heartmula   "
        "# list only; add --yes after you agree. Nothing was downloaded."
    )


def assert_generate_weights() -> None:
    if missing_slots("generate"):
        raise _missing_error("generate")


def assert_transcribe_weights() -> None:
    if missing_slots("transcribe"):
        raise _missing_error("transcribe")


def heartlib_installed() -> bool:
    return importlib.util.find_spec("heartlib") is not None


def _row(name: str, ok: bool, detail: str, fix: str = "") -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail, "fix": fix}


def check_heartlib() -> dict[str, Any]:
    if heartlib_installed():
        return _row("heartlib", True, "optional package importable (soft-import)")
    return _row(
        "heartlib",
        False,
        "optional package not installed (soft-import; master_agent still imports)",
        fix=(
            "git clone https://github.com/HeartMuLa/heartlib && pip install -e . "
            "(Buddy does not install it)"
        ),
    )


def check_heartmula_weights() -> dict[str, Any]:
    try:
        rows = scan_slots()
    except HeartMuLaConfigError as exc:
        return _row(
            "heartmula-weights",
            False,
            str(exc),
            fix="fix HEARTMULA_*_REPO to an attested id; nothing was downloaded",
        )
    present = sum(1 for row in rows if row["present"])
    total = len(rows)
    if present == total:
        return _row(
            "heartmula-weights",
            True,
            f"{present}/{total} attested files under {models_dir()}",
        )
    return _row(
        "heartmula-weights",
        False,
        f"{present}/{total} attested files; confirmed-missing listed by "
        "download-models --heartmula (no fetch)",
        fix="python -m master_agent download-models --heartmula",
    )


def check_heartmula_comfy() -> dict[str, Any]:
    """Report tower-registered classes if this machine's object_info cache has them."""
    from master_agent.config import OBJECT_INFO_CACHE

    wanted = set(COMFY_CLASSES)
    found: set[str] = set()
    if OBJECT_INFO_CACHE.is_file():
        try:
            import json

            data = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                found = wanted.intersection(data)
        except (OSError, json.JSONDecodeError):
            found = set()
    if found == wanted:
        return _row(
            "heartmula-comfy",
            True,
            "cached object_info has HeartMuLa_Generate and HeartMuLa_Transcribe",
        )
    return _row(
        "heartmula-comfy",
        False,
        "cached object_info is missing HeartMuLa_Generate / HeartMuLa_Transcribe. "
        f"Tower Comfy Desk registered both ({COMFY_COMMIT}, display names "
        "HeartMuLa Music Generator and HeartMuLa Lyrics Transcriber). "
        "This row does not fetch or install the pack.",
        fix="on the tower: python -m master_agent fetch-object-info",
    )


def heartmula_doctor_rows() -> list[dict[str, Any]]:
    return [check_heartlib(), check_heartmula_weights(), check_heartmula_comfy()]


def format_slot_report(rows: list[dict[str, Any]] | None = None) -> str:
    chosen = rows if rows is not None else scan_slots()
    missing = [row for row in chosen if not row["present"]]
    lines = [
        f"HeartMuLa slots: {len(chosen) - len(missing)}/{len(chosen)} present "
        f"under {models_dir()}",
        TOWER_FREE_NOTE,
    ]
    if not missing:
        lines.append("OK    all attested HeartMuLa files present")
        return "\n".join(lines)
    lines.append(f"NEED  {len(missing)} confirmed-missing (not downloaded):")
    for row in missing:
        lines.append(f"  - {row['repo_id']} {row['filename']} → {row['dest_rel']}")
    lines.append("Nothing downloaded. Re-run with --yes after you agree.")
    lines.append("  python -m master_agent download-models --heartmula --yes")
    return "\n".join(lines)


def download_missing_slots(
    *,
    yes: bool,
    progress: Callable[[str], None] = print,
    root: Path | None = None,
) -> list[Path]:
    """Copy Hub files into the heartlib layout. Refuses when ``yes`` is false."""
    if not yes:
        raise RuntimeError(
            "download-models --heartmula refuses to fetch without --yes"
        )
    base = Path(root) if root is not None else models_dir()
    missing = missing_slots(root=base)
    if not missing:
        progress("OK    nothing to download")
        return []
    from huggingface_hub import hf_hub_download

    written: list[Path] = []
    for row in missing:
        dest = base / row["dest_rel"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        progress(f"Downloading {row['repo_id']}/{row['filename']} ...")
        cached = hf_hub_download(repo_id=row["repo_id"], filename=row["filename"])
        shutil.copyfile(cached, dest)
        progress(f"OK   {dest}")
        written.append(dest)
    return written


def cmd_download_heartmula(args: Any) -> int:
    """CLI body for ``download-models --heartmula``."""
    try:
        report = format_slot_report()
    except HeartMuLaConfigError as exc:
        print(f"FAIL  {exc}")
        return 1
    print(report)
    missing = missing_slots()
    if getattr(args, "scan_only", False):
        print("\n--scan-only: nothing downloaded.")
        return 0 if not missing else 2
    if getattr(args, "use_existing", False):
        print("\n--use-existing: keeping files already on disk. Nothing downloaded.")
        return 0
    if not missing:
        return 0
    allow = bool(getattr(args, "yes", False))
    if not allow and not getattr(args, "download", False):
        print("\nNothing downloaded. Re-run with --yes after you agree, or pass --download to confirm.")
        return 2
    if not allow:
        from master_agent.setup import confirm_prompt

        if not confirm_prompt("Download the confirmed-missing HeartMuLa set? [y/N] "):
            print("\nNothing downloaded.")
            return 2
    try:
        paths = download_missing_slots(yes=True)
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1
    print(f"OK    {len(paths)} file(s) downloaded")
    return 0


def attested_repo_ids() -> tuple[str, ...]:
    return (DEFAULT_MULA_REPO, DEFAULT_CODEC_REPO, DEFAULT_TRANSCRIPTOR_REPO)
