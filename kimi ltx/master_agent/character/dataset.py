"""LoRA training dataset — kept sheet images + [trigger] captions + character.json.

Reads the sheet.json manifest written by ``sheet.generate_character_sheet``,
copies kept images into ``CHARACTERS_DIR/<name>/dataset/`` with same-stem
``.txt`` captions ("[trigger], {appearance}" — ai-toolkit replaces the literal
``[trigger]``), and writes ``character.json``. Best-effort KB log at the end.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from master_agent.config import CHARACTERS_DIR


def _log_to_kb(name: str, summary: dict[str, Any]) -> None:
    """Best-effort KB record; never fails the stage."""
    try:
        from master_agent.kb.store import COLLECTION_CHARACTERS, kb_available, upsert_docs

        if not kb_available():
            return
        text = (
            f"character {name}: trigger={summary.get('trigger_word')} "
            f"dataset_count={summary.get('dataset_count')} ok={summary.get('ok')}\n"
            f"appearance: {summary.get('appearance', '')[:300]}"
        )
        upsert_docs(
            COLLECTION_CHARACTERS,
            ids=[f"character:{name}"],
            texts=[text],
            metadatas=[
                {
                    "name": name,
                    "dataset_count": int(summary.get("dataset_count") or 0),
                    "ok": bool(summary.get("ok")),
                    "timestamp": summary.get("created") or "",
                }
            ],
        )
    except Exception:
        pass


def build_dataset(name: str, min_images: int = 10) -> dict:
    """Assemble the ai-toolkit dataset for character ``name``.

    Returns a summary dict; ``ok`` is False (with ``reason``) when fewer than
    ``min_images`` kept shots exist — whatever exists is still written.
    """
    char_dir = CHARACTERS_DIR / name
    sheet_dir = char_dir / "sheet"
    dataset_dir = char_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {}
    manifest_path = sheet_dir / "sheet.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    kept = [k for k in (manifest.get("kept") or []) if (sheet_dir / k).is_file()]
    if not kept:  # no manifest / stale manifest: use whatever images exist
        kept = [p.name for p in sorted(sheet_dir.glob("*.png"))] if sheet_dir.is_dir() else []

    trigger = str(manifest.get("trigger_word") or f"zxc_{name}")
    appearance = str(manifest.get("appearance") or name)
    caption = f"[trigger], {appearance}"

    copied: list[str] = []
    for fname in kept:
        src = sheet_dir / fname
        dest = dataset_dir / fname
        shutil.copy2(src, dest)
        dest.with_suffix(".txt").write_text(caption, encoding="utf-8")
        copied.append(fname)

    created = datetime.now(timezone.utc).isoformat()
    count = len(copied)
    ok = count >= min_images
    character = {
        "name": name,
        "trigger_word": trigger,
        "appearance": appearance,
        "shots": manifest.get("shots") or [],
        "scores": manifest.get("scores") or {},
        "dataset_count": count,
        "created": created,
    }
    (char_dir / "character.json").write_text(
        json.dumps(character, indent=2), encoding="utf-8"
    )

    summary = {
        "ok": ok,
        "name": name,
        "dataset_dir": str(dataset_dir),
        "dataset_count": count,
        "images": copied,
        "trigger_word": trigger,
        "appearance": appearance,
        "created": created,
        "character_json": str(char_dir / "character.json"),
    }
    if not ok:
        summary["reason"] = f"only {count} kept images (< min_images={min_images})"
    _log_to_kb(name, summary)
    return summary
