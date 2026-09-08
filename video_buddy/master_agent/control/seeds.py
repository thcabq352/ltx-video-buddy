"""Seed ledger: every seed with prompt version, score, and variant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def record_seed(
    ledger: Path,
    *,
    seed: int,
    prompt_version: str,
    score: float,
    variant: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "seed": int(seed),
        "prompt_version": prompt_version,
        "score": float(score),
        "variant": variant,
        **(extra or {}),
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def load_ledger(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.is_file():
        return []
    rows = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def replay(ledger: Path, seed: int) -> dict[str, Any] | None:
    for row in reversed(load_ledger(ledger)):
        if int(row.get("seed") or -1) == int(seed):
            return row
    return None
