"""Uniqueness gate: refuse stitch if any two window clips share a content hash."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from master_agent.provenance import sha256_file


class DuplicateClipError(RuntimeError):
    """Two or more windows burned the same bytes — MTV cut would repeat a clip."""

    def __init__(self, groups: list[dict]):
        self.groups = groups
        bits: list[str] = []
        for group in groups:
            idxs = ", ".join(str(i) for i in group["indices"])
            digest = str(group["hash"])[:16]
            bits.append(f"windows [{idxs}] share hash {digest}")
        super().__init__("refuse stitch: duplicate window clips: " + "; ".join(bits))


def check_unique_clips(
    clips: Iterable[tuple[int, str | Path]],
) -> dict[int, str]:
    """Hash every window clip. Raise ``DuplicateClipError`` listing dup indices.

    Returns ``{window_index: sha256}`` when every clip is unique and present.
    """
    by_hash: dict[str, list[int]] = defaultdict(list)
    hashes: dict[int, str] = {}
    missing: list[int] = []
    for raw_idx, raw_path in clips:
        idx = int(raw_idx)
        digest = sha256_file(raw_path)
        if not digest:
            missing.append(idx)
            continue
        hashes[idx] = digest
        by_hash[digest].append(idx)
    if missing:
        raise DuplicateClipError(
            [{"hash": "missing", "indices": missing}]
        )
    dups = [
        {"hash": digest, "indices": sorted(idxs)}
        for digest, idxs in by_hash.items()
        if len(idxs) > 1
    ]
    if dups:
        dups.sort(key=lambda g: g["indices"][0])
        raise DuplicateClipError(dups)
    return hashes
