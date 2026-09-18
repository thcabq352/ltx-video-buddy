"""Hands — live fit against a CapabilityContract.

Brain ranks stories. Hands answers *possible-right-now* from a live /
``object_info`` / inventory snapshot and owns the 8s last-frame chain.

Sibling: your-video-buddy ``Hands.can_fulfill`` → ``FitResult``
(``docs/BRAIN_HANDS.md``, PR #9). This tree is the Python live orchestrator.
Hands may use VRAM / weight inventory internally. Those keys never enter
the contract the brain writes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from master_agent.capability import CapabilityContract

CHAIN_CLIP_S = 8.0

REASON_INSUFFICIENT_VRAM = "insufficient_vram"
REASON_MISSING_WEIGHTS = "missing_weights"
REASON_MISSING_NODES = "missing_nodes"


@dataclass(frozen=True)
class ChainClip:
    index: int
    duration_s: float
    use_last_frame: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "duration_s": float(self.duration_s),
            "use_last_frame": bool(self.use_last_frame),
        }


@dataclass(frozen=True)
class LastFrameChain:
    clips: tuple[ChainClip, ...]
    story_duration_s: float
    clip_s: float = CHAIN_CLIP_S

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_s": float(self.clip_s),
            "count": len(self.clips),
            "story_duration_s": float(self.story_duration_s),
            "clips": [c.to_dict() for c in self.clips],
        }


def extract_last_frame(video_path: str | Path, dest: Path) -> Optional[Path]:
    """Hands-owned last-frame extract for the next I2V burn. None if ffmpeg fails."""
    try:
        from master_agent.video_concat import find_ffmpeg

        ff = find_ffmpeg()
        src = Path(video_path)
        if not ff or not src.is_file():
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        import subprocess

        proc = subprocess.run(
            [ff, "-y", "-sseof", "-0.1", "-i", str(src), "-update", "1", "-q:v", "2", str(dest)],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
            return dest
    except Exception:
        return None
    return None


def plan_last_frame_chain(story_duration_s: float) -> LastFrameChain:
    """Hands-owned split: long stories become 8s burns chained by last frame.

    20s → 3×8s. A story that already fits in one clip stays a single burn.
    """
    total = max(float(story_duration_s), 1.0)
    if total <= CHAIN_CLIP_S:
        clips = (ChainClip(index=0, duration_s=round(total, 4), use_last_frame=False),)
        return LastFrameChain(clips=clips, story_duration_s=total)
    n = max(1, int(math.ceil(total / CHAIN_CLIP_S)))
    clips = tuple(
        ChainClip(index=i, duration_s=CHAIN_CLIP_S, use_last_frame=i > 0)
        for i in range(n)
    )
    return LastFrameChain(clips=clips, story_duration_s=total)


@dataclass
class FitResult:
    """Rust ``FitResult`` shape: fit or reject with a reason."""

    ok: bool
    reason: Optional[str] = None
    detail: str = ""
    chain: Optional[LastFrameChain] = None

    @classmethod
    def fit(cls, chain: Optional[LastFrameChain] = None) -> "FitResult":
        return cls(ok=True, reason=None, detail="", chain=chain)

    @classmethod
    def reject(cls, reason: str, detail: str = "") -> "FitResult":
        return cls(ok=False, reason=reason, detail=detail, chain=None)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "status": "fit" if self.ok else "reject",
            "reason": self.reason,
            "detail": self.detail,
        }
        if self.chain is not None:
            payload["chain"] = self.chain.to_dict()
        return payload


@dataclass
class HandsSnapshot:
    """Injected live/object_info/inventory view. None fields = not probed."""

    vram_free_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    present_weights: Optional[frozenset[str]] = None
    object_info: Optional[dict[str, Any]] = None
    missing_weights: tuple[str, ...] = field(default_factory=tuple)


class Hands(Protocol):
    def can_fulfill(self, contract: CapabilityContract) -> FitResult: ...

    def plan_last_frame_chain(self, story_duration_s: float) -> LastFrameChain: ...


class SnapshotHands:
    """Testable Hands: answers fit only from the injected snapshot."""

    def __init__(self, snapshot: HandsSnapshot | None = None):
        self.snapshot = snapshot or HandsSnapshot()

    def plan_last_frame_chain(self, story_duration_s: float) -> LastFrameChain:
        return plan_last_frame_chain(story_duration_s)

    def can_fulfill(self, contract: CapabilityContract) -> FitResult:
        snap = self.snapshot
        variant = contract.variant or "base"

        if snap.vram_free_gb is not None:
            from master_agent.models.vram_policy import expected_vram_gb

            need = float(expected_vram_gb(variant))
            if need > float(snap.vram_free_gb):
                return FitResult.reject(
                    REASON_INSUFFICIENT_VRAM,
                    f"{variant} needs ~{need:.1f}GB; snapshot free={snap.vram_free_gb:.1f}GB",
                )

        if snap.missing_weights:
            return FitResult.reject(
                REASON_MISSING_WEIGHTS,
                "missing weights: " + ", ".join(snap.missing_weights),
            )

        if snap.present_weights is not None:
            from master_agent.models.vram_policy import workflow_row

            required = workflow_row(variant).default_pack
            if required and required not in snap.present_weights:
                return FitResult.reject(
                    REASON_MISSING_WEIGHTS,
                    f"snapshot inventory missing {required}",
                )

        if snap.object_info is not None:
            missing = _missing_nodes(variant, snap.object_info)
            if missing:
                return FitResult.reject(
                    REASON_MISSING_NODES,
                    "object_info missing: " + ", ".join(missing),
                )

        chain = None
        if float(contract.story_duration_s) > CHAIN_CLIP_S:
            chain = plan_last_frame_chain(contract.story_duration_s)
        return FitResult.fit(chain=chain)


class LiveHands:
    """Probe live Comfy / cache when possible; fail open if nothing is injected."""

    def __init__(
        self,
        snapshot: HandsSnapshot | None = None,
        client: Any = None,
    ):
        self._snapshot = snapshot
        self._client = client

    def plan_last_frame_chain(self, story_duration_s: float) -> LastFrameChain:
        return plan_last_frame_chain(story_duration_s)

    def can_fulfill(self, contract: CapabilityContract) -> FitResult:
        if self._snapshot is not None:
            return SnapshotHands(self._snapshot).can_fulfill(contract)
        if self._client is None:
            chain = None
            if float(contract.story_duration_s) > CHAIN_CLIP_S:
                chain = plan_last_frame_chain(contract.story_duration_s)
            return FitResult.fit(chain=chain)
        return SnapshotHands(probe_live_snapshot(self._client)).can_fulfill(contract)


def default_hands() -> LiveHands:
    return LiveHands()


def probe_live_snapshot(client: Any = None) -> HandsSnapshot:
    """Best-effort *live* Comfy snapshot. Missing probes stay None (fail-open).

    Cached inventory / object_info files are not treated as live — inject
    those on ``SnapshotHands`` when tests or a caller have a real view.
    Uses a short HTTP timeout so a down Comfy never blocks routing.
    """
    vram_free = None
    vram_total = None
    object_info = None
    base = getattr(client, "base_url", None) if client is not None else None
    if not base:
        return HandsSnapshot()
    try:
        import httpx

        with httpx.Client(timeout=0.4) as http:
            try:
                stats = http.get(f"{str(base).rstrip('/')}/system_stats")
                stats.raise_for_status()
                data = stats.json()
                devices = data.get("devices") or data.get("gpus") or []
                if devices:
                    dev = devices[0]
                    free = dev.get("vram_free") or dev.get("vram_free_gb")
                    total = dev.get("vram_total") or dev.get("vram_total_gb")
                    if free is not None:
                        vram_free = float(free) / (1e9 if float(free) > 1000 else 1.0)
                    if total is not None:
                        vram_total = float(total) / (1e9 if float(total) > 1000 else 1.0)
            except Exception:
                pass
            try:
                info = http.get(f"{str(base).rstrip('/')}/object_info")
                info.raise_for_status()
                payload = info.json()
                if isinstance(payload, dict) and payload:
                    object_info = payload
            except Exception:
                pass
    except Exception:
        pass
    return HandsSnapshot(
        vram_free_gb=vram_free,
        vram_total_gb=vram_total,
        present_weights=None,
        object_info=object_info,
    )


def _missing_nodes(variant: str, object_info: dict[str, Any]) -> list[str]:
    """Optional class-type check. Empty object_info means 'probed and empty'."""
    if not object_info:
        return []
    from master_agent.comfy.capabilities import CAPABILITY_CATALOG

    needed: list[str] = []
    for cap in CAPABILITY_CATALOG:
        if variant in cap.templates:
            needed.extend(cap.class_types)
            break
    missing = [name for name in needed if name not in object_info]
    return missing
