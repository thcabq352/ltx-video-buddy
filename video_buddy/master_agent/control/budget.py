"""Project render budget: cumulative VRAM-minutes with a pause-and-review gate."""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from master_agent.control.cost import estimate_cost
from master_agent.fileutil import atomic_write_text, file_lock


def vram_minutes(cost: dict[str, Any]) -> float:
    if cost.get("vram_min") is not None:
        return round(float(cost["vram_min"]), 4)
    return round(float(cost.get("vram_gb") or 0.0) * float(cost.get("time_s") or 0.0) / 60.0, 4)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RenderBudget:
    """Track cap + running total. Holding a scene pauses the rest of the queue."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        cap: float | None = None,
        used: float | None = None,
        ephemeral: bool = False,
    ):
        from master_agent.config import (
            RENDER_BUDGET_CAP_VRAM_MIN,
            RENDER_BUDGET_USED_VRAM_MIN,
            STATE_DIR,
        )

        self.path = None if ephemeral else (Path(path) if path else STATE_DIR / "control" / "budget.json")
        self.cap = float(RENDER_BUDGET_CAP_VRAM_MIN)
        self.used = float(RENDER_BUDGET_USED_VRAM_MIN)
        self.paused = False
        self.pending: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []
        self.shift_id = uuid.uuid4().hex[:12]
        self.shift_started_at = _now()
        if self.path is not None:
            self._load_from_disk()
        if cap is not None:
            self.cap = float(cap)
        if used is not None:
            self.used = float(used)
        if self.path is not None and (cap is not None or used is not None):
            self.persist()

    def _read(self) -> dict[str, Any] | None:
        """Return the ledger dict, ``{}`` when absent, or None when unreadable."""
        if self.path is None or not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _load_from_disk(self) -> None:
        data = self._read()
        if data is None:
            self._quarantine_unreadable()
            return
        if "cap" in data:
            self.cap = float(data["cap"])
        if "used" in data:
            self.used = float(data["used"])
        if "paused" in data:
            self.paused = bool(data["paused"])
        if "pending" in data:
            self.pending = list(data.get("pending") or [])
        if "log" in data:
            self.log = list(data.get("log") or [])
        if data.get("shift_id"):
            self.shift_id = str(data["shift_id"])
        if data.get("shift_started_at"):
            self.shift_started_at = str(data["shift_started_at"])

    def _quarantine_unreadable(self) -> None:
        # An unreadable ledger must not silently become an empty one: keep the
        # bytes and hold the queue until someone reviews the budget.
        backup = None
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
            shutil.copy2(self.path, backup)
        except OSError:
            backup = None
        print(
            f"[budget] {self.path} is unreadable; kept a copy at {backup} and paused the queue for review",
            file=sys.stderr,
        )
        self.paused = True
        self.log.append(
            {
                "ts": _now(),
                "event": "ledger_unreadable",
                "backup": str(backup) if backup else None,
            }
        )

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        """Lock the ledger, reload what other processes wrote, then save."""
        if self.path is None:
            yield
            return
        with file_lock(self.path):
            self._load_from_disk()
            yield
            self._write()
        self._sync_config()

    def snapshot(self) -> dict[str, Any]:
        return {
            "cap": round(self.cap, 4),
            "used": round(self.used, 4),
            "paused": self.paused,
            "pending": list(self.pending),
            "log": list(self.log),
            "shift_id": self.shift_id,
            "shift_started_at": self.shift_started_at,
        }

    def _write(self) -> None:
        atomic_write_text(self.path, json.dumps(self.snapshot(), indent=1) + "\n")

    def _sync_config(self) -> None:
        try:
            from master_agent.control.versioned_config import sync_budget_used

            sync_budget_used(self.used, self.cap)
        except (OSError, TypeError, ValueError):
            pass

    def persist(self) -> None:
        """Save this instance's state as-is (no reload of other writers)."""
        if self.path is None:
            return
        with file_lock(self.path):
            self._write()
        self._sync_config()

    def set_cap(self, cap: float) -> None:
        with self._transaction():
            self.cap = max(0.0, float(cap))
            if self.used < self.cap:
                self.paused = False

    def reset_used(self) -> None:
        with self._transaction():
            self.used = 0.0
            self.paused = False
            self.pending = []

    def reset_shift(self) -> dict[str, Any]:
        """Archive the current shift, then zero used. Never wipe the ledger."""
        with self._transaction():
            row = {
                "ts": _now(),
                "event": "shift_reset",
                "previous_used": round(self.used, 4),
                "previous_shift_id": self.shift_id,
            }
            self.log.append(row)
            self.used = 0.0
            self.paused = False
            self.pending = []
            self.shift_id = uuid.uuid4().hex[:12]
            self.shift_started_at = _now()
        return row

    def consider(
        self,
        scene_id: str,
        cost: dict[str, Any],
        *,
        commit: bool = True,
        charge: bool = True,
    ) -> dict[str, Any]:
        if not commit:
            return self._decide(scene_id, cost, commit=False, charge=charge)
        with self._transaction():
            return self._decide(scene_id, cost, commit=True, charge=charge)

    def _decide(
        self,
        scene_id: str,
        cost: dict[str, Any],
        *,
        commit: bool,
        charge: bool,
    ) -> dict[str, Any]:
        add = vram_minutes(cost)
        used_before = self.used
        would = used_before + add
        hold = bool(charge) and (self.paused or would > self.cap)
        decision = "hold" if hold else "admit"
        used_after = used_before if (hold or not charge) else would
        row = {
            "ts": _now(),
            "scene_id": scene_id,
            "decision": decision,
            "cost_vram_min": add,
            "used_before": round(used_before, 4),
            "used_after": round(used_after, 4),
            "cap": round(self.cap, 4),
            "paused": hold,
            "variant": cost.get("variant"),
            "frames": cost.get("frames"),
            "charged": bool(charge) and decision == "admit",
        }
        if commit:
            if charge and hold:
                self.paused = True
                pending = {
                    "id": scene_id,
                    "vram_min": add,
                    "variant": cost.get("variant"),
                    "frames": cost.get("frames"),
                    "vram_gb": cost.get("vram_gb"),
                    "time_s": cost.get("time_s"),
                }
                if not any(p.get("id") == scene_id for p in self.pending):
                    self.pending.append(pending)
            elif charge:
                self.used = used_after
            self.log.append(row)
        return row

    def apply_queue(
        self,
        scenes: list[dict[str, Any]],
        *,
        commit: bool = True,
        charge: bool = True,
    ) -> dict[str, Any]:
        admitted: list[dict[str, Any]] = []
        held: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for scene in scenes:
            scene_id = str(scene.get("id") or scene.get("scene_id") or "")
            cost = scene.get("cost") or estimate_cost(
                str(scene.get("variant") or "base"),
                int(scene.get("frames") or 81),
            )
            row = self.consider(scene_id, cost, commit=commit, charge=charge)
            decisions.append(row)
            tagged = dict(scene)
            tagged["budget"] = row
            if row["decision"] == "admit":
                admitted.append(tagged)
            else:
                held.append(tagged)
        return {
            "admitted": admitted,
            "held": held,
            "paused": self.paused if commit else any(d["decision"] == "hold" for d in decisions),
            "used": round(self.used, 4),
            "cap": round(self.cap, 4),
            "decisions": decisions,
        }


_BUDGET: RenderBudget | None = None


def get_project_budget(*, path: Path | None = None, reset: bool = False) -> RenderBudget:
    global _BUDGET
    if reset or _BUDGET is None:
        _BUDGET = RenderBudget(path=path)
    return _BUDGET


def frames_for_scene(duration_s: float, variant: str | None = None) -> int:
    from master_agent.config import frames_for_duration, get_variant_gen

    gen = get_variant_gen(variant)
    return frames_for_duration(
        duration_s,
        fps=int(gen["fps"]),
        snap=int(gen["frame_snap"]),
        variant=variant,
    )


def admit_scene(
    scene_id: str,
    *,
    variant: str | None,
    duration_s: float,
    budget: RenderBudget | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    from master_agent.config import COST_VRAM_THRESHOLD_GB

    frames = frames_for_scene(duration_s, variant)
    cost = estimate_cost(variant or "base", frames, threshold_vram_gb=COST_VRAM_THRESHOLD_GB)
    store = budget or get_project_budget()
    row = store.consider(scene_id, cost, commit=commit)
    row["cost"] = cost
    return row
