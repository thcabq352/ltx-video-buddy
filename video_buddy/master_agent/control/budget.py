"""Project render budget: cumulative VRAM-minutes with a pause-and-review gate."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from master_agent.control.cost import estimate_cost


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
        loaded = {} if ephemeral or self.path is None else self._read()
        self.cap = float(cap if cap is not None else loaded.get("cap", RENDER_BUDGET_CAP_VRAM_MIN))
        self.used = float(used if used is not None else loaded.get("used", RENDER_BUDGET_USED_VRAM_MIN))
        self.paused = bool(loaded.get("paused", False))
        self.pending: list[dict[str, Any]] = list(loaded.get("pending") or [])
        self.log: list[dict[str, Any]] = list(loaded.get("log") or [])
        self.shift_id = str(loaded.get("shift_id") or uuid.uuid4().hex[:12])
        self.shift_started_at = str(loaded.get("shift_started_at") or _now())

    def _read(self) -> dict[str, Any]:
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

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

    def persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.snapshot(), indent=1) + "\n", encoding="utf-8")
        try:
            from master_agent.control.versioned_config import sync_budget_used

            sync_budget_used(self.used, self.cap)
        except (OSError, TypeError, ValueError):
            pass

    def set_cap(self, cap: float) -> None:
        self.cap = max(0.0, float(cap))
        if self.used < self.cap:
            self.paused = False
        self.persist()

    def reset_used(self) -> None:
        self.used = 0.0
        self.paused = False
        self.pending = []
        self.persist()

    def reset_shift(self) -> dict[str, Any]:
        """Archive the current shift, then zero used. Never wipe the ledger."""
        row = {
            "ts": _now(),
            "event": "shift_reset",
            "previous_used": round(self.used, 4),
            "previous_shift_id": self.shift_id,
        }
        self.log.append(row)
        self.used = 0.0
        self.paused = False
        self.shift_id = uuid.uuid4().hex[:12]
        self.shift_started_at = _now()
        self.persist()
        return row

    def consider(
        self,
        scene_id: str,
        cost: dict[str, Any],
        *,
        commit: bool = True,
        charge: bool = True,
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
            self.persist()
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
    return frames_for_duration(duration_s, fps=int(gen["fps"]), snap=int(gen["frame_snap"]))


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
