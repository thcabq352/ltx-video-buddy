"""Append-only history for tunables, plus a stable hash of the active knob set."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TUNABLE_KEYS = (
    "judge_strictness",
    "learning_rate",
    "cost_vram_threshold_gb",
    "render_budget_cap_vram_min",
    "judge_score_threshold",
    "persona",
    "soul",
)
STRING_KEYS = frozenset({"persona", "soul"})

# Running total lives in the config snapshot but is not a knob (no history / hash).
SNAPSHOT_ONLY = ("render_budget_used_vram_min",)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _defaults() -> dict[str, Any]:
    from master_agent.config import (
        COST_VRAM_THRESHOLD_GB,
        JUDGE_SCORE_THRESHOLD,
        JUDGE_STRICTNESS,
        LEARNING_RATE,
        PERSONA,
        RENDER_BUDGET_CAP_VRAM_MIN,
        RENDER_BUDGET_USED_VRAM_MIN,
        SOUL,
    )

    return {
        "judge_strictness": float(JUDGE_STRICTNESS),
        "learning_rate": float(LEARNING_RATE),
        "cost_vram_threshold_gb": float(COST_VRAM_THRESHOLD_GB),
        "render_budget_cap_vram_min": float(RENDER_BUDGET_CAP_VRAM_MIN),
        "judge_score_threshold": float(JUDGE_SCORE_THRESHOLD),
        "render_budget_used_vram_min": float(RENDER_BUDGET_USED_VRAM_MIN),
        "persona": str(PERSONA or "ara").strip().lower(),
        "soul": str(SOUL or "studio").strip().lower(),
    }


def _coerce(key: str, raw: Any) -> Any:
    if key in STRING_KEYS:
        return str(raw).strip().lower()
    return float(raw)


def config_hash(values: dict[str, Any]) -> str:
    knobs = {k: values[k] for k in TUNABLE_KEYS if k in values}
    blob = json.dumps(knobs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def apply_live(values: dict[str, Any]) -> None:
    import master_agent.config as cfg

    if "judge_strictness" in values:
        cfg.JUDGE_STRICTNESS = float(values["judge_strictness"])
    if "learning_rate" in values:
        cfg.LEARNING_RATE = float(values["learning_rate"])
    if "cost_vram_threshold_gb" in values:
        cfg.COST_VRAM_THRESHOLD_GB = float(values["cost_vram_threshold_gb"])
    if "render_budget_cap_vram_min" in values:
        cfg.RENDER_BUDGET_CAP_VRAM_MIN = float(values["render_budget_cap_vram_min"])
    if "judge_score_threshold" in values:
        cfg.JUDGE_SCORE_THRESHOLD = float(values["judge_score_threshold"])
    if "render_budget_used_vram_min" in values:
        cfg.RENDER_BUDGET_USED_VRAM_MIN = float(values["render_budget_used_vram_min"])
    if "persona" in values:
        cfg.PERSONA = str(values["persona"]).strip().lower()
    if "soul" in values:
        cfg.SOUL = str(values["soul"]).strip().lower()


class VersionedConfig:
    def __init__(self, path: Path | None = None, history_path: Path | None = None):
        from master_agent.config import STATE_DIR

        root = STATE_DIR / "control"
        self.path = Path(path) if path else root / "config.json"
        self.history_path = Path(history_path) if history_path else root / "config_history.jsonl"
        self.values = _defaults()
        loaded = self._read_snapshot()
        if loaded:
            self.values.update({k: loaded[k] for k in (*TUNABLE_KEYS, *SNAPSHOT_ONLY) if k in loaded})
        apply_live(self.values)

    def _read_snapshot(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(data, dict) and isinstance(data.get("values"), dict):
            return data["values"]
        return data if isinstance(data, dict) else {}

    def persist(self) -> dict[str, Any]:
        snap = self.snapshot()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
        return snap

    def snapshot(self) -> dict[str, Any]:
        values = dict(self.values)
        return {
            "hash": config_hash(values),
            "values": values,
        }

    def history(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.history_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-max(0, int(limit)) :]

    def set_values(
        self,
        updates: dict[str, Any],
        *,
        session: str,
        sync_budget: bool = False,
    ) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        for key, raw in updates.items():
            if key not in TUNABLE_KEYS:
                continue
            new = _coerce(key, raw)
            old = self.values.get(key)
            if old == new:
                continue
            entry = {
                "ts": _now(),
                "key": key,
                "old": old,
                "new": new,
                "session": session,
            }
            self.values[key] = new
            changes.append(entry)
        if not changes:
            apply_live(self.values)
            self.persist()
            return []
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as fh:
            for entry in changes:
                fh.write(json.dumps(entry, default=str) + "\n")
        apply_live(self.values)
        self.persist()
        if sync_budget and any(c["key"] == "render_budget_cap_vram_min" for c in changes):
            try:
                from master_agent.control.budget import get_project_budget

                get_project_budget().set_cap(float(self.values["render_budget_cap_vram_min"]))
            except (OSError, TypeError, ValueError):
                pass
        return changes

    def sync_used(self, used: float, cap: float | None = None) -> None:
        self.values["render_budget_used_vram_min"] = round(float(used), 4)
        if cap is not None:
            self.values["render_budget_cap_vram_min"] = float(cap)
        apply_live(self.values)
        self.persist()


_STORE: VersionedConfig | None = None


def get_versioned_config(*, reset: bool = False) -> VersionedConfig:
    global _STORE
    if reset or _STORE is None:
        _STORE = VersionedConfig()
    return _STORE


def sync_budget_used(used: float, cap: float | None = None) -> None:
    if _STORE is None:
        return
    _STORE.sync_used(used, cap)


def announce_config(store: VersionedConfig | None = None) -> str:
    snap = (store or get_versioned_config()).snapshot()
    vals = snap["values"]
    bits = " ".join(f"{k}={vals[k]}" for k in TUNABLE_KEYS if k in vals)
    return f"config_hash={snap['hash']} {bits}"
