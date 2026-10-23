"""Install-time VRAM profile written into ``.env``.

``nvidia-smi`` total memory decides the lines. A value the user already set
is left in place and named in the notes.

- Under 20 GB: ``LLAMACPP_NGL=0`` so llama-server stays on CPU.
- 12 GB class (under 14 GB): also ``VRAM_GB=12`` and ``FORCE_LOADER=gguf``.
- 16 GB and up, still under 20 GB: CPU LLM, and bf16 / EROS fallbacks stay
  available because ``FORCE_LOADER`` is not forced.
- 20 GB and up: LLM may use the GPU when nothing is rendering. The seat
  rule still takes it off the GPU before a ComfyUI submit.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_ASSIGN = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

CPU_BELOW_GB = 20.0
TWELVE_CLASS_BELOW_GB = 14.0


@dataclass
class VramEnvPlan:
    detected_gb: float | None
    updates: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.notes and not self.updates:
            return "no VRAM change"
        parts = list(self.notes)
        if self.updates:
            wrote = ", ".join(f"{k}={v}" for k, v in self.updates.items())
            parts.append(f"write {wrote}")
        return "; ".join(parts)


def read_assignments(text: str) -> dict[str, str]:
    """Uncommented ``KEY=value`` lines. Comments are not user-set values."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGN.match(stripped)
        if match:
            found[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return found


def plan_vram_env(detected_gb: float | None, existing: dict[str, str]) -> VramEnvPlan:
    plan = VramEnvPlan(detected_gb=detected_gb)
    if detected_gb is None:
        plan.notes.append("nvidia-smi did not report VRAM; left LLM and loader lines unchanged")
        return plan
    gb = float(detected_gb)
    if gb < CPU_BELOW_GB:
        _set_or_note(plan, existing, "LLAMACPP_NGL", "0", f"VRAM {gb:.1f} GB is under 20, so the LLM stays on CPU")
    else:
        plan.notes.append(
            f"VRAM {gb:.1f} GB is 20 or more; LLAMACPP_NGL was not forced to 0 "
            "(the seat rule still unloads the LLM before a render)"
        )
    if gb < TWELVE_CLASS_BELOW_GB:
        _set_or_note(plan, existing, "VRAM_GB", "12", "12 GB class card")
        _set_or_note(plan, existing, "FORCE_LOADER", "gguf", "12 GB class card")
    elif gb < CPU_BELOW_GB:
        plan.notes.append(
            "16 GB class: FORCE_LOADER was not set, so a lone bf16 or EROS file can still load. "
            "The LLM and LTX are not on the GPU together."
        )
    return plan


def _set_or_note(plan: VramEnvPlan, existing: dict[str, str], key: str, value: str, why: str) -> None:
    current = existing.get(key)
    if current is None:
        plan.updates[key] = value
        plan.notes.append(f"{why}: set {key}={value}")
        return
    if current == value:
        plan.notes.append(f"{key}={current} already set")
        return
    plan.notes.append(f"left your {key}={current} ({why} would have set {value})")


def upsert_env(path: Path, updates: dict[str, str]) -> None:
    """Append missing keys. Never rewrite a key that is already set."""
    if not updates:
        return
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    existing = read_assignments(text)
    lines = [f"{key}={value}" for key, value in updates.items() if key not in existing]
    if not lines:
        return
    suffix = "" if text.endswith("\n") or text == "" else "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + suffix + "\n".join(lines) + "\n", encoding="utf-8")
    for key, value in updates.items():
        if key not in existing:
            os.environ[key] = value


def apply_vram_env(
    detected_gb: float | None,
    path: Path,
    *,
    apply: bool,
) -> VramEnvPlan:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    plan = plan_vram_env(detected_gb, read_assignments(text))
    if apply:
        upsert_env(path, plan.updates)
    return plan
