"""Fleet curriculum: LESSON_BUDDY_WORKS_HERE then overnight Part 2.

Source: Briefing for LTX Video Buddy (master_agent) — fleet curriculum &
field notes (2026-09-10). Part 2 is gated on Part 1 L5 + human OK.
"""

from __future__ import annotations

from typing import Any

PART1_ID = "LESSON_BUDDY_WORKS_HERE"
PART2_ID = "LESSON_BUDDY_PART2_OVERNIGHT"

PART1_LESSONS: tuple[dict[str, str], ...] = (
    {"id": "L0", "name": "tree", "do": "Know this tree: video_buddy / master_agent, not sibling studios."},
    {"id": "L1", "name": "about", "do": "python -m master_agent about — studio card before any GPU claim."},
    {"id": "L2", "name": "health", "do": "python -m master_agent health — Comfy :8188 up. Studio :8189 is not proof."},
    {"id": "L3", "name": "dry-run", "do": "python -m master_agent run \"BRIEF\" --dry-run — plan/lint only, no queue, no budget spend."},
    {"id": "L4", "name": "speed diagnose", "do": "python -m master_agent diagnose --variant base --prompt \"garden proof\" — 9-frame hull, print sec/step."},
    {"id": "L5", "name": "short proof", "do": "Short 9-frame proof in outputs/; ffprobe frames + size. Junk <100KB or <3 frames is FAIL."},
)


def part2_unlocked(*, part1_l5: bool, human_ok: bool) -> bool:
    """Overnight Part 2 stays gated until Part 1 L5 and a human OK."""
    return bool(part1_l5) and bool(human_ok)


def curriculum_card() -> dict[str, Any]:
    return {
        "part1": PART1_ID,
        "part2": PART2_ID,
        "part2_gate": "Part 1 L5 + human OK",
        "lessons": [dict(row) for row in PART1_LESSONS],
    }


def format_curriculum(card: dict[str, Any] | None = None) -> str:
    data = card or curriculum_card()
    lines = [
        f"Part 1  {data['part1']}  (L0→L5: tree→about→health→dry-run→speed diagnose→short proof)",
        f"Part 2  {data['part2']}  gated on {data['part2_gate']}",
        "",
    ]
    for row in data["lessons"]:
        lines.append(f"  {row['id']}  {row['name']:16} {row['do']}")
    lines.append("")
    lines.append("Do not start Part 2 overnight until L5 proof exists and a human says OK.")
    return "\n".join(lines)
