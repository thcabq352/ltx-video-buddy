"""Fleet curriculum: LESSON_BUDDY_WORKS_HERE then duration ladder Part 2.

Source: Briefing for LTX Video Buddy (master_agent) — fleet curriculum &
field notes (2026-09-10). Part 2 is gated on Part 1 L5 + human OK.

Part 2 is a duration ladder from 1s to 10s. Frame law: valid lengths are
8n+1, minimum 9, at the default 24 fps. So 25 frames ≈ 1.04s, 49 ≈ 2.04s,
…, 241 ≈ 10.04s. Each rung adds exactly one new concept.
"""

from __future__ import annotations

from typing import Any

PART1_ID = "LESSON_BUDDY_WORKS_HERE"
PART2_ID = "LESSON_BUDDY_DURATION_LADDER"

PART1_LESSONS: tuple[dict[str, str], ...] = (
    {"id": "L0", "name": "tree", "do": "Know this tree: video_buddy / master_agent, not sibling studios."},
    {"id": "L1", "name": "about", "do": "python -m master_agent about — studio card before any GPU claim."},
    {"id": "L2", "name": "health", "do": "python -m master_agent health — Comfy :8188 up. Studio :8189 is not proof."},
    {"id": "L3", "name": "dry-run", "do": "python -m master_agent run \"BRIEF\" --dry-run — plan/lint only, no queue, no budget spend."},
    {"id": "L4", "name": "speed diagnose", "do": "python -m master_agent diagnose --variant base --prompt \"garden proof\" — 9-frame hull, print sec/step."},
    {"id": "L5", "name": "short proof", "do": "Short 9-frame proof in outputs/; ffprobe frames + size. Junk <100KB or <3 frames is FAIL."},
)

# Duration ladder: 1s → 10s. frames = 8n+1. One new concept per rung.
PART2_LESSONS: tuple[dict[str, Any], ...] = (
    {
        "id": "D1",
        "seconds": 1,
        "frames": 25,
        "concept": "single object, one simple motion",
        "prompt": "a red ball rolls across a wooden table",
    },
    {
        "id": "D2",
        "seconds": 2,
        "frames": 49,
        "concept": "add a second element",
        "prompt": "the ball rolls, then a hand catches it",
    },
    {
        "id": "D3",
        "seconds": 3,
        "frames": 73,
        "concept": "introduce camera movement",
        "prompt": "camera slowly pans right as the ball rolls",
    },
    {
        "id": "D4",
        "seconds": 4,
        "frames": 97,
        "concept": "add lighting change",
        "prompt": "the ball rolls under a shifting spotlight",
    },
    {
        "id": "D5",
        "seconds": 5,
        "frames": 121,
        "concept": "two subjects interacting",
        "prompt": "a cat watches the ball, then bats it away",
    },
    {
        "id": "D6",
        "seconds": 6,
        "frames": 145,
        "concept": "environment interaction",
        "prompt": "the ball bounces off a wall and rolls back",
    },
    {
        "id": "D7",
        "seconds": 7,
        "frames": 169,
        "concept": "weather or particle effect",
        "prompt": "rain falls as the ball rolls through a puddle",
    },
    {
        "id": "D8",
        "seconds": 8,
        "frames": 193,
        "concept": "complex motion with timing",
        "prompt": "the ball rolls, stops, then accelerates toward the camera",
    },
    {
        "id": "D9",
        "seconds": 9,
        "frames": 217,
        "concept": "multi-element scene",
        "prompt": "ball rolls, hand catches it, camera pulls back to reveal a room",
    },
    {
        "id": "D10",
        "seconds": 10,
        "frames": 241,
        "concept": "full scene with subject, background, and camera",
        "prompt": "a red ball rolls across a wooden table in a sunlit room, camera slowly orbits",
    },
)


def part2_unlocked(*, part1_l5: bool, human_ok: bool) -> bool:
    """Duration ladder stays gated until Part 1 L5 and a human OK."""
    return bool(part1_l5) and bool(human_ok)


def curriculum_card() -> dict[str, Any]:
    return {
        "part1": PART1_ID,
        "part2": PART2_ID,
        "part2_gate": "Part 1 L5 + human OK",
        "lessons": [dict(row) for row in PART1_LESSONS],
        "duration_ladder": [dict(row) for row in PART2_LESSONS],
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
    lines.append("Part 2 duration ladder (1s → 10s, 8n+1 frames):")
    for row in data["duration_ladder"]:
        lines.append(
            f"  {row['id']}  {row['seconds']:>2}s  {row['frames']:>3}f  {row['concept']}"
        )
        lines.append(f"         prompt: {row['prompt']}")
    lines.append("")
    lines.append("Do not start Part 2 until L5 proof exists and a human says OK.")
    return "\n".join(lines)
