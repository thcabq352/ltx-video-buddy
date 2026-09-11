"""Fleet curriculum gates. No GPU.

Run: python -m pytest tests/test_curriculum.py -q
"""

from master_agent.curriculum import (
    PART1_ID,
    PART1_LESSONS,
    PART2_ID,
    curriculum_card,
    format_curriculum,
    part2_unlocked,
)


def test_part1_is_l0_through_l5():
    ids = [row["id"] for row in PART1_LESSONS]
    assert ids == ["L0", "L1", "L2", "L3", "L4", "L5"]
    names = [row["name"] for row in PART1_LESSONS]
    assert names == ["tree", "about", "health", "dry-run", "speed diagnose", "short proof"]
    card = curriculum_card()
    assert card["part1"] == PART1_ID == "LESSON_BUDDY_WORKS_HERE"
    assert card["part2"] == PART2_ID == "LESSON_BUDDY_PART2_OVERNIGHT"


def test_part2_requires_l5_and_human_ok():
    assert part2_unlocked(part1_l5=True, human_ok=True) is True
    assert part2_unlocked(part1_l5=True, human_ok=False) is False
    assert part2_unlocked(part1_l5=False, human_ok=True) is False
    text = format_curriculum()
    assert "L5" in text
    assert "human" in text.lower()
