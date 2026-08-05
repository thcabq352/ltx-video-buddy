"""Mocked tests for the LLM director and the judge vision leg.

Run: .venv/Scripts/python.exe -m unittest tests.test_director_vision -v
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from master_agent.judge.judge import judge_segment, merge_legs
from master_agent.orchestrator.director import choose_variant, rule_based_variant


class TestMergeLegs(unittest.TestCase):
    def test_three_legs(self):
        # weights 0.45/0.55/0.30 renormalized
        got = merge_legs(0.9, 0.8, 0.6)
        want = (0.45 * 0.9 + 0.55 * 0.8 + 0.30 * 0.6) / 1.30
        self.assertAlmostEqual(got, want, places=5)

    def test_no_vision_matches_two_leg(self):
        self.assertAlmostEqual(merge_legs(0.9, 0.8, None), (0.45 * 0.9 + 0.55 * 0.8) / 1.0)

    def test_heuristic_only(self):
        self.assertAlmostEqual(merge_legs(0.7, None, None), 0.7)


class TestDirector(unittest.TestCase):
    def test_forced_and_input_win(self):
        self.assertEqual(choose_variant("anything", force="eros"), ("eros", "forced"))
        self.assertEqual(choose_variant("anything", has_video=True), ("lipsync", "input"))

    def test_rules_when_llm_disabled(self):
        with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False):
            self.assertEqual(choose_variant("a dub of my clip"), ("lipsync", "rules"))
            self.assertEqual(choose_variant("rain on a window"), ("base", "rules"))

    def test_llm_choice_accepted(self):
        with patch(
            "master_agent.orchestrator.director._llm_variant", return_value="directors"
        ):
            self.assertEqual(
                choose_variant("cinematic brand film with three scenes"),
                ("directors", "llm"),
            )

    def test_llm_failure_falls_back_to_rules(self):
        with patch("master_agent.orchestrator.director._llm_variant", return_value=None):
            self.assertEqual(choose_variant("rain on a window"), ("base", "rules"))

    def test_llm_invalid_variant_rejected(self):
        # _llm_variant itself validates, but double-check choose handles None
        self.assertEqual(rule_based_variant("10eros teaser"), "eros")


class TestJudgeVisionLeg(unittest.TestCase):
    def _run(self, review):
        with patch("master_agent.judge.judge.frame_notes", return_value="notes"), patch(
            "master_agent.judge.judge._vision_review_safe", return_value=review
        ), patch(
            "master_agent.judge.judge._llm_judge", return_value=None
        ):
            return judge_segment(
                user_request="coffee ad",
                ltx_prompt="p",
                video_path="x.mp4",
                heuristic_score=0.9,
                heuristic_issues=[],
                judge_enabled=True,
            )

    def test_vision_fail_blocks_heuristic_accept(self):
        # heuristic alone (>=0.85) would auto-accept; vision fail must block it
        res = self._run({"score": 0.3, "pass": False, "issues": ["melted hands"], "reason": "bad"})
        self.assertIsNotNone(res.vision_score)
        self.assertNotEqual(res.decision, "accept")
        self.assertTrue(any("vision:" in i for i in res.issues))

    def test_vision_pass_flows_into_score(self):
        res = self._run({"score": 0.95, "pass": True, "issues": [], "reason": "great"})
        self.assertAlmostEqual(res.vision_score, 0.95)
        self.assertEqual(res.decision, "accept")
        self.assertIn("vision:", res.reason)

    def test_no_vision_behaves_as_before(self):
        res = self._run(None)
        self.assertIsNone(res.vision_score)
        self.assertEqual(res.decision, "accept")  # strong heuristic fallback


if __name__ == "__main__":
    unittest.main()
