"""Mocked unit tests for the storyboard LLM panel (no live providers).

Run: .venv/Scripts/python.exe -m unittest tests.test_llm_panel -v
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from master_agent.llm_panel import PanelCandidate, resolve_panel
from master_agent.storyboard.storyboard import build_storyboard_panel


def _board(title: str) -> str:
    return json.dumps(
        {
            "shots": [
                {
                    "index": 0,
                    "title": f"{title} A",
                    "duration_s": 5.0,
                    "ltx_prompt": f"{title} shot A prompt",
                },
                {
                    "index": 1,
                    "title": f"{title} B",
                    "duration_s": 5.0,
                    "ltx_prompt": f"{title} shot B prompt",
                },
            ],
            "global_style": f"{title} style",
        }
    )


def _cand(provider: str, text: str = "", error: str = "") -> PanelCandidate:
    return PanelCandidate(provider=provider, text=text, latency_s=0.1, error=error)


class TestResolvePanel(unittest.TestCase):
    def test_skips_unavailable(self):
        with patch("master_agent.llm_panel.provider_available") as avail:
            avail.side_effect = lambda spec: not spec.startswith("ollama:gemma")
            res = resolve_panel("ollama:qwen3.6-27b-fable,ollama:gemma4:latest")
        self.assertEqual(res.members, ["ollama:qwen3.6-27b-fable"])
        self.assertEqual(res.skipped, [("ollama:gemma4:latest", "provider unavailable")])

    def test_custom_csv(self):
        with patch("master_agent.llm_panel.provider_available", return_value=True):
            res = resolve_panel("kimi,grok")
        self.assertEqual(res.members, ["kimi", "grok"])


class TestBuildStoryboardPanel(unittest.TestCase):
    def test_zero_valid_falls_back_to_heuristic(self):
        cands = [_cand("a", error="boom"), _cand("b", text="not json at all")]
        with patch("master_agent.llm_panel.panel_complete", return_value=cands), patch(
            "master_agent.storyboard.storyboard._panel_judge_call"
        ) as judge:
            cards, style, meta = build_storyboard_panel(
                "coffee ad", [5.0, 5.0], members=["a", "b"]
            )
        self.assertFalse(judge.called)
        self.assertEqual(len(cards), 2)
        self.assertIn("heuristic", meta["fallback"])
        self.assertIsNone(meta["winner"])

    def test_one_valid_short_circuits_judge(self):
        cands = [_cand("a", text=_board("A")), _cand("b", error="timeout")]
        with patch("master_agent.llm_panel.panel_complete", return_value=cands), patch(
            "master_agent.storyboard.storyboard._panel_judge_call"
        ) as judge:
            cards, style, meta = build_storyboard_panel(
                "coffee ad", [5.0, 5.0], members=["a", "b"]
            )
        self.assertFalse(judge.called)
        self.assertEqual(meta["winner"], "a")
        self.assertEqual(style, "A style")
        self.assertEqual([c.title for c in cards], ["A A", "A B"])

    def test_judge_picks_winner(self):
        cands = [_cand("a", text=_board("A")), _cand("b", text=_board("B"))]
        with patch("master_agent.llm_panel.panel_complete", return_value=cands), patch(
            "master_agent.storyboard.storyboard._panel_judge_call",
            return_value={"winner": 1, "reason": "better continuity"},
        ):
            cards, style, meta = build_storyboard_panel(
                "coffee ad", [5.0, 5.0], members=["a", "b"], judge_provider="ollama"
            )
        self.assertEqual(meta["winner"], "b")
        self.assertEqual(meta["judge"], "ollama")
        self.assertEqual(meta["judge_reason"], "better continuity")
        self.assertEqual([c.title for c in cards], ["B A", "B B"])

    def test_judge_merge_honored(self):
        merged = [
            {"title": "merged 1", "ltx_prompt": "m1"},
            {"title": "merged 2", "ltx_prompt": "m2"},
        ]
        cands = [_cand("a", text=_board("A")), _cand("b", text=_board("B"))]
        with patch("master_agent.llm_panel.panel_complete", return_value=cands), patch(
            "master_agent.storyboard.storyboard._panel_judge_call",
            return_value={"winner": 0, "reason": "blend", "merged_shots": merged},
        ):
            cards, _style, meta = build_storyboard_panel(
                "coffee ad", [5.0, 5.0], members=["a", "b"]
            )
        self.assertTrue(meta.get("merged"))
        self.assertEqual([c.title for c in cards], ["merged 1", "merged 2"])

    def test_judge_failure_uses_first_valid(self):
        cands = [_cand("a", text=_board("A")), _cand("b", text=_board("B"))]
        with patch("master_agent.llm_panel.panel_complete", return_value=cands), patch(
            "master_agent.storyboard.storyboard._panel_judge_call", return_value=None
        ):
            cards, _style, meta = build_storyboard_panel(
                "coffee ad", [5.0, 5.0], members=["a", "b"]
            )
        self.assertEqual(meta["winner"], "a")
        self.assertIn("judge unavailable", meta["judge_reason"])
        self.assertEqual([c.title for c in cards], ["A A", "A B"])


if __name__ == "__main__":
    unittest.main()
