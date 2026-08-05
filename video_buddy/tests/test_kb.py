"""Mocked tests for the knowledge base (no live chromadb/ollama).

Run: .venv/Scripts/python.exe -m unittest tests.test_kb -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from master_agent.kb.ingest import _run_digest, _run_metadata, _workflow_digest
from master_agent.kb.recall import recall_similar_runs


class TestWorkflowDigest(unittest.TestCase):
    def test_digest_extracts_classes_and_prompts(self):
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a coffee ad, macro"}},
            "3": {"class_type": "CreateVideo", "inputs": {}},
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "base_t2v_i2v.json"
            p.write_text(json.dumps(wf), encoding="utf-8")
            digest = _workflow_digest(p)
        self.assertIn("workflow base_t2v_i2v", digest)
        self.assertIn("CheckpointLoaderSimple", digest)
        self.assertIn("prompt: a coffee ad, macro", digest)


class TestRunDigest(unittest.TestCase):
    def test_pipeline_record(self):
        rec = {
            "run_id": "abc123",
            "request": "coffee ad",
            "segment_paths": ["a.mp4", "b.mp4"],
            "segment_scores": [0.8, 0.9],
            "full_judge_score": 0.85,
            "full_judge_pass": True,
            "full_judge_notes": "solid pour, weak ice shot",
            "status": "done",
        }
        digest = _run_digest(rec)
        meta = _run_metadata(rec)
        self.assertIn("request: coffee ad", digest)
        self.assertIn("full_judge: score=0.85", digest)
        self.assertEqual(meta["kind"], "pipeline")
        self.assertEqual(meta["score"], 0.85)
        self.assertTrue(meta["passed"])

    def test_orchestrator_record(self):
        rec = {
            "run_id": "def456",
            "request": "rain window",
            "variant": "base",
            "judge_score": 0.7,
            "judge_decision": "rewrite",
            "judge_reason": "too dark",
            "state": "DONE",
        }
        meta = _run_metadata(rec)
        self.assertEqual(meta["kind"], "run")
        self.assertFalse(meta["passed"])
        self.assertIn("judge_reason: too dark", _run_digest(rec))


class TestRecall(unittest.TestCase):
    def test_recall_formats_hits(self):
        hits = [
            {
                "id": "pipeline:abc",
                "text": "request: coffee ad\njudge_reason: weak pour shot",
                "metadata": {"status": "done", "score": 0.85, "request": "coffee ad"},
                "distance": 0.1,
            }
        ]
        with patch("master_agent.kb.recall.search", return_value=hits):
            block = recall_similar_runs("espresso commercial")
        self.assertIn("Similar past runs", block)
        self.assertIn("score=0.85", block)
        self.assertIn("judge_reason: weak pour shot", block)

    def test_recall_empty(self):
        with patch("master_agent.kb.recall.search", return_value=[]):
            self.assertEqual(recall_similar_runs("anything"), "")


if __name__ == "__main__":
    unittest.main()
