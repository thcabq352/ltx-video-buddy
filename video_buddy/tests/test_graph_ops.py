"""Unit tests for Comfy graph ops + power mode (mocked LLM, no GPU)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from master_agent.comfy.graph_ops import (
    apply_ops,
    object_info_snippets,
    summarize_workflow,
)
from master_agent.comfy.power_mode import power_tune
from master_agent.comfy.validator import ValidationReport


def _tiny_workflow():
    return {
        "1": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1,
                "steps": 8,
                "cfg": 1.5,
                "sampler_name": "euler",
                "model": ["2", 0],
            },
        },
        "2": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "model.safetensors"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a cat", "clip": ["2", 1]},
        },
    }


def _tiny_object_info():
    return {
        "KSampler": {
            "input": {
                "required": {
                    "seed": ["INT", {"default": 0, "min": 0, "max": 2**32 - 1}],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}],
                    "sampler_name": ["COMBO", {"options": ["euler", "dpmpp_2m"]}],
                    "model": ["MODEL", {}],
                }
            },
            "output": ["LATENT"],
        },
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": ["COMBO", {}]}},
            "output": ["MODEL", "CLIP", "VAE"],
        },
        "CLIPTextEncode": {
            "input": {
                "required": {
                    "text": ["STRING", {"multiline": True}],
                    "clip": ["CLIP", {}],
                }
            },
            "output": ["CONDITIONING"],
        },
    }


class TestGraphOps(unittest.TestCase):
    def test_set_widget(self):
        wf, res = apply_ops(
            _tiny_workflow(),
            [{"op": "set_widget", "node_id": "1", "input": "steps", "value": 20}],
        )
        self.assertTrue(res.ok)
        self.assertEqual(wf["1"]["inputs"]["steps"], 20)

    def test_set_widget_by_class(self):
        wf, res = apply_ops(
            _tiny_workflow(),
            [
                {
                    "op": "set_widget_by_class",
                    "class_type": "KSampler",
                    "input": "cfg",
                    "value": 2.5,
                    "index": 0,
                }
            ],
        )
        self.assertTrue(res.ok)
        self.assertEqual(wf["1"]["inputs"]["cfg"], 2.5)

    def test_rewire(self):
        wf, res = apply_ops(
            _tiny_workflow(),
            [
                {
                    "op": "rewire",
                    "node_id": "3",
                    "input": "clip",
                    "from_node": "2",
                    "from_slot": 1,
                }
            ],
        )
        self.assertTrue(res.ok)
        self.assertEqual(wf["3"]["inputs"]["clip"], ["2", 1])

    def test_add_and_remove_node(self):
        wf, res = apply_ops(
            _tiny_workflow(),
            [
                {
                    "op": "add_node",
                    "node_id": "9",
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "neg", "clip": ["2", 1]},
                }
            ],
        )
        self.assertIn("9", wf)
        wf2, res2 = apply_ops(wf, [{"op": "remove_node", "node_id": "9"}], copy_graph=True)
        self.assertTrue(res2.ok)
        self.assertNotIn("9", wf2)

    def test_unknown_op_skipped(self):
        _, res = apply_ops(_tiny_workflow(), [{"op": "explode_graph"}])
        self.assertTrue(res.ok)  # skipped, not hard fail
        self.assertTrue(res.skipped)

    def test_summarize_and_snippets(self):
        s = summarize_workflow(_tiny_workflow())
        self.assertIn("KSampler", s)
        snip = object_info_snippets(_tiny_workflow(), _tiny_object_info())
        self.assertIn("steps", snip)
        self.assertIn("KSampler", snip)


class TestPowerMode(unittest.TestCase):
    def test_power_tune_applies_and_validates(self):
        ops = [
            {
                "op": "set_widget",
                "node_id": "1",
                "input": "steps",
                "value": 16,
            }
        ]

        class FakeResp:
            content = '{"reason":"more steps","ops":' + __import__("json").dumps(ops) + "}"

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = FakeResp()

        ok_report = ValidationReport(file="t", object_info_source="test")
        # empty errors => ok

        with patch("master_agent.comfy.power_mode.get_llm", create=True), patch(
            "master_agent.llm.get_llm", return_value=fake_llm
        ), patch(
            "master_agent.comfy.power_mode.validate_workflow", return_value=ok_report
        ), patch(
            "master_agent.comfy.power_mode._rag_context", return_value=("", False)
        ):
            result = power_tune(
                _tiny_workflow(),
                request="cinematic coffee ad",
                object_info=_tiny_object_info(),
                repair=False,
                log=lambda *_: None,
            )
        self.assertTrue(result.valid)
        self.assertEqual(result.workflow["1"]["inputs"]["steps"], 16)
        self.assertEqual(len(result.applied), 1)
        self.assertIn("steps", result.reason or "more steps")

    def test_power_tune_keeps_base_when_invalid(self):
        ops = [{"op": "set_widget", "node_id": "1", "input": "steps", "value": 99}]

        class FakeResp:
            content = '{"reason":"bad","ops":' + __import__("json").dumps(ops) + "}"

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = FakeResp()

        bad = ValidationReport(file="t", object_info_source="test")
        bad.error("1", "steps", "too high")

        with patch("master_agent.llm.get_llm", return_value=fake_llm), patch(
            "master_agent.comfy.power_mode.validate_workflow", return_value=bad
        ), patch(
            "master_agent.comfy.power_mode._rag_context", return_value=("", False)
        ):
            base = _tiny_workflow()
            result = power_tune(
                base,
                request="x",
                object_info=_tiny_object_info(),
                repair=False,
                log=lambda *_: None,
            )
        self.assertFalse(result.valid)
        # original steps preserved on returned workflow
        self.assertEqual(result.workflow["1"]["inputs"]["steps"], 8)


if __name__ == "__main__":
    unittest.main()
