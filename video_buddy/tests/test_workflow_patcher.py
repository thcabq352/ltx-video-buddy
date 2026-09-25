"""Tests for workflow patching (Flux t2i variant + lora node insertion).

Run: .venv/Scripts/python.exe -m pytest tests/test_workflow_patcher.py -q
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from master_agent.comfy.workflow_patcher import (
    _ensure_lora_node,
    _find_nodes_by_class,
    _heuristic_patch,
    load_and_patch_workflow,
    load_workflow_template,
)

# Resolve every weight name so injection is deterministic regardless of
# whether the Flux downloads have finished.
_FAKE_RESOLVE = patch(
    "master_agent.comfy.workflow_patcher.resolve_model_path",
    return_value=Path("fake.safetensors"),
)


class TestFluxVariant(unittest.TestCase):
    def setUp(self):
        _FAKE_RESOLVE.start()
        self.addCleanup(_FAKE_RESOLVE.stop)
        self.workflow, self.meta = load_and_patch_workflow(
            "flux",
            prompt="a red cube on a table",
            width=768,
            height=512,
            seed=123,
            steps=20,
        )

    def _node(self, class_type):
        matches = _find_nodes_by_class(self.workflow, class_type)
        self.assertTrue(matches, f"no {class_type} node in flux workflow")
        return matches[0][1]

    def test_prompt_lands_on_clip_text_encode(self):
        node = self._node("CLIPTextEncode")
        self.assertEqual(node["inputs"]["text"], "a red cube on a table")

    def test_size_lands_on_empty_latent(self):
        node = self._node("EmptyLatentImage")
        self.assertEqual(node["inputs"]["width"], self.meta["width"])
        self.assertEqual(node["inputs"]["height"], self.meta["height"])
        # EmptyLatentImage must not get video frame-count inputs
        self.assertNotIn("length", node["inputs"])

    def test_seed_steps_land_on_ksampler(self):
        node = self._node("KSampler")
        self.assertEqual(node["inputs"]["seed"], 123)
        self.assertEqual(node["inputs"]["steps"], 20)

    def test_clip_names_land_on_dual_clip_loader(self):
        node = self._node("DualCLIPLoader")
        self.assertEqual(node["inputs"]["clip_name1"], "clip_l.safetensors")
        self.assertEqual(node["inputs"]["clip_name2"], "t5xxl_fp8_e4m3fn.safetensors")

    def test_vae_name_lands_on_vae_loader(self):
        node = self._node("VAELoader")
        self.assertEqual(node["inputs"]["vae_name"], "ae.safetensors")

    def test_unet_loader_gets_resolved_checkpoint(self):
        node = self._node("UNETLoader")
        self.assertEqual(self.meta["checkpoint"], "flux1-dev-fp8.safetensors")
        self.assertEqual(node["inputs"]["unet_name"], "flux1-dev-fp8.safetensors")


class TestWan22Variant(unittest.TestCase):
    """wan22: dual-UNET Wan 2.2 graph — no LTX ckpt spray, 4n+1 frames."""

    def setUp(self):
        _FAKE_RESOLVE.start()
        self.addCleanup(_FAKE_RESOLVE.stop)
        self.workflow, self.meta = load_and_patch_workflow(
            "wan22",
            prompt="photorealistic rainy street at dusk",
            duration_s=3.0,
            seed=123,
            steps=10,
        )

    def test_frames_snap_4n_plus_1(self):
        # 3s @ 16fps -> 48 -> 4n+1 = 49
        self.assertEqual(self.meta["frames"], 49)
        self.assertEqual((self.meta["frames"] - 1) % 4, 0)

    def test_dual_unets_get_wan_weights_not_ltx(self):
        self.assertIsNone(self.meta["checkpoint"])
        self.assertEqual(
            self.workflow["106"]["inputs"]["unet_name"],
            "wan\\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        )
        self.assertEqual(
            self.workflow["104"]["inputs"]["unet_name"],
            "wan\\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        )

    def test_steps_cfg_land_on_both_samplers(self):
        for nid in ("168", "169"):
            inputs = self.workflow[nid]["inputs"]
            self.assertEqual(inputs["steps"], 10)
            self.assertEqual(inputs["cfg"], 1.0)

    def test_prompt_and_size_land(self):
        self.assertEqual(
            self.workflow["3"]["inputs"]["text"],
            "photorealistic rainy street at dusk",
        )
        self.assertEqual(self.workflow["5"]["inputs"]["length"], self.meta["frames"])

    def test_vhs_combine_prefix_and_fps(self):
        inputs = self.workflow["194"]["inputs"]
        self.assertEqual(inputs["filename_prefix"], self.meta["filename_prefix"])
        self.assertEqual(inputs["frame_rate"], 16)


class TestEnsureLoraNode(unittest.TestCase):
    def test_inserts_lora_and_rewires_model_link(self):
        wf = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "flux1-dev-fp8.safetensors"},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {"model": ["1", 0]},
            },
        }
        new_id = _ensure_lora_node(wf, "my_lora.safetensors")
        self.assertEqual(new_id, "3")
        lora_node = wf["3"]
        self.assertEqual(lora_node["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(lora_node["inputs"]["lora_name"], "my_lora.safetensors")
        self.assertEqual(lora_node["inputs"]["model"], ["1", 0])
        self.assertEqual(lora_node["inputs"]["strength_model"], 1.0)
        # KSampler model input now points at the lora node
        self.assertEqual(wf["2"]["inputs"]["model"], ["3", 0])

    def test_only_model_output_rewired_on_checkpoint_loader(self):
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "x.safetensors"},
            },
            "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        }
        new_id = _ensure_lora_node(wf, "my_lora.safetensors")
        self.assertIsNotNone(new_id)
        self.assertEqual(wf["2"]["inputs"]["model"], [new_id, 0])
        # CLIP output (index 1) must stay untouched
        self.assertEqual(wf["3"]["inputs"]["clip"], ["1", 1])

    def test_no_duplicate_when_template_has_lora(self):
        wf = load_workflow_template("base")
        self.assertTrue(_find_nodes_by_class(wf, "LoraLoaderModelOnly"))
        before = len(wf)
        self.assertIsNone(_ensure_lora_node(wf, "my_lora.safetensors"))
        self.assertEqual(len(wf), before)

    def test_noop_without_lora_name(self):
        wf = {"1": {"class_type": "UNETLoader", "inputs": {}}}
        self.assertIsNone(_ensure_lora_node(wf, ""))
        self.assertEqual(len(wf), 1)


class TestMissingTemplateHardFail(unittest.TestCase):
    def test_gitignored_slug_raises_named_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_workflow_template("vb_rtx_superres")
        msg = str(ctx.exception)
        self.assertIn("vb_rtx_superres", msg)
        self.assertIn("NVIDIA-RTX-SUPER-RESOLUTION", msg)

    def test_ltx25_t2v_i2v_still_loads(self):
        wf = load_workflow_template("ltx25_t2v_i2v")
        self.assertTrue(any(isinstance(n, dict) and n.get("class_type") for n in wf.values()))


class TestLanPaintKSamplerPatch(unittest.TestCase):
    def test_heuristic_writes_seed_steps_cfg(self):
        wf = {
            "1": {
                "class_type": "LanPaint_KSampler",
                "inputs": {"seed": 0, "steps": 20, "cfg": 7.0},
            }
        }
        _heuristic_patch(wf, {"seed": 99, "steps": 8, "cfg": 1.5})
        inputs = wf["1"]["inputs"]
        self.assertEqual(inputs["seed"], 99)
        self.assertEqual(inputs["steps"], 8)
        self.assertEqual(inputs["cfg"], 1.5)


class TestAudioToVideoClock(unittest.TestCase):
    def test_a2v_duration_audio_and_enhancer_retarget(self):
        with patch(
            "master_agent.comfy.workflow_patcher._text_enhancer_filename",
            return_value=None,
        ):
            wf, meta = load_and_patch_workflow(
                "ltx25_a2v",
                prompt="she speaks",
                seed=1,
                duration_s=3.0,
                audio_start_s=1.5,
                image_name="face.png",
                audio_name="voice.wav",
            )
        self.assertEqual(wf["5512"]["inputs"]["value"], 3.0)
        self.assertEqual(wf["5601"]["inputs"]["value"], 1.5)
        latents = [
            node["inputs"].get("length")
            for node in wf.values()
            if isinstance(node, dict) and node.get("class_type") == "EmptyLTXVLatentVideo"
        ]
        self.assertIn(73, latents)
        images = [
            node["inputs"].get("image")
            for node in wf.values()
            if isinstance(node, dict) and node.get("class_type") == "LoadImage"
        ]
        self.assertIn("face.png", images)
        audios = [
            node["inputs"].get("audio")
            for node in wf.values()
            if isinstance(node, dict) and node.get("class_type") == "LoadAudio"
        ]
        self.assertIn("voice.wav", audios)
        booleans = [
            node["inputs"].get("value")
            for node in wf.values()
            if isinstance(node, dict) and node.get("class_type") == "PrimitiveBoolean"
        ]
        self.assertIn(True, booleans)
        inplace = [
            node
            for node in wf.values()
            if isinstance(node, dict) and node.get("class_type") == "LTXVImgToVideoInplace"
        ]
        self.assertTrue(inplace)
        self.assertFalse(inplace[0]["inputs"].get("bypass"))
        blob = json.dumps(wf)
        self.assertNotIn("gemma4_e2b", blob)
        self.assertEqual(meta["frames"], 73)


if __name__ == "__main__":
    unittest.main()
