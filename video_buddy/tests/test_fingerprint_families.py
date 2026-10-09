"""match_family on the shipped graphs: no false lipsync / sulphur routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from master_agent.comfy.ingest.fingerprint import match_family

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"

EXPECTED = {
    "lipsync_ia2v.json": "lipsync",
    "ltx23_inoutpaint_api.json": "inoutpaint",
    "ltx-2.5/LTX-2.5_ICLoRA_Inpaint_Outpaint_Two_Stage_Distilled_api.json": "inoutpaint",
    "sulphur/ltx23_i2v_base_api.json": "sulphur",
    "sulphur/ltx23_t2v_base_api.json": "sulphur",
    "sulphur/ltx23_t2v_distilled_api.json": "sulphur",
    "ltx-2.5/LTX-2.5_MSR_Multi_Reference_api.json": None,
    "ltx-2.5/LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled_api.json": None,
    "ltx-2.5/LTX-2.5_A2V_Two_Stage_Distilled_api.json": None,
    "260507_VIDEO-BUDDY_MOVIE-BUILDER_1-1_ADV_api.json": None,
    "260603_LTX2-3_3D-RENDERING_LIP-SYNC_v08_api.json": None,
}


@pytest.mark.parametrize("rel, family", sorted(EXPECTED.items()))
def test_shipped_graph_family(rel, family):
    path = WORKFLOWS / rel
    if not path.is_file():
        pytest.skip(f"{rel} not in this checkout")
    assert match_family(json.loads(path.read_text(encoding="utf-8"))) == family


def test_ic_lora_guide_without_audio_is_not_lipsync():
    graph = {
        "1": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {}},
        "2": {"class_type": "LTXVSetAudioRefTokens", "inputs": {}},
    }
    assert match_family(graph) is None


def test_sage_attention_alone_is_not_sulphur():
    graph = {
        "1": {"class_type": "PathchSageAttentionKJ", "inputs": {}},
        "2": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "movie"}},
    }
    assert match_family(graph) is None
