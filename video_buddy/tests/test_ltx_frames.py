"""LTX 8n+1 frame law (Rainey stop-line). No GPU.

Run: python -m pytest tests/test_ltx_frames.py -q
"""

from __future__ import annotations

from master_agent.comfy.validator import validate_workflow
from master_agent.comfy.workflow_patcher import _heuristic_patch
from master_agent.config import (
    DEFAULT_FRAMES,
    DIAGNOSE_FRAMES,
    DOWNSCALE_LADDER,
    QUALITY_PROFILES,
    is_valid_ltx_frames,
    snap_ltx_frames,
)


LTX_OBJECT_INFO = {
    "EmptyLTXVLatentVideo": {
        "input": {
            "required": {
                "width": ["INT", {}],
                "height": ["INT", {}],
                "length": ["INT", {}],
            }
        },
        "output": ["LATENT"],
    },
    "LTXVEmptyLatentAudio": {
        "input": {
            "required": {
                "frames_number": ["INT", {}],
            }
        },
        "output": ["LATENT"],
    },
}


def _ltx_pair(length: int, audio: int | None = None) -> dict:
    audio_frames = length if audio is None else audio
    return {
        "1": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 768, "height": 512, "length": length},
        },
        "2": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {"frames_number": audio_frames},
        },
    }


def test_snap_ltx_frames_table():
    assert snap_ltx_frames(8) == 9
    assert snap_ltx_frames(9) == 9
    assert snap_ltx_frames(16) == 17
    assert snap_ltx_frames(120) == 121
    assert snap_ltx_frames(1) == 9
    assert snap_ltx_frames(121) == 121
    assert not is_valid_ltx_frames(8)
    assert is_valid_ltx_frames(9)
    assert is_valid_ltx_frames(17)


def test_defaults_are_rainey_safe():
    assert DEFAULT_FRAMES == 9
    assert DIAGNOSE_FRAMES == 9
    assert QUALITY_PROFILES["draft"]["frames"] == 9
    assert DOWNSCALE_LADDER[0][2] != 121
    for _w, _h, frames in DOWNSCALE_LADDER:
        assert is_valid_ltx_frames(frames)
        assert frames < 121 or frames == snap_ltx_frames(frames)


def test_patcher_snaps_and_pairs_audio():
    wf = _ltx_pair(8, 8)
    _heuristic_patch(wf, {"frames": 8, "fps": 24})
    assert wf["1"]["inputs"]["length"] == 9
    assert wf["2"]["inputs"]["frames_number"] == 9

    wf16 = _ltx_pair(1, 1)
    _heuristic_patch(wf16, {"frames": 16, "fps": 24})
    assert wf16["1"]["inputs"]["length"] == 17
    assert wf16["2"]["inputs"]["frames_number"] == 17

    wf9 = _ltx_pair(9, 9)
    _heuristic_patch(wf9, {"frames": 9, "fps": 24})
    assert wf9["1"]["inputs"]["length"] == 9
    assert wf9["2"]["inputs"]["frames_number"] == 9

    wf120 = _ltx_pair(120, 8)
    _heuristic_patch(wf120, {"frames": 120, "fps": 24})
    assert wf120["1"]["inputs"]["length"] == 121
    assert wf120["2"]["inputs"]["frames_number"] == 121


def test_validator_autocorrects_length_8_unless_strict():
    wf = _ltx_pair(8, 8)
    report = validate_workflow(wf, LTX_OBJECT_INFO, file_label="pair", strict=False)
    assert report.ok
    assert report.warnings
    assert wf["1"]["inputs"]["length"] == 9
    assert wf["2"]["inputs"]["frames_number"] == 9
    assert is_valid_ltx_frames(wf["1"]["inputs"]["length"])

    strict_wf = _ltx_pair(8, 8)
    strict = validate_workflow(
        strict_wf, LTX_OBJECT_INFO, file_label="strict", strict=True
    )
    assert not strict.ok
    assert any("8n+1" in str(err) or "length=8" in str(err) for err in strict.errors)
    # strict does not rewrite; length=8 cannot queue
    assert strict_wf["1"]["inputs"]["length"] == 8
