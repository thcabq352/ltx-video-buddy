"""Workflow linter hard gate. Run: .venv/Scripts/python.exe -m pytest tests/test_linter.py -q"""

import pytest

from master_agent.comfy.linter import LintBlocked, hard_gate, lint_workflow
from master_agent.comfy.validator import ValidationReport


def test_linter_blocks_unknown_class():
    wf = {"1": {"class_type": "NotANode", "inputs": {}}}
    report = lint_workflow(wf, object_info={}, file_label="bad.json")
    assert report.ok is False
    with pytest.raises(LintBlocked):
        hard_gate(report)
    clean = ValidationReport(file="ok")
    assert hard_gate(clean) is clean


def test_v3_combo_and_lipsync_placeholder():
    from master_agent.comfy.validator import validate_workflow
    from master_agent.comfy.workflow_patcher import load_workflow_template

    info = {
        "LoadVideo": {
            "input": {
                "required": {
                    "file": ["COMBO", {"options": ["a.mp4", "warehouse_src_30fps.mp4"]}],
                }
            },
            "output": ["VIDEO"],
        }
    }
    bad = {
        "1": {
            "class_type": "LoadVideo",
            "inputs": {"file": "other.mp4"},
        }
    }
    report = validate_workflow(bad, info, file_label="combo")
    assert any("not in combo choices" in err.message for err in report.errors)

    lipsync = load_workflow_template("lipsync")
    held = validate_workflow(lipsync, info, file_label="lipsync")
    assert any("warehouse_src_30fps.mp4" in err.message for err in held.errors)


def test_autogrow_slots_and_combo_choices_alias():
    """Dotted autogrow slots use the template type; V3 COMBO accepts `choices`."""
    import copy

    from master_agent.comfy.validator import validate_workflow

    autogrow = [
        "COMFY_AUTOGROW_V3",
        {
            "template": {
                "input": {"optional": {"image": ["IMAGE", {}]}},
                "prefix": "image",
                "min": 0,
                "max": 4,
            }
        },
    ]
    info = {
        "LoadImage": {
            "input": {"required": {"image": ["STRING", {}]}},
            "output": ["IMAGE"],
        },
        "Sink": {
            "input": {
                "required": {
                    "imgs": autogrow,
                    "mode": ["COMBO", {"choices": ["a", "b"]}],
                }
            },
            "output": ["*"],
        },
    }
    ok = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "2": {
            "class_type": "Sink",
            "inputs": {"imgs.image0": ["1", 0], "mode": "a"},
        },
    }
    report = validate_workflow(ok, info, file_label="autogrow")
    assert report.ok, report.errors

    audio = copy.deepcopy(info)
    audio["LoadImage"]["output"] = ["AUDIO"]
    mismatched = validate_workflow(ok, audio, file_label="autogrow-type")
    assert any("type mismatch" in err.message and "IMAGE" in err.message for err in mismatched.errors)

    bare = copy.deepcopy(ok)
    bare["2"]["inputs"]["imgs"] = ["1", 0]
    del bare["2"]["inputs"]["imgs.image0"]
    bare_report = validate_workflow(bare, info, file_label="autogrow-bare")
    assert any("COMFY_AUTOGROW_V3" in err.message for err in bare_report.errors)

    bad_combo = copy.deepcopy(ok)
    bad_combo["2"]["inputs"]["mode"] = "nope"
    combo_report = validate_workflow(bad_combo, info, file_label="combo-alias")
    assert any("not in combo choices" in err.message for err in combo_report.errors)


def test_dynamic_combo_requires_prefixed_children():
    """COMFY_DYNAMICCOMBO_V3 children live under options, as parent.child."""
    from master_agent.comfy.validator import validate_workflow

    resize = [
        "COMFY_DYNAMICCOMBO_V3",
        {
            "options": [
                {
                    "key": "scale dimensions",
                    "inputs": {
                        "required": {
                            "width": ["INT", {"min": 0, "max": 4096}],
                            "height": ["INT", {"min": 0, "max": 4096}],
                            "crop": ["COMBO", {"options": ["disabled", "center"]}],
                        }
                    },
                },
                {
                    "key": "match size",
                    "inputs": {
                        "required": {
                            "match": ["IMAGE,MASK", {}],
                            "crop": ["COMBO", {"options": ["disabled", "center"]}],
                        }
                    },
                },
            ]
        },
    ]
    info = {
        "Src": {"input": {"required": {}}, "output": ["IMAGE"]},
        "ResizeImageMaskNode": {
            "input": {
                "required": {
                    "input": ["IMAGE", {}],
                    "resize_type": resize,
                    "scale_method": ["COMBO", {"options": ["area", "lanczos"]}],
                }
            },
            "output": ["IMAGE"],
        },
    }
    plain = {
        "1": {"class_type": "Src", "inputs": {}},
        "2": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["1", 0],
                "resize_type": "scale dimensions",
                "scale_method": "area",
                "width": 768,
                "height": 448,
            },
        },
    }
    report = validate_workflow(plain, info, file_label="plain-size")
    missing = {err.input_name for err in report.errors}
    assert missing == {"resize_type.width", "resize_type.height", "resize_type.crop"}

    prefixed = {
        "1": {"class_type": "Src", "inputs": {}},
        "2": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["1", 0],
                "resize_type": "scale dimensions",
                "scale_method": "lanczos",
                "resize_type.width": 384,
                "resize_type.height": 224,
                "resize_type.crop": "disabled",
            },
        },
    }
    ok = validate_workflow(prefixed, info, file_label="prefixed")
    assert ok.ok, ok.errors

    matched = {
        "1": {"class_type": "Src", "inputs": {}},
        "2": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["1", 0],
                "resize_type": "match size",
                "scale_method": "area",
                "resize_type.match": ["1", 0],
                "resize_type.crop": "center",
            },
        },
    }
    match_report = validate_workflow(matched, info, file_label="match")
    assert match_report.ok, match_report.errors
