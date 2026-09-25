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
