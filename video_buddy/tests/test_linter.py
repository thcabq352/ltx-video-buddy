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
