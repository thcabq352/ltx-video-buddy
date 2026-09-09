"""Workflow linter — live registry validation as a hard gate before render."""

from __future__ import annotations

from typing import Any

from master_agent.comfy.validator import ValidationReport, validate_workflow


class LintBlocked(RuntimeError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(f"workflow linter blocked queue ({len(report.errors)} error(s))")


def lint_workflow(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
    *,
    file_label: str = "<workflow>",
) -> ValidationReport:
    return validate_workflow(workflow, object_info, file_label=file_label)


def hard_gate(report: ValidationReport) -> ValidationReport:
    if not report.ok:
        raise LintBlocked(report)
    return report
