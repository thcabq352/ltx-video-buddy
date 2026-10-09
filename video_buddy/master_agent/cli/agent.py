"""``python -m master_agent agent`` — JSON front end over ``agent_api``.

stdout carries exactly one JSON document so the gateway can parse it;
pipeline logs go to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import sys
from pathlib import Path


def _load_args(raw: str | None) -> dict:
    if not raw:
        return {}
    text = Path(raw[1:]).read_text(encoding="utf-8") if raw.startswith("@") else raw
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("--args must be a JSON object")
    return data


def _catalog() -> list[dict]:
    from master_agent.agent_api import TOOLS

    rows = []
    for name, fn in TOOLS.items():
        doc = inspect.getdoc(fn) or ""
        rows.append(
            {
                "tool": name,
                "summary": doc.split("\n\n")[0].replace("\n", " "),
                "params": list(inspect.signature(fn).parameters),
            }
        )
    return rows


def cmd_agent(args: argparse.Namespace) -> int:
    from master_agent.agent_api import TOOLS, call

    out = sys.stdout
    if args.tool == "list":
        print(json.dumps(_catalog(), indent=1), file=out)
        return 0
    if args.tool not in TOOLS:
        print(json.dumps({"status": "error", "error": f"unknown tool: {args.tool}"}), file=out)
        return 2
    try:
        kwargs = _load_args(args.args)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": f"bad --args: {exc}"}), file=out)
        return 2
    try:
        inspect.signature(TOOLS[args.tool]).bind(**kwargs)
    except TypeError as exc:
        print(json.dumps({"status": "error", "error": f"{args.tool}: {exc}"}), file=out)
        return 2
    with contextlib.redirect_stdout(sys.stderr):
        result = call(args.tool, kwargs)
    print(json.dumps(result, indent=1, default=str), file=out)
    if isinstance(result, dict) and result.get("status") in ("error", "busy"):
        return 1
    return 0
