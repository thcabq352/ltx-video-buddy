"""Buddy owns the render handoff. Agents write a brief file; this process runs it.

The process is started detached (see :mod:`master_agent.process_detach`), so
closing the terminal does not kill the job. It is not a thread.

Brief JSON::

    {"args": {"request": "...", "duration_s": 5}, "result": "/path/to/result.json"}

Tower (24/7 Facebook and Instagram studio) and the RAINEY1 laptop fleet
(12 GB RTX 4000 Ada) use the same handoff. The seat rule inside
``create_video`` unloads the LLM before ComfyUI is queued.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def write_brief(path: Path, args: dict[str, Any], *, result: Path | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    result_path = result or path.with_suffix(".result.json")
    payload = {"args": dict(args), "result": str(result_path)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_brief(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    args = payload.get("args") if isinstance(payload, dict) else None
    if not isinstance(args, dict):
        raise ValueError(f"brief {path} has no args object")
    result_path = Path(str(payload.get("result") or Path(path).with_suffix(".result.json")))
    from master_agent.agent_api import create_video

    try:
        result = create_video(**args)
    except TypeError as exc:
        result = {"status": "error", "error": f"brief args were not accepted: {exc}"}
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        result = {"status": "error", "error": "create_video did not return an object"}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m master_agent.detached_job BRIEF.json", file=sys.stderr)
        return 2
    try:
        result = run_brief(Path(args[0]))
    except Exception as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    status = str(result.get("status") or "error")
    if status in {"done", "done_with_warnings", "dry-run", "paused", "busy"}:
        return 0
    print(result.get("error") or status, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
