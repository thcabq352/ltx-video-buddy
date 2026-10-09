"""Run short-lived media tools (ffmpeg / ffprobe) with a timeout and no stdin.

A hung ffmpeg must not pin a render forever, and it must never read the
parent's stdin (under MCP that is the protocol pipe).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

MEDIA_TIMEOUT_S = float(os.getenv("FFMPEG_TIMEOUT_S", "900"))


def run_media(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """``subprocess.run`` with captured output, DEVNULL stdin and a timeout.

    On timeout the child is killed and a CompletedProcess with return code
    124 is returned, so callers keep their existing failure handling.
    """
    timeout = kwargs.pop("timeout", MEDIA_TIMEOUT_S)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    try:
        return subprocess.run(cmd, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        msg = f"{Path(str(cmd[0])).name} timed out after {timeout:.0f}s"
        text = bool(kwargs.get("text") or kwargs.get("encoding") or kwargs.get("universal_newlines"))
        return subprocess.CompletedProcess(
            cmd,
            124,
            stdout="" if text else b"",
            stderr=msg if text else msg.encode(),
        )
