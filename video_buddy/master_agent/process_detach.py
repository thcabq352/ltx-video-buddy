"""Start a child that outlives the terminal.

Windows uses ``DETACHED_PROCESS`` and ``CREATE_NEW_PROCESS_GROUP``. POSIX uses
``start_new_session`` (``setsid``), which is what ``nohup`` achieves for
SIGHUP. The child is not a thread. Stdout and stderr go to a log file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def detached_popen_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        flags = int(getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
        flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
        return {"creationflags": flags, "close_fds": True}
    return {"start_new_session": True, "close_fds": True}


def spawn_detached(
    argv: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    popen=None,
) -> Any:
    """Spawn ``argv`` detached. ``popen`` replaces :func:`subprocess.Popen` in tests."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    launcher = popen or subprocess.Popen
    try:
        return launcher(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            cwd=str(cwd) if cwd is not None else None,
            **detached_popen_kwargs(),
        )
    finally:
        handle.close()


def buddy_python_argv(*args: str) -> list[str]:
    return [sys.executable, *args]
