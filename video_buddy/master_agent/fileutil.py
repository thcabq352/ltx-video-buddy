"""Atomic writes and a cross-process file lock for shared state files.

Several Buddy processes (MCP children, the studio, CLI runs) can touch the
same ``state/`` files. Writers go through ``atomic_write_text`` so a reader
never sees a half-written file, and read-modify-write cycles hold
``file_lock`` so two processes cannot interleave.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to a sibling temp file, fsync, then ``os.replace`` it in."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _try_lock(fh) -> bool:
    if os.name == "nt":
        import msvcrt

        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fh) -> None:
    if os.name == "nt":
        import msvcrt

        with contextlib.suppress(OSError):
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    with contextlib.suppress(OSError):
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def file_lock(path: Path | str, *, timeout_s: float = 30.0, blocking: bool = True) -> Iterator[bool]:
    """Hold an exclusive lock on ``<path>.lock``.

    Yields True when the lock is held. With ``blocking=False`` it yields
    False immediately if another process holds it. A blocking wait that
    exceeds ``timeout_s`` raises ``TimeoutError``.
    """
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    held = False
    try:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            held = _try_lock(fh)
            if held or not blocking:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {lock_path}")
            time.sleep(0.05)
        yield held
    finally:
        if held:
            _unlock(fh)
        fh.close()
