"""Managed llama.cpp server (``llama-server``).

Buddy starts it when ``LLM_PROVIDER`` is ``auto`` (the default) or
``llamacpp``, pointed at ``MODELS_DIR`` (default ``video_buddy/models/``),
and stops that process on shutdown. An already-listening ``LLAMACPP_URL``
is left alone. A missing binary warns and returns; it does not raise.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from master_agent import config as cfg

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_proc: subprocess.Popen | None = None
_atexit_done = False
_failure_at = 0.0
_warned: set[str] = set()
_COOLDOWN_S = 30.0
_START_TIMEOUT_S = 8.0


def reset_state() -> None:
    """Drop owned-process bookkeeping without signaling it. Tests only."""
    global _proc, _failure_at
    _proc = None
    _failure_at = 0.0
    _warned.clear()


def resolve_binary() -> str | None:
    """``LLAMACPP_BIN`` if it exists, else ``llama-server`` on ``PATH``, else the
    one automatic install put under ``LLAMACPP_ROOT``."""
    explicit = (cfg.LLAMACPP_BIN or "").strip().strip('"')
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        return shutil.which(explicit)
    on_path = shutil.which("llama-server")
    if on_path:
        return on_path
    from master_agent.llamacpp_install import buddy_binary

    mine = buddy_binary(getattr(cfg, "LLAMACPP_ROOT", None))
    return str(mine) if mine else None


def endpoint_parts(url: str | None = None) -> tuple[str, int, str]:
    """Return ``(bind_host, port, connect_host)`` for ``LLAMACPP_URL``."""
    raw = (url if url is not None else cfg.LLAMACPP_URL) or "http://127.0.0.1:8080"
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    bind = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 8080)
    connect = "127.0.0.1" if bind in ("0.0.0.0", "::") else bind
    return bind, port, connect


def server_argv(
    binary: str,
    *,
    host: str,
    port: int,
    models_dir: Path,
) -> list[str]:
    """``llama-server`` router mode pointed at the central models directory."""
    return [
        binary,
        "--host",
        host,
        "--port",
        str(port),
        "--models-dir",
        str(models_dir),
    ]


def _tcp_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _listening(connect_host: str, port: int) -> bool:
    """Cached reachability (tests patch this path) or a live TCP connect."""
    from master_agent.llm import endpoint_up, reset_endpoint_cache

    if endpoint_up(f"http://{connect_host}:{port}", port):
        return True
    if _tcp_open(connect_host, port):
        reset_endpoint_cache()
        return True
    return False


def _warn(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    log.warning("%s", message)
    print(f"WARNING: {message}", file=sys.stderr)


def _fall_through(flag: bool | None) -> bool:
    if flag is not None:
        return flag
    from master_agent.llm import normalize_provider_name

    return normalize_provider_name(cfg.LLM_PROVIDER) != "llamacpp"


def _missing_message(fall_through: bool) -> str:
    base = (
        "llama.cpp binary not found "
        "(set LLAMACPP_BIN, install llama-server on PATH, or run automatic install). "
    )
    if fall_through:
        return base + "Falling through to Ollama."
    return base + "LLM_PROVIDER=llamacpp stays on llama.cpp."


def _popen(argv: list[str]) -> subprocess.Popen:
    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(argv, **kwargs)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                return
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def _register_atexit() -> None:
    global _atexit_done
    if _atexit_done:
        return
    atexit.register(stop)
    _atexit_done = True


def _wait_until_listening(
    host: str, port: int, proc: subprocess.Popen, timeout_s: float = _START_TIMEOUT_S
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if _tcp_open(host, port):
            return True
        time.sleep(0.2)
    return proc.poll() is None and _tcp_open(host, port)


def ensure_started(*, fall_through: bool | None = None) -> str:
    """Start llama.cpp when nothing is listening. Never raises.

    Returns ``external`` (already up), ``started``, ``missing``, or ``failed``.
    """
    with _LOCK:
        return _ensure(fall_through)


def _ensure(fall_through: bool | None) -> str:
    global _proc, _failure_at

    from master_agent.llm import reset_endpoint_cache

    bind, port, connect = endpoint_parts()
    hop = _fall_through(fall_through)
    owned = _proc is not None and _proc.poll() is None
    if owned and _listening(connect, port):
        reset_endpoint_cache()
        return "started"
    if not owned and _listening(connect, port):
        reset_endpoint_cache()
        return "external"
    if _failure_at and time.time() - _failure_at < _COOLDOWN_S:
        return "failed"

    binary = resolve_binary()
    if not binary:
        _warn("missing", _missing_message(hop))
        return "missing"

    argv = server_argv(binary, host=bind, port=port, models_dir=Path(cfg.MODELS_DIR))
    try:
        proc = _popen(argv)
    except OSError as exc:
        _failure_at = time.time()
        tail = "Falling through to Ollama." if hop else "Not switching backends."
        _warn("start", f"llama.cpp failed to start ({exc}). {tail}")
        return "failed"

    _proc = proc
    _register_atexit()
    if _wait_until_listening(connect, port, proc):
        reset_endpoint_cache()
        return "started"

    code = proc.poll()
    _terminate(proc)
    _proc = None
    _failure_at = time.time()
    tail = "Falling through to Ollama." if hop else "Not switching backends."
    _warn(
        "listen",
        f"llama.cpp server exited before listening on {connect}:{port} "
        f"(code {code}). {tail}",
    )
    return "failed"


def stop() -> None:
    """Stop the llama.cpp process Buddy started. External servers stay up."""
    global _proc
    with _LOCK:
        proc = _proc
        _proc = None
    if proc is None:
        return
    _terminate(proc)
