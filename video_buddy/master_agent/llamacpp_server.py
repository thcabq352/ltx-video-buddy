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
_last_argv: list[str] = []
_atexit_done = False
_failure_at = 0.0
_warned: set[str] = set()
_COOLDOWN_S = 30.0
_START_TIMEOUT_S = 8.0


def reset_state() -> None:
    """Drop owned-process bookkeeping without signaling it. Tests only."""
    global _proc, _failure_at, _last_argv
    _proc = None
    _failure_at = 0.0
    _last_argv = []
    _warned.clear()


class LocatedLlama:
    """Where ``llama-server`` was found, and a stale ``LLAMACPP_BIN`` if any."""

    def __init__(self, path: str | None, source: str = "", stale: str = "") -> None:
        self.path = path
        self.source = source
        self.stale = stale


def _from_explicit(explicit: str) -> Path | None:
    """A file, ``llama-server`` inside a directory, or ``which`` of the value."""
    path = Path(explicit)
    if path.is_file():
        return path
    if path.is_dir():
        from master_agent.llamacpp_install import server_in_tree

        hit = server_in_tree(path)
        if hit is not None:
            return hit
    found = shutil.which(explicit)
    return Path(found) if found else None


def locate_server() -> LocatedLlama:
    """``LLAMACPP_BIN``, else ``llama-server`` on ``PATH``, else Buddy's install.

    A directory ``LLAMACPP_BIN`` is searched for ``llama-server`` (``.exe`` on
    Windows), including ``build/bin`` and ``build/bin/Release`` under it.
    A value that names nothing is logged and does not hide PATH or the Buddy
    install under ``LLAMACPP_ROOT``.
    """
    explicit = (cfg.LLAMACPP_BIN or "").strip().strip('"')
    stale = ""
    if explicit:
        found = _from_explicit(explicit)
        if found is not None:
            return LocatedLlama(str(found), "LLAMACPP_BIN")
        stale = explicit
        log.warning(
            "LLAMACPP_BIN=%s is not a llama-server; looking on PATH and in the Buddy install",
            explicit,
        )
    on_path = shutil.which("llama-server")
    if on_path:
        return LocatedLlama(on_path, "PATH", stale)
    from master_agent.llamacpp_install import buddy_binary

    mine = buddy_binary(getattr(cfg, "LLAMACPP_ROOT", None))
    if mine is not None:
        return LocatedLlama(str(mine), "Buddy install", stale)
    return LocatedLlama(None, "", stale)


def resolve_binary() -> str | None:
    """Path from :func:`locate_server`, or ``None`` when nothing is installed."""
    return locate_server().path


def describe_located(found: LocatedLlama, root: Path | str | None = None) -> str:
    """Doctor line: which source won, the path, the build, and a stale-bin note."""
    from master_agent.llamacpp_install import NOT_INSTALLED, llama_status_detail

    note = f"LLAMACPP_BIN={found.stale} is not a llama-server" if found.stale else ""
    if not found.path:
        return f"{NOT_INSTALLED}; {note}" if note else NOT_INSTALLED
    text = f"{found.source}: {llama_status_detail(found.path, root)}"
    return f"{text}; {note}" if note else text


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
    """``llama-server`` router mode pointed at the central models directory.

    When automatic install wrote ``state/llamacpp-models.ini``, pass it as
    ``--models-preset`` so the served names (``qwen3-vl-heretic``,
    ``nomic-embed-text``) point at the GGUF files, with ``embedding`` on the
    embed slot. A ``mmproj`` line is how llama-server loads the projector
    (``--mmproj``) for that model, including when the files live in another
    folder such as ``gguf_vl_heretic/``.
    """
    argv = [
        binary,
        "--host",
        host,
        "--port",
        str(port),
        "--models-dir",
        str(models_dir),
    ]
    try:
        from master_agent.models.llm_gguf import preset_path

        preset = preset_path()
        if preset.is_file():
            argv.extend(["--models-preset", str(preset)])
    except Exception:
        pass
    # Absent means llama.cpp's own default (layers on the GPU). Only pin
    # ``-ngl`` when the operator or the VRAM profile set LLAMACPP_NGL.
    raw = (os.getenv("LLAMACPP_NGL") or "").strip()
    if raw:
        try:
            argv = _apply_ngl(argv, int(raw))
        except ValueError:
            pass
    return argv


def _apply_ngl(argv: list[str], ngl: int | None) -> list[str]:
    """Replace any GPU-layer flag. ``None`` leaves the argv without ``-ngl``."""
    cleaned: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
            skip = True
            continue
        if arg.startswith("--n-gpu-layers=") or arg.startswith("--gpu-layers="):
            continue
        cleaned.append(arg)
    if ngl is not None:
        cleaned.extend(["-ngl", str(int(ngl))])
    return cleaned


def owned_running() -> bool:
    return _proc is not None and _proc.poll() is None


def owned_argv() -> list[str]:
    return list(_last_argv)


def external_view() -> dict | None:
    """A llama-server we did not start, or None.

    An open port with no llama-server on the command line is not a GPU LLM.
    """
    _bind, port, connect = endpoint_parts()
    if not _listening(connect, port):
        return None
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        if "llama-server" not in line:
            continue
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        return {"pid": pid, "argv": parts[1].split()}
    return None


def stop_pid(pid: int) -> None:
    """Stop one llama-server. Owned processes go through :func:`stop`."""
    if pid <= 0:
        return
    if _proc is not None and _proc.pid == pid:
        stop()
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        return


def start_with_layers(ngl: int) -> str:
    """Restart the server Buddy owns with an explicit ``-ngl``."""
    global _proc, _last_argv
    stop()
    binary = resolve_binary()
    if not binary:
        raise RuntimeError("llama.cpp binary not found")
    bind, port, _connect = endpoint_parts()
    argv = _apply_ngl(
        server_argv(binary, host=bind, port=port, models_dir=Path(cfg.MODELS_DIR)),
        int(ngl),
    )
    proc = _popen(argv)
    with _LOCK:
        _proc = proc
        _last_argv = list(argv)
    _register_atexit()
    return "started"


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

    Returns ``external`` (already up), ``started``, ``missing``, ``failed``,
    or ``disabled`` (``LLAMACPP_AUTOSTART=0`` and nothing listening).
    """
    with _LOCK:
        return _ensure(fall_through)


def _ensure(fall_through: bool | None) -> str:
    global _proc, _failure_at, _last_argv

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
    if not cfg.LLAMACPP_AUTOSTART:
        return "disabled"
    if _failure_at and time.time() - _failure_at < _COOLDOWN_S:
        return "failed"

    binary = resolve_binary()
    if not binary:
        _warn("missing", _missing_message(hop))
        return "missing"

    argv = server_argv(binary, host=bind, port=port, models_dir=Path(cfg.MODELS_DIR))
    try:
        from master_agent.gpu_seat import llm_must_be_cpu

        if llm_must_be_cpu():
            argv = _apply_ngl(argv, 0)
    except Exception:
        pass
    _last_argv = list(argv)
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
