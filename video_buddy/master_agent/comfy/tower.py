"""Managed ComfyUI lifecycle (start / stop / status / restart).

Buddy can own a local ComfyUI process in the folder ``comfy/root.py``
resolves (``MANAGED_COMFY_ROOT``, default ``PROJECT_ROOT/ComfyUI``). The HTTP
client, graph ops, linter, and generate path are unchanged: they keep talking
to whatever answers on ``COMFYUI_URL``.

``start`` runs ``<comfy python> main.py ...`` itself, detached, with stdout and
stderr in ``state/logs/comfyui_<port>.log``, then polls the HTTP port. It does
not use ``comfy launch --background``: that path re-runs ``comfy`` from PATH
and drops any output printed before ComfyUI's banner, so a failed start showed
an empty error box. When the process exits before the port answers, the
``TowerError`` carries the exit code, command, cwd, python, and log tail.

``start``, ``stop``, ``status``, and ``restart`` never run ``comfy install``
or ``comfy update`` and never download models. Opt-in updates live in
``comfy/updates.py``. ``comfy_mode=external`` (state file or ``COMFY_MODE``)
refuses start, stop, and restart so a user-owned server is left alone.
Managed start writes a Buddy-owned ``extra_model_paths.yaml`` under state
and passes it with ``--extra-model-paths-config``. That file is not
``comfy attach`` (WorkflowPatchPlan). See ``comfy/model_paths.py``.

``comfy attach`` in the CLI is still WorkflowPatchPlan. It is not this module.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from master_agent.comfy.root import SAVED_SOURCES, resolve_managed_comfy_root
from master_agent.config import COMFYUI_PORT, COMFYUI_URL, PROJECT_ROOT, STATE_DIR

log = logging.getLogger(__name__)
_LOCK = threading.Lock()

DEFAULT_PORT = 8188
DEFAULT_READY_TIMEOUT_S = 180.0
DEFAULT_POLL_S = 2.0
DEFAULT_WATCH_INTERVAL_S = 15.0
# --no-wait still watches this long so an instant crash is reported, not hidden.
EARLY_EXIT_GRACE_S = 3.0
STOP_WAIT_S = 10.0
LOG_TAIL_LINES = 40
STATE_FILENAME = "managed_comfy.json"
LISTEN_HOST = "127.0.0.1"
_MODES = frozenset({"managed", "external"})

__all__ = [
    "ComfyLaunchError",
    "ManagedComfyTower",
    "TowerError",
    "TowerState",
    "cmd_tower",
    "comfy_cli_present",
    "comfy_log_path",
    "effective_mode",
    "http_ready",
    "load_state",
    "resolve_comfy_cli",
    "run_watchdog_forever",
    "save_state",
    "state_path",
    "wait_until_ready",
    "watchdog_tick",
]


class TowerError(RuntimeError):
    """Managed Comfy lifecycle failed or was refused."""


class ComfyLaunchError(TowerError):
    """ComfyUI did not come up. ``details`` names what ran and what it printed."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


@dataclass
class TowerState:
    """Persisted bookkeeping under ``state/managed_comfy.json``."""

    mode: str = "managed"  # managed | external
    workspace: str = ""
    base_url: str = "http://127.0.0.1:8188"
    port: int = DEFAULT_PORT
    want_running: bool = False
    last_start_at: float | None = None
    last_stop_at: float | None = None
    last_error: str = ""
    restart_count: int = 0
    watchdog_pid: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["comfy_mode"] = self.mode
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TowerState:
        if not isinstance(data, dict):
            return cls()
        known = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in known}
        if "mode" not in kwargs and isinstance(data.get("comfy_mode"), str):
            kwargs["mode"] = data["comfy_mode"]
        try:
            st = cls(**kwargs)
        except TypeError:
            log.warning("managed_comfy state had an unexpected shape; using defaults")
            return cls()
        mode = str(st.mode or "managed").strip().lower()
        if mode not in _MODES:
            log.warning("managed_comfy mode %r is invalid; using managed", st.mode)
            mode = "managed"
        st.mode = mode
        if not isinstance(st.extra, dict):
            st.extra = {}
        try:
            st.port = int(st.port)
        except (TypeError, ValueError):
            st.port = DEFAULT_PORT
        try:
            st.restart_count = int(st.restart_count)
        except (TypeError, ValueError):
            st.restart_count = 0
        if st.watchdog_pid in ("", None):
            st.watchdog_pid = None
        else:
            try:
                st.watchdog_pid = int(st.watchdog_pid)
            except (TypeError, ValueError):
                st.watchdog_pid = None
        return st


def _default_base_url(port: int | None = None) -> str:
    url = (COMFYUI_URL or "").rstrip("/")
    chosen = int(port or COMFYUI_PORT or DEFAULT_PORT)
    if url and (port is None or url.endswith(f":{chosen}")):
        return url
    return f"http://{LISTEN_HOST}:{chosen}"


def resolve_comfy_cli() -> list[str]:
    """Argv prefix for comfy-cli. ``COMFY_CLI`` wins, then ``comfy`` on PATH."""
    override = (os.getenv("COMFY_CLI") or "").strip()
    if override:
        parts = shlex.split(override)
        if parts:
            return parts
    which = shutil.which("comfy")
    if which:
        return [which]
    return [sys.executable, "-m", "comfy_cli"]


def sageattention_available() -> bool:
    """True when this interpreter can import ``sageattention``.

    Managed Comfy is launched via comfy-cli. When that CLI is
    ``sys.executable -m comfy_cli``, this is the same interpreter Comfy uses.
    A separate Comfy venv is not probed.
    """
    try:
        return importlib.util.find_spec("sageattention") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def comfy_sageattention_available(workspace: Path | str | None) -> bool:
    """Probe the interpreter Comfy runs in (``<workspace>/.venv``).

    When that interpreter cannot be found, fall back to
    :func:`sageattention_available` on this interpreter.
    """
    from master_agent.comfy.comfy_venv import module_importable, workspace_venv_python

    py = workspace_venv_python(workspace)
    if py is None:
        return sageattention_available()
    return module_importable(py, "sageattention")


def comfy_cli_present() -> bool:
    """True when an operator binary, PATH entry, or the ``comfy_cli`` module exists."""
    if (os.getenv("COMFY_CLI") or "").strip():
        return True
    if shutil.which("comfy"):
        return True
    try:
        return importlib.util.find_spec("comfy_cli") is not None
    except (ImportError, ValueError):
        return False


def state_path() -> Path:
    return Path(STATE_DIR) / STATE_FILENAME


def load_state() -> TowerState:
    """Saved bookkeeping. ``workspace`` is only an explicit ``--workspace`` choice;
    resolve the folder with ``comfy/root.py``, not from this field."""
    path = state_path()
    if not path.is_file():
        return TowerState(
            base_url=_default_base_url(),
            port=int(COMFYUI_PORT or DEFAULT_PORT),
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("managed_comfy state unreadable (%s); using defaults", exc)
        return TowerState(base_url=_default_base_url())
    st = TowerState.from_dict(data)
    if not st.base_url:
        st.base_url = _default_base_url(st.port)
    return st


def save_state(st: TowerState) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(st.to_dict(), indent=2) + "\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def effective_mode(st: TowerState | None = None) -> str:
    """``COMFY_MODE`` overrides the state file. Default is managed.

    Accepted values are ``managed`` and ``external`` (product name: comfy_mode).
    """
    raw = (os.getenv("COMFY_MODE") or "").strip().lower()
    if raw:
        if raw not in _MODES:
            raise TowerError(
                f"COMFY_MODE must be managed or external, got {raw!r}"
            )
        return raw
    if st is None:
        st = load_state()
    mode = (st.mode or "managed").strip().lower()
    return mode if mode in _MODES else "managed"


def _child_env(workspace: Path | str | None = None) -> dict[str, str]:
    from master_agent.comfy.comfy_venv import strip_outer_venv

    env = os.environ.copy()
    # comfy-cli routes to Comfy Cloud when a session exists. Force local.
    env["COMFY_WHERE"] = "local"
    return strip_outer_venv(env, workspace)


def _run_comfy(
    args: list[str],
    *,
    workspace: Path,
    timeout: float | None = 120.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """One comfy-cli invocation: workspace + ``--where local`` + no prompts."""
    cmd = [
        *resolve_comfy_cli(),
        f"--workspace={workspace}",
        "--where",
        "local",
        "--skip-prompt",
        *args,
    ]
    log.info("comfy-cli: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_child_env(workspace),
        )
    except FileNotFoundError as exc:
        raise TowerError(
            "comfy-cli not found. Install with: pip install comfy-cli==1.20.0  "
            f"(looked for: {resolve_comfy_cli()!r}). "
            "This command does not install or update ComfyUI."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TowerError(f"comfy-cli timed out after {timeout}s: {' '.join(cmd)}") from exc
    if check and proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise TowerError(f"comfy-cli failed ({proc.returncode}): {err[:800]}.")
    return proc


def http_ready(base_url: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    """Probe ``/system_stats`` then ``/object_info``. Returns ``(ok, detail)``."""
    base = (base_url or "").rstrip("/")
    last = "unreachable"
    if not base:
        return False, last
    for path in ("/system_stats", "/object_info"):
        url = f"{base}{path}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = getattr(resp, "status", None) or resp.getcode()
                if 200 <= int(code) < 300:
                    return True, f"{path} ok"
                last = f"{path}: HTTP {code}"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = f"{path}: {exc}"
    return False, last


def wait_until_ready(
    base_url: str,
    *,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    on_tick: Callable[[float, str], None] | None = None,
) -> None:
    """Block until Comfy answers a health probe, or raise ``TowerError``."""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    last = "waiting"
    while True:
        ok, detail = http_ready(base_url)
        last = detail
        remaining = max(0.0, deadline - time.monotonic())
        if on_tick:
            on_tick(remaining, detail)
        if ok:
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.0, float(poll_s)))
    raise TowerError(
        f"ComfyUI did not become ready at {base_url} within {timeout_s:.0f}s ({last})"
    )


def _pid_alive(pid: int | None) -> bool:
    if not pid or int(pid) <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) on Windows sends CTRL_C_EVENT; ask the kernel instead.
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def watchdog_command() -> list[str]:
    """Detached interpreter that runs :func:`run_watchdog_forever`."""
    return [sys.executable, "-m", "master_agent.comfy.tower"]


def comfy_log_path(port: int) -> Path:
    """ComfyUI's stdout + stderr for the managed server on ``port``. Truncated per launch."""
    return Path(STATE_DIR) / "logs" / f"comfyui_{int(port)}.log"


def read_log_tail(path: Path | str, lines: int = LOG_TAIL_LINES) -> list[str]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-max(0, int(lines)):]


def _detached_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


def _terminate_tree(pid: int) -> None:
    """Stop the ComfyUI process Buddy launched (and its children). Never a port owner."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return
    try:
        # start_new_session made the ComfyUI pid its own process-group leader.
        os.killpg(int(pid), signal.SIGTERM)
    except (OSError, AttributeError):
        os.kill(int(pid), signal.SIGTERM)


def _quote_cmd(cmd: list[Any]) -> str:
    parts = [str(p) for p in cmd]
    return subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)


def _format_launch_failure(headline: str, details: dict[str, Any]) -> str:
    lines = [
        headline,
        f"  command: {_quote_cmd(details.get('command') or [])}",
        f"  cwd:     {details.get('cwd')}",
        f"  python:  {details.get('python')}",
        f"  workspace: {details.get('workspace')} (from {details.get('workspace_source')})",
        f"  log:     {details.get('log')}",
    ]
    if details.get("exit_code") is not None:
        lines.insert(1, f"  exit code: {details['exit_code']}")
    tail = details.get("log_tail") or []
    if tail:
        lines.append(f"  last {len(tail)} line(s) of ComfyUI output:")
        lines.extend(f"    {line}" for line in tail)
    else:
        lines.append("  ComfyUI printed nothing before it stopped.")
    return "\n".join(lines)


class ManagedComfyTower:
    """Start, stop, and crash-restart one Buddy-owned local ComfyUI."""

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        base_url: str | None = None,
        port: int | None = None,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
        watch_interval_s: float = DEFAULT_WATCH_INTERVAL_S,
    ) -> None:
        st = load_state()
        root = resolve_managed_comfy_root(workspace)
        self.workspace = root.path
        self.workspace_source = root.source
        self.port = int(port if port is not None else (st.port or COMFYUI_PORT or DEFAULT_PORT))
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif port is None and st.base_url:
            self.base_url = st.base_url.rstrip("/")
        else:
            self.base_url = _default_base_url(self.port)
        self.ready_timeout_s = float(ready_timeout_s)
        self.watch_interval_s = float(watch_interval_s)

    def _sync_state(self, **updates: Any) -> TowerState:
        st = load_state()
        if self.workspace_source in SAVED_SOURCES:
            st.workspace = str(self.workspace)
        st.base_url = self.base_url
        st.port = self.port
        for key, value in updates.items():
            if hasattr(st, key):
                setattr(st, key, value)
        save_state(st)
        return st

    def _require_managed(self, action: str) -> None:
        mode = effective_mode(load_state())
        if mode == "external":
            raise TowerError(
                f"comfy_mode=external: refusing to {action} a user-owned Comfy. "
                "Buddy will not start, stop, or restart a foreign server."
            )

    def status(self) -> dict[str, Any]:
        st = load_state()
        mode = effective_mode(st)
        ok, detail = http_ready(self.base_url)
        pid = st.watchdog_pid
        return {
            "ok": ok,
            "detail": detail,
            "mode": mode,
            "comfy_mode": mode,
            "want_running": bool(st.want_running),
            "workspace": str(self.workspace),
            "workspace_source": self.workspace_source,
            "base_url": self.base_url,
            "port": self.port,
            "restart_count": int(st.restart_count),
            "last_error": st.last_error,
            "comfy_pid": (st.extra or {}).get("comfy_pid"),
            "log": str(comfy_log_path(self.port)),
            "watchdog_pid": pid,
            "watching": _pid_alive(pid),
            "comfy_cli": resolve_comfy_cli(),
            "comfy_cli_present": comfy_cli_present(),
            "extra_model_paths": str((st.extra or {}).get("extra_model_paths") or ""),
        }

    def _comfy_python(self) -> Path:
        """Interpreter ComfyUI runs in: ``<workspace>/.venv`` (or ``venv``), else this one."""
        from master_agent.comfy.comfy_venv import workspace_venv_python

        if not (self.workspace / "main.py").is_file():
            from master_agent.install_report import FULL_INSTALL_CMD

            raise TowerError(
                f"ComfyUI is not installed in {self.workspace} (from {self.workspace_source}): "
                "there is no main.py there. "
                f"Install it with: {FULL_INSTALL_CMD}  "
                "or set MANAGED_COMFY_ROOT to the folder that has ComfyUI's main.py. "
                "comfy start does not install ComfyUI."
            )
        py = workspace_venv_python(self.workspace)
        if py is None:
            log.warning("no .venv in %s; launching ComfyUI with %s", self.workspace, sys.executable)
            return Path(sys.executable)
        return Path(py)

    def _launch_args(self, extra_model_paths: Path | None) -> list[str]:
        """ComfyUI's own flags (what follows ``main.py``)."""
        args = [
            "--disable-auto-launch",
            "--port",
            str(self.port),
            "--listen",
            LISTEN_HOST,
        ]
        if extra_model_paths is not None:
            emp = Path(extra_model_paths)
            if not emp.is_file():
                raise TowerError(f"extra_model_paths not found: {emp}")
            args.extend(["--extra-model-paths-config", str(emp)])
        # Last, after extra-model-paths when present. The check runs in the
        # workspace venv Comfy launches with; without one it falls back to this
        # interpreter. A miss omits the flag instead of crashing that launch.
        if comfy_sageattention_available(self.workspace):
            args.append("--use-sage-attention")
        else:
            from master_agent.comfy.comfy_venv import workspace_venv_python

            log.warning(
                "sageattention is not importable in %s; omitting --use-sage-attention "
                "so Comfy does not crash. Install sageattention in the Comfy "
                "interpreter to enable it.",
                workspace_venv_python(self.workspace) or sys.executable,
            )
        return args

    def _launch(self, extra_model_paths: Path | None = None) -> tuple[Any, dict[str, Any]]:
        """Spawn ComfyUI detached. Returns ``(Popen, details)``; does not wait for it."""
        python = self._comfy_python()
        cmd = [str(python), "main.py", *self._launch_args(extra_model_paths)]
        log_path = comfy_log_path(self.port)
        details: dict[str, Any] = {
            "command": cmd,
            "cwd": str(self.workspace),
            "python": str(python),
            "workspace": str(self.workspace),
            "workspace_source": self.workspace_source,
            "log": str(log_path),
            "exit_code": None,
            "log_tail": [],
        }
        env = _child_env(self.workspace)
        env["PYTHONUNBUFFERED"] = "1"
        # A cp1252 console makes ComfyUI's first non-ASCII log line raise.
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logfh = open(log_path, "w", encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ComfyLaunchError(
                _format_launch_failure(f"Could not open the ComfyUI log {log_path}: {exc}", details), details
            ) from exc
        log.info("launching ComfyUI: %s (cwd=%s)", " ".join(cmd), self.workspace)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.workspace),
                stdin=subprocess.DEVNULL,
                stdout=logfh,
                stderr=subprocess.STDOUT,
                env=env,
                **_detached_kwargs(),
            )
        except OSError as exc:
            raise ComfyLaunchError(
                _format_launch_failure(f"Could not run ComfyUI's python ({exc}).", details), details
            ) from exc
        finally:
            logfh.close()
        details["pid"] = int(proc.pid)
        extra = dict(load_state().extra or {})
        extra.update({"comfy_pid": int(proc.pid), "comfy_log": str(log_path), "comfy_command": cmd})
        self._sync_state(extra=extra)
        return proc, details

    def _await_ready(self, proc: Any, details: dict[str, Any], *, timeout_s: float) -> tuple[bool, str]:
        """Poll the HTTP port until it answers, ComfyUI exits, or ``timeout_s`` passes.

        Raises :class:`ComfyLaunchError` when the process is gone before the
        port answers. Returns ``(False, probe)`` on timeout with ComfyUI still running.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        probe = "waiting"
        while True:
            ok, probe = http_ready(self.base_url)
            if ok:
                return True, probe
            code = proc.poll() if hasattr(proc, "poll") else None
            if code is not None:
                details["exit_code"] = int(code)
                details["log_tail"] = read_log_tail(details["log"])
                raise ComfyLaunchError(
                    _format_launch_failure(
                        f"ComfyUI exited with code {code} before it answered on {self.base_url}.", details
                    ),
                    details,
                )
            if time.monotonic() >= deadline:
                return False, probe
            log.info("waiting for Comfy (%.0fs left): %s", deadline - time.monotonic(), probe)
            time.sleep(DEFAULT_POLL_S)

    def _launch_failed(self, exc: TowerError) -> None:
        extra = dict(load_state().extra or {})
        if isinstance(exc, ComfyLaunchError):
            extra["last_launch_failure"] = exc.details
        self._sync_state(want_running=False, last_error=str(exc)[:4000], extra=extra)

    def _recorded_comfy_pid(self) -> int | None:
        pid = (load_state().extra or {}).get("comfy_pid")
        try:
            return int(pid) if pid else None
        except (TypeError, ValueError):
            return None

    def _kill_recorded_comfy(self) -> int | None:
        """Terminate the ComfyUI pid this module launched. Returns it, or None."""
        pid = self._recorded_comfy_pid()
        if pid is None or pid == os.getpid() or not _pid_alive(pid):
            return None
        try:
            _terminate_tree(pid)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("ComfyUI pid %s did not stop: %s", pid, exc)
        extra = dict(load_state().extra or {})
        extra.pop("comfy_pid", None)
        self._sync_state(extra=extra)
        return pid

    def _remember_yaml(self, path: Path) -> None:
        st = load_state()
        extra = dict(st.extra or {})
        extra["extra_model_paths"] = str(path)
        self._sync_state(extra=extra)

    def _yaml_for_launch(self, explicit: Path | None) -> Path:
        """Buddy-owned YAML, or an existing file the caller already has.

        Does not write into an attached Comfy tree.
        """
        from master_agent.comfy.model_paths import ModelPathsError, write_extra_model_paths

        try:
            if explicit is not None:
                emp = Path(explicit).expanduser()
                if not emp.is_file():
                    raise TowerError(f"extra_model_paths not found: {emp}")
                resolved = emp.resolve()
            else:
                resolved = write_extra_model_paths(directory=state_path().parent)
        except ModelPathsError as exc:
            raise TowerError(str(exc)) from exc
        self._remember_yaml(resolved)
        return resolved

    def _maybe_write_external_yaml(self, enabled: bool, source: Path | None) -> Path | None:
        if not enabled:
            return None
        from master_agent.comfy.model_paths import ModelPathsError, write_external_yaml_copy

        try:
            return write_external_yaml_copy(source)
        except ModelPathsError as exc:
            raise TowerError(str(exc)) from exc

    def start_watchdog(self) -> int | None:
        """Spawn a detached watchdog so crash recovery outlives ``comfy start``.

        comfy-cli already daemonizes ComfyUI. A thread inside the CLI process
        would die on exit, so the supervisor is ``python -m master_agent.comfy.tower``.
        """
        st = load_state()
        if _pid_alive(st.watchdog_pid):
            return int(st.watchdog_pid)  # type: ignore[arg-type]
        cmd = watchdog_command()
        kwargs: dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": _child_env(),
        }
        if os.name == "nt":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            kwargs["creationflags"] = flags
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except OSError as exc:
            raise TowerError(f"failed to spawn managed Comfy watchdog: {exc}") from exc
        self._sync_state(watchdog_pid=int(proc.pid))
        log.info("managed Comfy watchdog pid=%s", proc.pid)
        return int(proc.pid)

    def stop_watchdog(self) -> None:
        st = load_state()
        pid = int(st.watchdog_pid or 0)
        self._sync_state(watchdog_pid=None)
        if not pid or pid == os.getpid() or not _pid_alive(pid):
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            log.warning("watchdog pid %s did not exit: %s", pid, exc)

    def start(
        self,
        *,
        extra_model_paths: Path | None = None,
        wait: bool = True,
        watch: bool = True,
        write_yaml_into_external: bool = False,
    ) -> dict[str, Any]:
        """Launch managed Comfy in the background and optionally wait until ready.

        A server that is already answering, and that this module did not mark
        ``want_running``, is treated as foreign and left untouched.

        The Buddy-owned model-paths YAML is written even when launch is then
        refused (external mode), so the operator still has a snippet. The
        attached tree is touched only when ``write_yaml_into_external`` is set.
        """
        yaml_path = self._yaml_for_launch(extra_model_paths)
        external_yaml = self._maybe_write_external_yaml(write_yaml_into_external, yaml_path)
        self._require_managed("start")
        launched = False
        with _LOCK:
            ok, detail = http_ready(self.base_url)
            st = load_state()
            if ok and st.want_running:
                self._sync_state(
                    mode="managed",
                    want_running=True,
                    last_error="",
                    last_start_at=st.last_start_at or time.time(),
                )
            elif ok:
                raise TowerError(
                    f"Comfy is already reachable at {self.base_url} ({detail}). "
                    "Refusing to adopt a foreign server. "
                    "Set COMFY_MODE=external if this Comfy is user-owned, "
                    "or stop it before `comfy start`."
                )
            else:
                try:
                    proc, launch = self._launch(yaml_path)
                except TowerError as exc:
                    self._launch_failed(exc)
                    raise
                self._sync_state(
                    mode="managed",
                    want_running=True,
                    last_start_at=time.time(),
                    last_error="",
                )
                launched = True
                detail = f"launched pid {launch['pid']}"

        if launched:
            try:
                ready_now, probe = self._await_ready(
                    proc, launch, timeout_s=self.ready_timeout_s if wait else EARLY_EXIT_GRACE_S
                )
            except ComfyLaunchError as exc:
                self._launch_failed(exc)
                raise
            if wait and not ready_now:
                launch["log_tail"] = read_log_tail(launch["log"])
                message = _format_launch_failure(
                    f"ComfyUI is still running (pid {launch['pid']}) but did not answer on "
                    f"{self.base_url} within {self.ready_timeout_s:.0f}s ({probe}). "
                    "It may still be loading; check `python -m master_agent comfy status`.",
                    launch,
                )
                self._sync_state(last_error=message[:4000])
                if watch:
                    self.start_watchdog()
                raise ComfyLaunchError(message, launch)

        if watch:
            self.start_watchdog()

        ready, probe = http_ready(self.base_url)
        return {
            "status": "started" if launched else "already_up",
            "detail": probe if ready else detail,
            "base_url": self.base_url,
            "workspace": str(self.workspace),
            "workspace_source": self.workspace_source,
            "ready": ready,
            "log": str(comfy_log_path(self.port)),
            "extra_model_paths": str(yaml_path),
            "external_yaml": str(external_yaml or ""),
        }

    def stop(self) -> dict[str, Any]:
        """Stop the ComfyUI pid Buddy launched. Does not kill an arbitrary port owner.

        With no live recorded pid (a server an older Buddy started through
        ``comfy launch --background``), falls back to ``comfy stop``.
        """
        self._require_managed("stop")
        self._sync_state(want_running=False, last_stop_at=time.time())
        self.stop_watchdog()
        killed = self._kill_recorded_comfy()
        if killed is not None:
            cli_detail = f"stopped ComfyUI pid {killed}"
            returncode = 0
            deadline = time.monotonic() + STOP_WAIT_S
            while http_ready(self.base_url)[0] and time.monotonic() < deadline:
                time.sleep(0.5)
        else:
            try:
                proc = _run_comfy(["stop"], workspace=self.workspace, timeout=60.0, check=False)
            except TowerError as exc:
                self._sync_state(last_error=str(exc))
                raise
            cli_detail = (proc.stderr or proc.stdout or "").strip()
            returncode = proc.returncode
        self._sync_state(
            want_running=False,
            last_stop_at=time.time(),
            last_error="" if returncode == 0 else cli_detail[:500],
        )
        ok, detail = http_ready(self.base_url)
        return {
            "status": "stopped" if not ok else "stop_issued_still_reachable",
            "detail": detail,
            "cli_detail": cli_detail,
            "returncode": returncode,
            "base_url": self.base_url,
            "workspace": str(self.workspace),
        }

    def restart(
        self,
        *,
        extra_model_paths: Path | None = None,
        wait: bool = True,
        watch: bool = True,
        write_yaml_into_external: bool = False,
    ) -> dict[str, Any]:
        """Stop, then start. Increments ``restart_count`` after a successful start."""
        yaml_path = self._yaml_for_launch(extra_model_paths)
        external_yaml = self._maybe_write_external_yaml(write_yaml_into_external, yaml_path)
        self._require_managed("restart")
        try:
            self.stop()
        except TowerError as exc:
            log.warning("stop before restart: %s", exc)
        result = self.start(
            extra_model_paths=yaml_path,
            wait=wait,
            watch=watch,
            write_yaml_into_external=False,
        )
        if external_yaml is not None:
            result["external_yaml"] = str(external_yaml)
        st = load_state()
        self._sync_state(restart_count=int(st.restart_count) + 1)
        result["status"] = "restarted"
        result["restart_count"] = int(load_state().restart_count)
        return result

    def recover_from_crash(self) -> dict[str, Any]:
        """Relaunch after the health probe failed. Does not stop the watchdog process."""
        self._require_managed("restart")
        st = load_state()
        if not st.want_running:
            raise TowerError("managed Comfy is not marked want_running; refusing crash restart")
        with _LOCK:
            st = load_state()
            if effective_mode(st) != "managed" or not st.want_running:
                raise TowerError("managed Comfy is no longer wanted; refusing crash restart")
            if self._kill_recorded_comfy() is None:
                try:
                    _run_comfy(["stop"], workspace=self.workspace, timeout=60.0, check=False)
                except TowerError as exc:
                    log.warning("comfy stop before crash restart: %s", exc)
            st = load_state()
            if not st.want_running or effective_mode(st) != "managed":
                return {"status": "aborted", "base_url": self.base_url}
            try:
                proc, launch = self._launch(self._yaml_for_launch(None))
            except TowerError as exc:
                self._sync_state(last_error=str(exc))
                raise
            count = int(load_state().restart_count) + 1
            self._sync_state(
                mode="managed",
                want_running=True,
                last_start_at=time.time(),
                last_error="",
                restart_count=count,
            )
        try:
            ready, probe = self._await_ready(proc, launch, timeout_s=self.ready_timeout_s)
        except ComfyLaunchError as exc:
            self._sync_state(last_error=str(exc)[:4000])
            raise
        if not ready:
            raise TowerError(
                f"ComfyUI did not become ready at {self.base_url} within {self.ready_timeout_s:.0f}s ({probe})"
            )
        return {
            "status": "restarted",
            "base_url": self.base_url,
            "workspace": str(self.workspace),
            "restart_count": int(load_state().restart_count),
            "ready": True,
        }


def watchdog_tick(tower: ManagedComfyTower | None = None) -> str:
    """One supervisor pass. Returns ``exit``, ``healthy``, ``starting``, ``restarted``, or ``error``."""
    tower = tower or ManagedComfyTower()
    st = load_state()
    try:
        mode = effective_mode(st)
    except TowerError as exc:
        log.error("watchdog mode: %s", exc)
        return "exit"
    if mode != "managed" or not st.want_running:
        return "exit"
    ok, detail = http_ready(tower.base_url)
    if ok:
        if st.last_error:
            tower._sync_state(last_error="")
        return "healthy"
    started = float(st.last_start_at or 0)
    if started and (time.time() - started) < tower.ready_timeout_s:
        log.info("managed Comfy still starting (%s)", detail)
        return "starting"
    log.warning("managed Comfy down (%s); restarting", detail)
    try:
        tower.recover_from_crash()
    except TowerError as exc:
        log.error("watchdog restart failed: %s", exc)
        tower._sync_state(last_error=str(exc))
        return "error"
    return "restarted"


def run_watchdog_forever(interval_s: float | None = None) -> None:
    """Block until comfy_mode is external or ``want_running`` is cleared."""
    tower = ManagedComfyTower()
    interval = tower.watch_interval_s if interval_s is None else float(interval_s)
    log.info(
        "managed Comfy watchdog pid=%s url=%s workspace=%s",
        os.getpid(),
        tower.base_url,
        tower.workspace,
    )
    while True:
        action = watchdog_tick(tower)
        if action == "exit":
            log.info("managed Comfy watchdog exiting")
            return
        time.sleep(max(0.05, interval))


def _wants_json(args: Any) -> bool:
    if bool(getattr(args, "as_json", False)):
        return True
    flag = getattr(args, "json", False)
    return flag is True


TOWER_COMMANDS = ("start", "stop", "status", "restart")


def run_tower_command(
    command: str,
    *,
    workspace: Path | None = None,
    port: int | None = None,
    base_url: str | None = None,
    wait: bool = True,
    watch: bool = True,
    ready_timeout_s: float | None = None,
    extra_model_paths: Path | None = None,
    write_yaml_into_external: bool = False,
) -> dict[str, Any]:
    """Body of ``comfy start|stop|status|restart``, shared by the CLI and the agent tools.

    Raises ``TowerError`` (``ComfyLaunchError`` carries the launch details) or
    ``ModelPathsError``; :func:`tower_error_payload` turns those into JSON.
    """
    if command not in TOWER_COMMANDS:
        raise TowerError(f"unknown comfy tower command: {command!r}")
    kwargs: dict[str, Any] = {"workspace": workspace, "port": port, "base_url": base_url}
    if ready_timeout_s is not None:
        kwargs["ready_timeout_s"] = ready_timeout_s
    tower = ManagedComfyTower(**kwargs)
    if command == "status":
        from master_agent.comfy.model_paths import write_external_yaml_copy, write_extra_model_paths

        buddy_yaml = write_extra_model_paths(directory=state_path().parent)
        result = tower.status()
        result["extra_model_paths"] = str(buddy_yaml)
        if write_yaml_into_external:
            result["external_yaml"] = str(write_external_yaml_copy(buddy_yaml))
    elif command == "start":
        result = tower.start(
            extra_model_paths=extra_model_paths,
            wait=wait,
            watch=watch,
            write_yaml_into_external=write_yaml_into_external,
        )
    elif command == "stop":
        return tower.stop()
    else:
        result = tower.restart(
            extra_model_paths=extra_model_paths,
            wait=wait,
            watch=watch,
            write_yaml_into_external=write_yaml_into_external,
        )
    try:
        from master_agent.comfy.updates import startup_report

        result["startup"] = startup_report(tower.workspace)
    except Exception as exc:
        result["startup"] = {
            "updated": False,
            "blocks_install": False,
            "lines": [f"Startup check failed ({exc}). No update ran."],
        }
    return result


def tower_error_payload(command: str, exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "error", "command": command, "error": str(exc)}
    if isinstance(exc, ComfyLaunchError):
        payload["launch"] = exc.details
    return payload


def cmd_tower(args: Any) -> int:
    """CLI handler for ``python -m master_agent comfy {start,stop,status,restart}``."""
    command = getattr(args, "comfy_command", None) or getattr(args, "tower_command", None)
    workspace = getattr(args, "workspace", None)
    port = getattr(args, "port", None)
    base_url = getattr(args, "base_url", None)
    no_wait = bool(getattr(args, "no_wait", False))
    no_watch = bool(getattr(args, "no_watch", False))
    extra = getattr(args, "extra_model_paths", None)
    write_into_external = bool(getattr(args, "write_yaml_into_external", False))

    try:
        chosen_port = int(port) if port else None
    except (TypeError, ValueError) as exc:
        print(f"FAIL  invalid --port {port!r}")
        return 2
    if chosen_port is not None and not (1 <= chosen_port <= 65535):
        print(f"FAIL  invalid --port {chosen_port}")
        return 2

    if command not in TOWER_COMMANDS:
        print(f"FAIL  unknown comfy tower command: {command!r}")
        return 2
    if write_into_external and command not in {"status", "start", "restart"}:
        print("FAIL  --write-yaml-into-external is only valid with status, start, or restart")
        return 2

    from master_agent.comfy.model_paths import ModelPathsError

    try:
        result = run_tower_command(
            command,
            workspace=Path(workspace).expanduser().resolve() if workspace else None,
            port=chosen_port,
            base_url=base_url,
            wait=not no_wait,
            watch=not no_watch,
            extra_model_paths=Path(extra) if extra else None,
            write_yaml_into_external=write_into_external,
        )
    except (TowerError, ModelPathsError) as exc:
        if _wants_json(args):
            print(json.dumps(tower_error_payload(command, exc), indent=2, default=str))
        else:
            print(f"FAIL  comfy {command}: {exc}")
        return 1

    if _wants_json(args):
        print(json.dumps(result, indent=2, default=str))
    else:
        if command == "status":
            mark = "OK" if result.get("ok") else "DOWN"
            label = "up" if result.get("ok") else "down"
        else:
            mark = "OK"
            label = str(result.get("status") or "done")
        startup = result.get("startup") if isinstance(result.get("startup"), dict) else None
        if startup:
            for line in startup.get("lines") or []:
                print(line)
        print(f"{mark:4}  comfy {command}: {label}")
        if result.get("base_url"):
            print(f"      url={result['base_url']}")
        if result.get("workspace"):
            source = f" (from {result['workspace_source']})" if result.get("workspace_source") else ""
            print(f"      workspace={result['workspace']}{source}")
        if result.get("log"):
            print(f"      log={result['log']}")
        if result.get("mode") or result.get("comfy_mode"):
            print(f"      comfy_mode={result.get('comfy_mode') or result.get('mode')}")
        if result.get("extra_model_paths"):
            print(f"      extra_model_paths={result['extra_model_paths']}")
        if result.get("external_yaml"):
            print(f"      external_yaml={result['external_yaml']}")
        if result.get("detail"):
            print(f"      {result['detail']}")
        if result.get("last_error"):
            print(f"      last_error={result['last_error']}")
        if result.get("restart_count"):
            print(f"      restart_count={result['restart_count']}")
    if command == "status" and not result.get("ok"):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the detached watchdog (``python -m master_agent.comfy.tower``)."""
    del argv
    run_watchdog_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
