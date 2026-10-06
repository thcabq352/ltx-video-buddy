"""Managed ComfyUI lifecycle via comfy-cli (start / stop / status / restart).

Buddy can own a local ComfyUI process under ``MANAGED_COMFY_ROOT`` (default
``PROJECT_ROOT/ComfyUI``). The HTTP client, graph ops, linter, and generate
path are unchanged: they keep talking to whatever answers on ``COMFYUI_URL``.

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

from master_agent.config import COMFYUI_PORT, COMFYUI_URL, PROJECT_ROOT, STATE_DIR

log = logging.getLogger(__name__)
_LOCK = threading.Lock()

DEFAULT_PORT = 8188
DEFAULT_READY_TIMEOUT_S = 180.0
DEFAULT_POLL_S = 2.0
DEFAULT_WATCH_INTERVAL_S = 15.0
STATE_FILENAME = "managed_comfy.json"
LISTEN_HOST = "127.0.0.1"
_MODES = frozenset({"managed", "external"})

__all__ = [
    "ManagedComfyTower",
    "TowerError",
    "TowerState",
    "cmd_tower",
    "comfy_cli_present",
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


def _default_workspace() -> Path:
    raw = (os.getenv("MANAGED_COMFY_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path(PROJECT_ROOT) / "ComfyUI").resolve()


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
    path = state_path()
    if not path.is_file():
        return TowerState(
            workspace=str(_default_workspace()),
            base_url=_default_base_url(),
            port=int(COMFYUI_PORT or DEFAULT_PORT),
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("managed_comfy state unreadable (%s); using defaults", exc)
        return TowerState(
            workspace=str(_default_workspace()),
            base_url=_default_base_url(),
        )
    st = TowerState.from_dict(data)
    if not st.workspace:
        st.workspace = str(_default_workspace())
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


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    # comfy-cli routes to Comfy Cloud when a session exists. Force local.
    env["COMFY_WHERE"] = "local"
    return env


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
            env=_child_env(),
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
        hint = ""
        if "launch" in args:
            hint = " Managed start does not install or update ComfyUI."
        raise TowerError(f"comfy-cli failed ({proc.returncode}): {err[:800]}.{hint}")
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
        self.workspace = Path(workspace or st.workspace or _default_workspace()).resolve()
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
            "base_url": self.base_url,
            "port": self.port,
            "restart_count": int(st.restart_count),
            "last_error": st.last_error,
            "watchdog_pid": pid,
            "watching": _pid_alive(pid),
            "comfy_cli": resolve_comfy_cli(),
            "comfy_cli_present": comfy_cli_present(),
            "extra_model_paths": str((st.extra or {}).get("extra_model_paths") or ""),
        }

    def _launch_args(self, extra_model_paths: Path | None) -> list[str]:
        args = [
            "launch",
            "--background",
            "--",
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
            # ComfyUI flag, so it stays after `--`. This function only forwards the path.
            args.extend(["--extra-model-paths-config", str(emp)])
        # ComfyUI flag, last, after `--` and after extra-model-paths when present.
        # The check is this interpreter. comfy-cli on the tower may use another
        # Python; a miss here omits the flag instead of crashing that launch.
        if sageattention_available():
            args.append("--use-sage-attention")
        else:
            log.warning(
                "sageattention is not importable in %s; omitting --use-sage-attention "
                "so Comfy does not crash. Install sageattention in the Comfy "
                "interpreter to enable it.",
                sys.executable,
            )
        return args

    def _launch(self, extra_model_paths: Path | None = None) -> subprocess.CompletedProcess[str]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        return _run_comfy(
            self._launch_args(extra_model_paths),
            workspace=self.workspace,
            timeout=300.0,
        )

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
                    self._launch(yaml_path)
                except TowerError as exc:
                    self._sync_state(want_running=False, last_error=str(exc))
                    raise
                self._sync_state(
                    mode="managed",
                    want_running=True,
                    last_start_at=time.time(),
                    last_error="",
                )
                launched = True
                detail = "launch issued"

        if launched and wait:
            try:
                wait_until_ready(
                    self.base_url,
                    timeout_s=self.ready_timeout_s,
                    on_tick=lambda remaining, probe: log.info(
                        "waiting for Comfy (%.0fs left): %s", remaining, probe
                    ),
                )
            except TowerError as exc:
                self._sync_state(last_error=str(exc))
                if watch:
                    self.start_watchdog()
                raise

        if watch:
            self.start_watchdog()

        ready, probe = http_ready(self.base_url)
        return {
            "status": "started" if launched else "already_up",
            "detail": probe if ready else detail,
            "base_url": self.base_url,
            "workspace": str(self.workspace),
            "ready": ready,
            "extra_model_paths": str(yaml_path),
            "external_yaml": str(external_yaml or ""),
        }

    def stop(self) -> dict[str, Any]:
        """Stop the comfy-cli background instance. Does not kill an arbitrary port owner."""
        self._require_managed("stop")
        self._sync_state(want_running=False, last_stop_at=time.time())
        self.stop_watchdog()
        try:
            proc = _run_comfy(["stop"], workspace=self.workspace, timeout=60.0, check=False)
        except TowerError as exc:
            self._sync_state(last_error=str(exc))
            raise
        cli_detail = (proc.stderr or proc.stdout or "").strip()
        self._sync_state(
            want_running=False,
            last_stop_at=time.time(),
            last_error="" if proc.returncode == 0 else cli_detail[:500],
        )
        ok, detail = http_ready(self.base_url)
        return {
            "status": "stopped" if not ok else "stop_issued_still_reachable",
            "detail": detail,
            "cli_detail": cli_detail,
            "returncode": proc.returncode,
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
            try:
                _run_comfy(["stop"], workspace=self.workspace, timeout=60.0, check=False)
            except TowerError as exc:
                log.warning("comfy stop before crash restart: %s", exc)
            st = load_state()
            if not st.want_running or effective_mode(st) != "managed":
                return {"status": "aborted", "base_url": self.base_url}
            try:
                self._launch(self._yaml_for_launch(None))
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
        wait_until_ready(self.base_url, timeout_s=self.ready_timeout_s)
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

    tower = ManagedComfyTower(
        workspace=Path(workspace).expanduser().resolve() if workspace else None,
        port=chosen_port,
        base_url=base_url,
    )

    from master_agent.comfy.model_paths import ModelPathsError

    try:
        if write_into_external and command not in {"status", "start", "restart"}:
            print("FAIL  --write-yaml-into-external is only valid with status, start, or restart")
            return 2
        if command == "status":
            from master_agent.comfy.model_paths import (
                write_external_yaml_copy,
                write_extra_model_paths,
            )

            buddy_yaml = write_extra_model_paths(directory=state_path().parent)
            result = tower.status()
            result["extra_model_paths"] = str(buddy_yaml)
            if write_into_external:
                result["external_yaml"] = str(write_external_yaml_copy(buddy_yaml))
        elif command == "start":
            result = tower.start(
                extra_model_paths=Path(extra) if extra else None,
                wait=not no_wait,
                watch=not no_watch,
                write_yaml_into_external=write_into_external,
            )
        elif command == "stop":
            result = tower.stop()
        elif command == "restart":
            result = tower.restart(
                extra_model_paths=Path(extra) if extra else None,
                wait=not no_wait,
                watch=not no_watch,
                write_yaml_into_external=write_into_external,
            )
        else:
            print(f"FAIL  unknown comfy tower command: {command!r}")
            return 2
    except (TowerError, ModelPathsError) as exc:
        print(f"FAIL  {exc}")
        return 1

    if command in {"start", "status", "restart"}:
        try:
            from master_agent.comfy.updates import startup_report

            result["startup"] = startup_report(tower.workspace)
        except Exception as exc:
            result["startup"] = {
                "updated": False,
                "blocks_install": False,
                "lines": [f"Startup check failed ({exc}). No update ran."],
            }

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
            print(f"      workspace={result['workspace']}")
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
