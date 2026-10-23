"""Seat rule: the local LLM is off the GPU before a ComfyUI render is queued.

Every submit goes through :func:`acquire_for_submit`. When llama-server is
running with GPU layers (``-ngl`` / ``--n-gpu-layers`` missing or not 0),
Buddy stops it, then reads free VRAM from ``nvidia-smi``. Under the budget,
the submit is refused with a loud message. The server is started again only
after the render, and only on CPU when the card is under 20 GB or
``LLAMACPP_NGL=0``.

The probe and the server controls are replaceable so tests never need a GPU.
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass, field

_LOCK = threading.Lock()
_holds = 0
_restore_ngl: int | None = None
_restore = False

REFUSAL_PREFIX = "SEAT RULE:"


class SeatRefused(RuntimeError):
    """Free VRAM is still under the render budget. Nothing was queued."""


@dataclass
class ServerView:
    running: bool = False
    gpu_layers: int | None = None
    owned: bool = False
    pid: int | None = None
    argv: list[str] = field(default_factory=list)


def reset_state() -> None:
    """Drop hold bookkeeping. Tests only. Does not signal a process."""
    global _holds, _restore_ngl, _restore
    with _LOCK:
        _holds = 0
        _restore_ngl = None
        _restore = False


def holds() -> int:
    return _holds


def llm_must_be_cpu() -> bool:
    """True while a render holds the seat, or when install pinned ``-ngl 0``."""
    if _holds > 0:
        return True
    raw = (os.getenv("LLAMACPP_NGL") or "").strip()
    return raw == "0"


def gpu_layers_from_argv(argv: list[str] | None) -> int | None:
    """``None`` when the flag is absent. llama.cpp then puts layers on the GPU."""
    if not argv:
        return None
    for index, arg in enumerate(argv):
        if arg in ("-ngl", "--n-gpu-layers", "--gpu-layers") and index + 1 < len(argv):
            try:
                return int(argv[index + 1])
            except ValueError:
                return None
        if arg.startswith("--n-gpu-layers=") or arg.startswith("--gpu-layers="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
    return None


def layers_use_gpu(gpu_layers: int | None) -> bool:
    return gpu_layers != 0


def required_free_gb(total_gb: float | None = None) -> float:
    """Free VRAM the render needs after the LLM has left the GPU."""
    raw = (os.getenv("SEAT_MIN_FREE_GB") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    if total_gb is None:
        try:
            from master_agent import config as cfg

            total_gb = float(cfg.VRAM_GB)
        except Exception:
            total_gb = 16.0
    return round(max(6.0, min(float(total_gb) * 0.55, 16.0)), 1)


def probe_free_vram_gb(runner=None) -> float | None:
    """Free GPU memory in GB from ``nvidia-smi``, or None when it cannot be read."""
    run = runner or subprocess.run
    try:
        proc = run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    lines = (getattr(proc, "stdout", "") or "").strip().splitlines()
    if not lines:
        return None
    token = lines[0].split(",")[0].strip()
    try:
        mib = float(token)
    except ValueError:
        return None
    if mib < 0:
        return None
    return round(mib / 1024.0, 2)


def _default_view() -> ServerView:
    from master_agent import llamacpp_server as srv

    owned = srv.owned_running()
    argv = list(srv.owned_argv())
    if owned:
        return ServerView(running=True, gpu_layers=gpu_layers_from_argv(argv), owned=True, argv=argv)
    external = srv.external_view()
    if not external:
        return ServerView()
    ext_argv = list(external.get("argv") or [])
    return ServerView(
        running=True,
        gpu_layers=gpu_layers_from_argv(ext_argv) if ext_argv else external.get("gpu_layers"),
        owned=False,
        pid=external.get("pid"),
        argv=ext_argv,
    )


def _default_stop(view: ServerView) -> None:
    from master_agent import llamacpp_server as srv

    if view.owned:
        srv.stop()
        return
    if view.pid:
        srv.stop_pid(int(view.pid))


def _default_start(ngl: int) -> None:
    from master_agent import llamacpp_server as srv

    srv.start_with_layers(ngl)


def _cpu_after_render() -> bool:
    raw = (os.getenv("LLAMACPP_NGL") or "").strip()
    if raw == "0":
        return True
    try:
        from master_agent import config as cfg

        return float(cfg.VRAM_GB) < 20.0
    except Exception:
        return True


def acquire_for_submit(
    *,
    purpose: str = "ComfyUI render",
    view: ServerView | None = None,
    free_gb: float | None = None,
    free_probe=None,
    stop=None,
    inspect=None,
) -> str:
    """Unload a GPU LLM, then refuse when free VRAM is under the budget.

    Returns a one-line status. Raises :class:`SeatRefused` when the card is
    still too full to queue. A missing ``nvidia-smi`` does not refuse (there
    is no free-memory reading to compare); the LLM is still unloaded when
    Buddy can see GPU layers.
    """
    global _holds, _restore, _restore_ngl
    look = inspect or _default_view
    current = view if view is not None else look()
    stopper = stop or _default_stop
    notes: list[str] = []
    with _LOCK:
        if current.running and layers_use_gpu(current.gpu_layers):
            stopper(current)
            _restore = True
            _restore_ngl = 0 if current.gpu_layers is None else int(current.gpu_layers)
            shown = "default GPU layers" if current.gpu_layers is None else f"-ngl {current.gpu_layers}"
            notes.append(f"stopped llama-server ({shown}) so the render can use the GPU")
        elif current.running:
            notes.append("llama-server is already CPU-only (-ngl 0)")
        else:
            notes.append("llama-server is not running")
        if free_gb is None and free_probe is not None:
            free_gb = free_probe()
        elif free_gb is None and free_probe is None and view is None:
            free_gb = probe_free_vram_gb()
        need = required_free_gb()
        if free_gb is None:
            require = (os.getenv("SEAT_REQUIRE_PROBE") or "").strip().lower() in {"1", "true", "yes"}
            if require:
                raise SeatRefused(
                    f"{REFUSAL_PREFIX} refusing {purpose}. nvidia-smi did not report free VRAM, "
                    "so Buddy cannot prove the GPU is free. Nothing was queued."
                )
            notes.append("nvidia-smi did not report free VRAM; queued without a free-memory reading")
        elif free_gb < need:
            detail = "; ".join(notes)
            raise SeatRefused(
                f"{REFUSAL_PREFIX} refusing {purpose}. Free VRAM is {free_gb:.2f} GB and this "
                f"render needs at least {need:.1f} GB after the LLM leaves the GPU. {detail}. "
                "Nothing was queued. Buddy will not ask you to unload it."
            )
        else:
            notes.append(f"free VRAM {free_gb:.2f} GB meets the {need:.1f} GB budget")
        _holds += 1
    return f"{REFUSAL_PREFIX} {purpose} seated. " + "; ".join(notes)


def release_after_render(*, start=None, free_probe=None) -> str | None:
    """After the render finishes, bring the LLM back if this submit stopped it.

    CPU (``-ngl 0``) when the card is under 20 GB, ``LLAMACPP_NGL=0``, or the
    card is still tight. The previous GPU layer count is used only when the
    card is in the 20 GB class and free memory still has room.
    """
    global _holds, _restore, _restore_ngl
    with _LOCK:
        if _holds <= 0:
            return None
        _holds -= 1
        if _holds > 0 or not _restore:
            return None
        _restore = False
        previous = _restore_ngl
        _restore_ngl = None
    ngl = 0
    if previous and not _cpu_after_render():
        free = free_probe() if free_probe is not None else probe_free_vram_gb()
        if free is None or free >= 4.0:
            ngl = int(previous)
    starter = start or _default_start
    try:
        starter(ngl)
    except Exception as exc:
        return f"{REFUSAL_PREFIX} render finished; llama-server was not restarted ({exc})."
    where = "CPU (-ngl 0)" if ngl == 0 else f"GPU (-ngl {ngl})"
    return f"{REFUSAL_PREFIX} render finished; llama-server restarted on {where}."
