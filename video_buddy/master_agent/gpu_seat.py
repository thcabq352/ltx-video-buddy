"""Seat rule: a ComfyUI submit needs free VRAM and does not take it by force.

Every submit goes through :func:`acquire_for_submit`. Buddy reads free VRAM
from ``nvidia-smi`` and refuses when the card is under the budget. A running
llama-server is left alone. Buddy does not stop, kill, unload, or restart it
to free the GPU. If the server is on the GPU and free VRAM is unknown or too
low, the submit is refused and nothing is queued.

The probe is replaceable so tests never need a GPU.
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass, field

_LOCK = threading.Lock()
_holds = 0

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
    global _holds
    with _LOCK:
        _holds = 0


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


def acquire_for_submit(
    *,
    purpose: str = "ComfyUI render",
    view: ServerView | None = None,
    free_gb: float | None = None,
    free_probe=None,
    stop=None,
    inspect=None,
) -> str:
    """Refuse when free VRAM is under the budget. Never stop a server.

    Returns a one-line status. Raises :class:`SeatRefused` when the card is
    too full, when a GPU llama-server is running and free VRAM cannot be
    read, or when the probe fails. ``stop`` is ignored: a caller cannot
    inject a process kill on this path.
    """
    global _holds
    del stop
    look = inspect or _default_view
    current = view if view is not None else look()
    gpu_resident = bool(current.running and layers_use_gpu(current.gpu_layers))
    notes: list[str] = []
    if gpu_resident:
        shown = "default GPU layers" if current.gpu_layers is None else f"-ngl {current.gpu_layers}"
        notes.append(f"llama-server stays running ({shown}); Buddy will not stop it")
    elif current.running:
        notes.append("llama-server is already CPU-only (-ngl 0)")
    else:
        notes.append("llama-server is not running")
    with _LOCK:
        try:
            measured = free_gb
            if measured is None and free_probe is not None:
                measured = free_probe()
            elif measured is None and view is None:
                measured = probe_free_vram_gb()
        except Exception as exc:
            raise SeatRefused(
                f"{REFUSAL_PREFIX} refusing {purpose}. Free VRAM could not be read "
                f"({type(exc).__name__}: {exc}). No server was stopped. Nothing was queued."
            ) from exc
        need = required_free_gb()
        if measured is None and gpu_resident:
            raise SeatRefused(
                f"{REFUSAL_PREFIX} refusing {purpose}. llama-server is using the GPU and "
                "nvidia-smi did not report free VRAM, so Buddy cannot prove the card has room "
                "without stopping that server. No server was stopped. Nothing was queued. "
                "Move the LLM to CPU (LLAMACPP_NGL=0) or free VRAM yourself, then retry."
            )
        if measured is None:
            require = (os.getenv("SEAT_REQUIRE_PROBE") or "").strip().lower() in {"1", "true", "yes"}
            if require:
                raise SeatRefused(
                    f"{REFUSAL_PREFIX} refusing {purpose}. nvidia-smi did not report free VRAM, "
                    "so Buddy cannot prove the GPU is free. No server was stopped. Nothing was queued."
                )
            notes.append("nvidia-smi did not report free VRAM; queued without a free-memory reading")
        elif float(measured) < need:
            detail = "; ".join(notes)
            raise SeatRefused(
                f"{REFUSAL_PREFIX} refusing {purpose}. Free VRAM is {float(measured):.2f} GB and this "
                f"render needs at least {need:.1f} GB. {detail}. No server was stopped. "
                "Nothing was queued."
            )
        else:
            notes.append(f"free VRAM {float(measured):.2f} GB meets the {need:.1f} GB budget")
        _holds += 1
    return f"{REFUSAL_PREFIX} {purpose} seated. " + "; ".join(notes)


def release_after_render(*, start=None, free_probe=None) -> str | None:
    """Drop one seat hold. Does not start, stop, or restart any server.

    ``start`` and ``free_probe`` are ignored so this path cannot restart a
    service that was left running.
    """
    del start, free_probe
    global _holds
    with _LOCK:
        if _holds <= 0:
            return None
        _holds -= 1
        if _holds > 0:
            return None
    return f"{REFUSAL_PREFIX} render finished; no server was restarted."
