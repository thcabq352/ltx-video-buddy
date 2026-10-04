"""OOM retry for a queued Comfy job.

Walks the variant downscale ladder instead of failing the run on the first
CUDA OOM. The Orchestrator stays the coordinator and calls ``handle_job_error``.
"""

from __future__ import annotations

import re

from master_agent.config import DOWNSCALE_LADDER, MAX_RETRIES
from master_agent.orchestrator.state import RunState

_OOM_PATTERN = re.compile(
    r"out of memory|cuda oom|CUDAOOM|allocation on device|OOM", re.IGNORECASE
)


def looks_oom(exc: Exception) -> bool:
    return bool(_OOM_PATTERN.search(str(exc) or ""))


def handle_job_error(orch, st: RunState, exc: Exception, *, phase: str) -> bool:
    if looks_oom(exc):
        st.retries += 1
        try:
            from master_agent.models.vram_policy import downscale_ladder_for

            ladder = downscale_ladder_for(st.variant or "")
        except Exception:
            ladder = DOWNSCALE_LADDER
        if st.retries > MAX_RETRIES or st.downscale_level + 1 >= len(ladder):
            st.fail(f"{phase} OOM after {st.retries} retries: {exc}")
            return False
        st.transition("PLAN_OOM_RETRY")
        st.downscale_level += 1
        w, h, frames = ladder[st.downscale_level]
        st.width, st.height = w, h
        st.log(f"OOM at {phase}: downscaling to {w}x{h} ({frames}f), retry {st.retries}")
        return orch._patch(st) and orch._validate(st) and orch._submit_and_poll(st)
    st.fail(f"{phase} failed: {exc}")
    return False

