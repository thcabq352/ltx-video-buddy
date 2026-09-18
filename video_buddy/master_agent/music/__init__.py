"""Beat-synced music video pipeline (dependency-free beat detection).

Classic ``music`` CLI still muxes with ffmpeg. MTV / Remotion mode lives in
``plan`` / ``burn`` / ``unique`` / ``remotion`` / ``mv`` — see
``docs/MUSIC_VIDEO.md``.
"""

from master_agent.music.plan import BEAT_PLAN_SCHEMA, BeatPlan, build_beat_plan
from master_agent.music.unique import DuplicateClipError, check_unique_clips

__all__ = [
    "BEAT_PLAN_SCHEMA",
    "BeatPlan",
    "DuplicateClipError",
    "build_beat_plan",
    "check_unique_clips",
]
