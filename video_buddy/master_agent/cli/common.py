"""Shared argparse helpers for the VIDEO BUDDY CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _weight_mode(args: argparse.Namespace) -> str:
    if getattr(args, "scan_only", False):
        return "scan"
    if getattr(args, "use_existing", False):
        return "existing"
    if getattr(args, "download", False):
        return "download"
    return "report"


class _DurationSet(argparse.Action):
    """Record that the user passed --duration (brief parsing must not override it)."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "duration_set", True)


class _DimSet(argparse.Action):
    """Record that the user passed --width or --height (outpaint canvas)."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, int(values))
        setattr(namespace, f"{self.dest}_set", True)


def _inoutpaint_plan(args: argparse.Namespace):
    """Outpaint layout, or None when this run is not extending the canvas."""
    aspect = getattr(args, "aspect", None)
    if not getattr(args, "outpaint", False) and not aspect:
        return None
    video = getattr(args, "video", None)
    if not video:
        raise ValueError("outpaint needs --video (the clip to extend)")
    from master_agent.comfy.inoutpaint import outpaint_layout, probe_video_size

    src_w, src_h = probe_video_size(Path(video))
    if aspect:
        return outpaint_layout(src_w, src_h, aspect=str(aspect))
    if getattr(args, "width_set", False) or getattr(args, "height_set", False):
        return outpaint_layout(
            src_w,
            src_h,
            target_w=getattr(args, "width", None),
            target_h=getattr(args, "height", None),
        )
    raise ValueError(
        "outpaint needs --aspect W:H (for example 9:16) "
        "or --width and --height as the target canvas"
    )


def _reframe_flag(args: argparse.Namespace) -> bool:
    """True only for ``--reframe on``. Default is off (experimental opt-in)."""
    from master_agent.config import LIPDUB_REFRAME

    value = getattr(args, "reframe", None)
    if value is None:
        return bool(LIPDUB_REFRAME)
    return str(value).strip().lower() == "on"


def _music_intent(request: str, audio_path: str, quality: str | None) -> bool:
    """Auto-routing rule: music keywords, or audio longer than one segment."""
    from master_agent.config import get_quality_profile
    from master_agent.music.beats import audio_duration
    from master_agent.music.pipeline import MUSIC_KEYWORDS

    text = request.lower()
    if any(kw in text for kw in MUSIC_KEYWORDS):
        return True
    duration = audio_duration(audio_path)
    cap = float(get_quality_profile(quality)["segment_max_s"])
    return duration > cap


def _maybe_interview(request: str, *, no_interview: bool = False) -> str:
    """Pre-generation intake interview (default on for interactive sessions)."""
    from master_agent.config import INTAKE_ENABLED

    if no_interview or not INTAKE_ENABLED or not request.strip():
        return request
    if not sys.stdin.isatty():
        return request
    from master_agent.persona.intake import run_interview_cli

    brief = run_interview_cli(request)
    return brief.to_request()


def _parse_override_flags(flags: list[str]) -> dict[str, dict]:
    overrides: dict[str, dict] = {}
    for raw in flags or []:
        if "=" not in raw or "." not in raw.split("=", 1)[0]:
            raise ValueError(f"override must look like NODE.FIELD=VALUE, got {raw!r}")
        left, value = raw.split("=", 1)
        node_id, field = left.split(".", 1)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        overrides.setdefault(node_id, {})[field] = parsed
    return overrides


def _add_selector_flags(parser: argparse.ArgumentParser, *, include_optional: bool = True) -> None:
    """Flags shared by ``models select`` and ``download-models --selector``.

    ``download-models`` already uses ``--optional`` as a store_true for the
    pack fetch. That parser omits the selector's repeatable ``--optional``
    and treats its existing flag as ``--all-optional`` when ``--selector`` is set.
    """
    parser.add_argument(
        "--version",
        dest="ltx_version",
        choices=["2.3", "2.5"],
        help="LTX radio. Swaps the whole video-generation checklist.",
    )
    if include_optional:
        parser.add_argument(
            "--optional",
            action="append",
            default=None,
            help="optional row id (repeatable, or comma-separated). Required rows stay on.",
        )
    parser.add_argument(
        "--all-optional",
        action="store_true",
        help="select every optional video row for the chosen version",
    )
    parser.add_argument(
        "--soundtrack",
        action="store_true",
        help="Enable Soundtrack Studio (existing download-models --heartmula consent path)",
    )
    parser.add_argument(
        "--mbps",
        type=float,
        default=50.0,
        help="assumed link speed in megabits/s for the download ETA (default 50)",
    )
    switch = parser.add_mutually_exclusive_group()
    switch.add_argument(
        "--keep",
        action="store_true",
        help="when switching versions, keep the previous pack on disk",
    )
    switch.add_argument(
        "--wipe",
        action="store_true",
        help="when switching versions, delete the previous pack under MODELS_DIR",
    )

