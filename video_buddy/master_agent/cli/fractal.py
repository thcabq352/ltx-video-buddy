"""CPU fractal video command."""

from __future__ import annotations

import argparse

from master_agent.cli.common import _maybe_interview

def cmd_fractal(args: argparse.Namespace) -> int:
    from master_agent.config import FRACTAL_DEFAULTS
    from master_agent.fractal.pipeline import run_fractal

    mode = getattr(args, "mode", None) or "zoom"
    if mode in ("inpaint", "outpaint") and not args.image:
        print(f"FAIL  fractal --mode {mode} needs --image <file>")
        return 2
    request = _maybe_interview(args.request or "", no_interview=args.no_interview)
    rec = run_fractal(
        request,
        mode=mode,
        duration_s=args.duration,
        fps=args.fps or FRACTAL_DEFAULTS["fps"],
        width=args.width,
        height=args.height,
        target=args.target,
        palette=args.palette,
        seed=args.seed,
        julia=args.julia,
        audio_path=args.audio,
        image_path=args.image,
        mask_path=getattr(args, "mask", None),
        expand=getattr(args, "expand", 128),
        cover=getattr(args, "cover", 0.4),
        feather=getattr(args, "feather", 28),
    )
    video = rec["video_path"]
    if args.upscale:
        try:
            from master_agent.upscale import upscale_video

            video = str(upscale_video(video, method=args.upscale, run_id=rec["run_id"]))
            print(f"upscaled: {video}")
        except Exception as e:
            print(f"WARN  upscale failed: {e}")
    print(f"DONE   video: {video}")
    return 0

