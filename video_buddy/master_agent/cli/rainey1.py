"""CLI body for ``python -m master_agent rainey1-batch``."""

from __future__ import annotations

import argparse


def cmd_rainey1_batch(args: argparse.Namespace) -> int:
    """Batch Rainey1 seeds. --dry-run queues zero Comfy jobs."""
    from master_agent.rainey1.batch import format_batch_report, run_rainey1_batch

    try:
        result = run_rainey1_batch(
            recipe=args.recipe,
            seeds=args.seeds,
            top_k=args.top_k,
            out=args.out,
            dry_run=bool(args.dry_run),
            allow_empty=bool(args.allow_empty),
            llm_judge=bool(args.llm_judge),
        )
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(format_batch_report(result))
    return result.exit_code
