"""CLI verbs for Phase A workflow ingest / learn / dry-run / run."""

from __future__ import annotations

import argparse
from pathlib import Path

from master_agent.comfy.ingest.normalize import IngestError
from master_agent.comfy.ingest.run import dry_run_slug, format_dry_run, run_ingested
from master_agent.comfy.ingest.store import ingest_file, refresh_learned
from master_agent.comfy.ingest.validate import MissingCustomNodeError
from master_agent.comfy.vae_guard import TinyVAETiledDecodeError


def _params(args: argparse.Namespace) -> dict:
    return {
        "prompt": getattr(args, "prompt", None),
        "negative_prompt": getattr(args, "negative_prompt", None),
        "seed": getattr(args, "seed", None),
        "width": getattr(args, "width", None),
        "height": getattr(args, "height", None),
        "frames": getattr(args, "frames", None),
        "vae": getattr(args, "vae", None),
    }


def _node_overrides(args: argparse.Namespace) -> dict:
    flags = getattr(args, "set", None) or []
    if not flags:
        return {}
    from master_agent.cli.common import _parse_override_flags

    return _parse_override_flags(list(flags))


def _fail(exc: BaseException) -> int:
    print(f"FAIL  {exc}")
    return 1


def _print_learned_warnings(learned: dict) -> None:
    missing = (learned.get("readiness") or {}).get("missing_nodes") or []
    if missing:
        listed = ", ".join(str(item) for item in missing)
        print(
            f"WARN  missing custom node(s): {listed}. "
            "Buddy does not auto-install custom nodes."
        )
    for warning in learned.get("warnings") or []:
        print(f"WARN  low confidence {warning.get('role')}: {warning.get('detail')}")


def cmd_ingest(args: argparse.Namespace) -> int:
    target = getattr(args, "target", None) or getattr(args, "workflow_json", None)
    if not target:
        print("FAIL  comfy ingest requires PATH.json")
        return 1
    try:
        result = ingest_file(Path(target), slug=getattr(args, "slug", None))
    except (IngestError, OSError) as exc:
        return _fail(exc)
    learned = result["learned"]
    print(f"ingested slug={result['slug']}")
    print(f"wrote {result['dir'] / 'workflow_api.json'}")
    print(f"wrote {result['dir'] / 'learned.yaml'}")
    print(f"wrote {result['dir'] / 'provenance.json'}")
    _print_learned_warnings(learned)
    if getattr(args, "queue", False):
        print("queue: ingest --queue")
        return cmd_run_ingested(args, slug=result["slug"])
    print(f"next: {result['next']}")
    print("default next action is dry-run; this command did not queue")
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    slug = getattr(args, "target", None) or getattr(args, "slug", None)
    if not slug:
        print("FAIL  comfy learn requires SLUG")
        return 1
    try:
        bundle = refresh_learned(str(slug))
    except (IngestError, OSError) as exc:
        return _fail(exc)
    _print_learned_warnings(bundle["learned"])
    text = (bundle["dir"] / "learned.yaml").read_text(encoding="utf-8")
    print(text, end="" if text.endswith("\n") else "\n")
    print(f"next: python -m master_agent comfy dry-run {bundle['slug']}")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    slug = getattr(args, "target", None) or getattr(args, "slug", None)
    if not slug:
        print("FAIL  comfy dry-run requires SLUG")
        return 1
    try:
        report = dry_run_slug(str(slug), node_overrides=_node_overrides(args), **_params(args))
    except (IngestError, MissingCustomNodeError, TinyVAETiledDecodeError) as exc:
        return _fail(exc)
    print(report["text"])
    return 0


def cmd_run_ingested(args: argparse.Namespace, slug: str | None = None) -> int:
    chosen = slug or getattr(args, "ingested", None) or getattr(args, "target", None)
    if not chosen:
        print("FAIL  comfy run --ingested requires SLUG")
        return 1
    try:
        report = run_ingested(
            str(chosen),
            node_overrides=_node_overrides(args),
            **_params(args),
        )
    except (IngestError, MissingCustomNodeError, TinyVAETiledDecodeError) as exc:
        return _fail(exc)
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1
    print(report.get("text") or format_dry_run(report))
    if report.get("prompt_id"):
        print(f"queued prompt_id={report['prompt_id']}")
    return 0


def dispatch_ingest_command(args: argparse.Namespace) -> int:
    command = getattr(args, "comfy_command", "")
    if command == "ingest":
        return cmd_ingest(args)
    if command == "learn":
        return cmd_learn(args)
    if command == "dry-run":
        return cmd_dry_run(args)
    if command == "run":
        return cmd_run_ingested(args)
    print(f"FAIL  unknown comfy command {command}")
    return 1
