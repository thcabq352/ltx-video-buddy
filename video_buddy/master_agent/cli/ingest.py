"""CLI verbs for workflow ingest / learn / dry-run / run / promote."""

from __future__ import annotations

import argparse
from pathlib import Path

from master_agent.comfy.ingest.normalize import IngestError
from master_agent.comfy.ingest.run import dry_run_slug, format_dry_run, run_ingested
from master_agent.comfy.ingest.store import ingest_file, ingest_history, refresh_learned
from master_agent.comfy.ingest.validate import MissingCustomNodeError, MissingModelError
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
        "no_family_route": bool(getattr(args, "no_family_route", False)),
    }


def _llm(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "llm_assist", False))


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
    readiness = learned.get("readiness") or {}
    packs = {
        str(row.get("class_type")): row.get("pack")
        for row in (readiness.get("missing_node_packs") or [])
        if isinstance(row, dict)
    }
    missing = readiness.get("missing_nodes") or []
    if missing:
        rendered = []
        for item in missing:
            pack = packs.get(str(item))
            rendered.append(f"{item} ({pack})" if pack else str(item))
        listed = ", ".join(rendered)
        print(
            f"WARN  missing custom node(s): {listed}. "
            "Buddy does not auto-install custom nodes."
        )
    models = readiness.get("missing_models") or []
    if models:
        listed = ", ".join(str(item) for item in models)
        print(
            f"WARN  missing model(s): {listed}. "
            "Buddy does not substitute another weight. "
            "python -m master_agent doctor. "
            "python -m master_agent download-models."
        )
    for pointer in readiness.get("pointers") or []:
        print(f"pointer: {pointer}")
    if learned.get("llm_warning"):
        print(f"WARN  {learned['llm_warning']}")
    for proposal in learned.get("llm_proposals") or []:
        print(
            f"llm: {proposal.get('from')} -> {proposal.get('role')} "
            f"[{proposal.get('confidence')}] source {proposal.get('source')}"
        )
    if learned.get("family_warning"):
        print(f"WARN  {learned['family_warning']}")
    elif learned.get("family"):
        print(f"family: {learned['family']} ({learned.get('family_route')})")
    for warning in learned.get("warnings") or []:
        print(f"WARN  low confidence {warning.get('role')}: {warning.get('detail')}")


def _history_prompt_id(args: argparse.Namespace) -> str | None:
    raw = getattr(args, "ingest_from", None)
    if not raw:
        return None
    text = str(raw).strip()
    if not text.startswith("history:"):
        raise IngestError(
            f"unsupported --from {text!r}. Use history:PROMPT_ID or a JSON path."
        )
    prompt_id = text.split(":", 1)[1].strip()
    if not prompt_id:
        raise IngestError("history ingest requires history:PROMPT_ID.")
    return prompt_id


def cmd_ingest(args: argparse.Namespace) -> int:
    target = getattr(args, "target", None) or getattr(args, "workflow_json", None)
    try:
        prompt_id = _history_prompt_id(args)
    except IngestError as exc:
        return _fail(exc)
    if prompt_id:
        try:
            result = ingest_history(
                prompt_id,
                slug=getattr(args, "slug", None),
                no_family_route=bool(getattr(args, "no_family_route", False)),
                llm_assist=_llm(args),
            )
        except (IngestError, OSError) as exc:
            return _fail(exc)
    else:
        if not target:
            print("FAIL  comfy ingest requires PATH.json or --from history:PROMPT_ID")
            return 1
        try:
            from master_agent.comfy.client import ComfyClient

            result = ingest_file(
                Path(target),
                slug=getattr(args, "slug", None),
                client=ComfyClient(),
                no_family_route=bool(getattr(args, "no_family_route", False)),
                llm_assist=_llm(args),
            )
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
        bundle = refresh_learned(
            str(slug),
            no_family_route=bool(getattr(args, "no_family_route", False)),
            llm_assist=_llm(args),
        )
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
    except (IngestError, MissingCustomNodeError, MissingModelError, TinyVAETiledDecodeError) as exc:
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
    except (IngestError, MissingCustomNodeError, MissingModelError, TinyVAETiledDecodeError) as exc:
        return _fail(exc)
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1
    print(report.get("text") or format_dry_run(report))
    if report.get("prompt_id"):
        print(f"queued prompt_id={report['prompt_id']}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    slug = getattr(args, "target", None) or getattr(args, "slug", None)
    if not slug:
        print("FAIL  comfy promote requires SLUG")
        return 1
    from master_agent.comfy.ingest.promote import format_promote, promote_slug

    try:
        result = promote_slug(
            str(slug),
            variant_name=getattr(args, "variant_name", None),
            force=bool(getattr(args, "force", False)),
        )
    except (IngestError, OSError) as exc:
        return _fail(exc)
    print(format_promote(result), end="")
    return 0


def dispatch_ingest_command(args: argparse.Namespace) -> int:
    command = getattr(args, "comfy_command", "")
    if command == "ingest":
        return cmd_ingest(args)
    if command == "learn":
        return cmd_learn(args)
    if command == "dry-run":
        return cmd_dry_run(args)
    if command == "promote":
        return cmd_promote(args)
    if command == "run":
        return cmd_run_ingested(args)
    print(f"FAIL  unknown comfy command {command}")
    return 1
