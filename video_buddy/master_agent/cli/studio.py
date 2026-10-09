"""Curriculum, about, knowledge base, studio UI, budget, and Hermes commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def cmd_curriculum(args: argparse.Namespace) -> int:
    from master_agent.curriculum import curriculum_card, format_curriculum

    card = curriculum_card()
    if args.json:
        print(json.dumps(card, indent=1))
        return 0
    print(format_curriculum(card))
    return 0


def cmd_about(args: argparse.Namespace) -> int:
    from master_agent.about import format_about, studio_about

    card = studio_about()
    if args.json:
        print(json.dumps(card, indent=1))
        return 0
    print(format_about(card), end="")
    return 0


def cmd_kb(args: argparse.Namespace) -> int:
    from master_agent.kb.ingest import ingest_all_runs, ingest_knowledge, ingest_workflows
    from master_agent.kb.store import (
        COLLECTION_KNOWLEDGE,
        COLLECTION_RUNS,
        COLLECTION_WORKFLOWS,
        collection_count,
        kb_available,
        search,
    )

    if not kb_available():
        print("FAIL  knowledge base unavailable (KB_ENABLED=0 or chromadb missing)")
        return 1

    if args.kb_command == "ingest":
        n_wf = ingest_workflows()
        n_runs = ingest_all_runs()
        knowledge = ingest_knowledge()
        print(
            f"OK    ingested {n_wf} workflow(s), {n_runs} run record(s), "
            f"{knowledge.docs} knowledge doc(s)"
        )
        if knowledge.skipped:
            print(f"      skipped {knowledge.skipped} knowledge file(s) (schema, EXAMPLE, or secret)")
        print(
            f"      workflows={collection_count(COLLECTION_WORKFLOWS)} "
            f"runs={collection_count(COLLECTION_RUNS)} "
            f"knowledge={collection_count(COLLECTION_KNOWLEDGE)}"
        )
        return 0

    if args.kb_command == "search":
        if not args.query:
            print("FAIL  pass a query: kb search \"coffee ad\"")
            return 2
        if args.knowledge:
            coll = COLLECTION_KNOWLEDGE
        elif args.workflows:
            coll = COLLECTION_WORKFLOWS
        else:
            coll = COLLECTION_RUNS
        hits = search(coll, args.query, k=args.k)
        if not hits:
            print("no hits")
            return 0
        for h in hits:
            m = h.get("metadata") or {}
            dist = h.get("distance")
            print(
                f"- [{dist:.3f}] {h.get('id')} "
                f"{m.get('request') or m.get('path') or m.get('name') or ''}"
            )
            first = (h.get("text") or "").splitlines()[0] if h.get("text") else ""
            print(f"    {first[:160]}")
        return 0

    if args.kb_command == "stats":
        print(
            f"workflows={collection_count(COLLECTION_WORKFLOWS)} "
            f"runs={collection_count(COLLECTION_RUNS)} "
            f"knowledge={collection_count(COLLECTION_KNOWLEDGE)}"
        )
        return 0

    return 2


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("FAIL  uvicorn not installed (pip install uvicorn fastapi)")
        return 1
    from master_agent.web.app import app
    from master_agent.web.local_guard import LocalOnlyGuard, is_loopback_host

    allow_remote = bool(getattr(args, "allow_remote", False))
    if not allow_remote and not is_loopback_host(args.host):
        print(
            f"FAIL  --host {args.host} is not loopback. The studio has no auth; "
            "pass --allow-remote to expose it anyway."
        )
        return 2
    serve = app if allow_remote else LocalOnlyGuard(app)
    if allow_remote:
        print("WARN  --allow-remote: no auth, no Host/Origin check. Anyone who can reach this port can queue renders.")
    print(f"VIDEO BUDDY studio: http://{args.host}:{args.port}")
    print("  Voice chat: open in Chrome/Edge → Voice tab (mic + spoken replies)")
    uvicorn.run(serve, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    from master_agent.control.budget import get_project_budget

    store = get_project_budget()
    if args.budget_command == "reset-shift":
        row = store.reset_shift()
        print(
            f"OK    shift reset previous_shift_id={row['previous_shift_id']} "
            f"previous_used={row['previous_used']} now={store.shift_id}"
        )
        from master_agent.orchestrator.pipeline import resume_after_budget_clear

        continued = resume_after_budget_clear(background=False)
        if continued:
            print("OK    continuing paused generate: " + ", ".join(continued))
    snap = store.snapshot()
    if args.json:
        print(json.dumps(snap, indent=1))
        return 0
    print(
        f"budget  used={snap['used']} cap={snap['cap']} paused={snap['paused']} "
        f"shift_id={snap['shift_id']} started={snap['shift_started_at']}"
    )
    print(f"        pending={len(snap['pending'])} ledger={len(snap['log'])}")
    return 0


def cmd_hermes(args: argparse.Namespace) -> int:
    from master_agent.hermes.gateways import discover_gateways, discover_primary_seat
    from master_agent.hermes.profile import default_python, register_ltx_profile

    video_buddy_root = Path(__file__).resolve().parent.parent
    home = Path(args.hermes_home).expanduser() if getattr(args, "hermes_home", None) else None
    if args.hermes_command == "register":
        result = register_ltx_profile(
            home=home,
            video_buddy_root=video_buddy_root,
            python=default_python(video_buddy_root),
            force=bool(getattr(args, "force", False)),
        )
        print(f"OK    ltx profile at {result.profile_dir}")
        if result.soul_written:
            print("      SOUL.md written from LTX_RESEARCH_SYSTEM")
        if result.soul_skipped:
            print("      custom SOUL.md left in place (pass --force to overwrite)")
        for note in result.notes:
            print(f"      {note}")
        return 0

    rows = discover_gateways(home=home, host="127.0.0.1")
    primary = discover_primary_seat(home=home, host="127.0.0.1")
    payload = {
        "primary": None
        if primary is None
        else {
            "profile": primary.profile,
            "source": primary.source,
            "port": primary.port,
            "chat_url": primary.chat_url,
            "healthy": primary.healthy,
        },
        "gateways": [
            {
                "profile": g.profile,
                "source": g.source,
                "port": g.port,
                "chat_url": g.chat_url,
                "healthy": g.healthy,
                "can_speak": g.can_speak,
            }
            for g in rows
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=1))
        return 0
    if primary:
        print(
            f"primary  {primary.profile} source={primary.source} "
            f"healthy={primary.healthy} {primary.chat_url}"
        )
    else:
        print("primary  (none)")
    for g in rows:
        mark = "*" if primary is not None and g.chat_url == primary.chat_url and g.source == primary.source else " "
        print(
            f"{mark} {g.profile:12} {g.source:14} port={g.port:<5} "
            f"healthy={str(g.healthy):5} {g.chat_url}"
        )
    return 0

