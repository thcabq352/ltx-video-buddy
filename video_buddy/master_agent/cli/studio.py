"""Curriculum, about, knowledge base, studio UI, budget, and Hermes commands."""

from __future__ import annotations

import argparse
import json

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
