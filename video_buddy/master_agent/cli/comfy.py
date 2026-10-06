"""Comfy drive, validate, diagnose, capabilities, and power-tune commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from master_agent.cli.runtime import comfy_client
from master_agent.comfy.validator import format_report, validate_workflow_file
from master_agent.config import WORKFLOWS_DIR
from master_agent.cli.common import _inoutpaint_plan, _parse_override_flags

def cmd_fetch_object_info(args: argparse.Namespace) -> int:
    client = comfy_client()
    try:
        info = client.refresh_object_info_cache()
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    print(f"OK    cached {len(info)} node classes to state/object_info.json")
    return 0


def cmd_capabilities(args: argparse.Namespace) -> int:
    from master_agent.comfy.capabilities import format_matrix, run_probe
    from master_agent.config import WORKFLOW_FILES

    prefer_live = not bool(args.offline)
    try:
        rows, source = run_probe(prefer_live=prefer_live)
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "object_info": source,
                    "director_allowlist": sorted(WORKFLOW_FILES),
                    "rows": [r.to_dict() for r in rows],
                },
                indent=1,
            )
        )
        return 0
    print(format_matrix(rows, source=source), end="")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    targets: list[Path] = []
    if args.all:
        if not WORKFLOWS_DIR.is_dir():
            print(f"FAIL  workflows dir not found: {WORKFLOWS_DIR}")
            return 1
        targets = sorted(WORKFLOWS_DIR.glob("*.json"))
        if not targets:
            print(f"FAIL  no *.json in {WORKFLOWS_DIR}")
            return 1
    elif args.file:
        targets = [Path(args.file)]
    else:
        print("FAIL  pass a workflow file or --all")
        return 2

    client = comfy_client()
    reports = []
    for path in targets:
        if not path.is_file():
            print(f"FAIL  {path} not found")
            return 1
        try:
            report = validate_workflow_file(
                path,
                client=client,
                prefer_live=not args.offline,
                strict=bool(getattr(args, "strict", False)),
            )
        except Exception as e:
            print(f"FAIL  {path}: {e}")
            return 1
        reports.append(report)

    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=1))
    else:
        for report in reports:
            print(format_report(report))
            print()
    return 0 if all(r.ok for r in reports) else 1


def cmd_power_tune(args: argparse.Namespace) -> int:
    """Dry-run power mode: heuristic patch + LLM graph ops + validate (no GPU)."""
    from master_agent.comfy.power_mode import power_tune
    from master_agent.comfy.workflow_patcher import load_and_patch_workflow
    from master_agent.config import get_quality_profile

    profile = get_quality_profile(args.quality)
    try:
        wf, meta = load_and_patch_workflow(
            args.variant,
            prompt=args.request,
            duration_s=args.duration,
            seed=args.seed,
            steps=profile.get("steps"),
            width=int(profile.get("max_width") or 768),
            height=int(profile.get("max_height") or 512),
        )
    except Exception as e:
        print(f"FAIL  patch: {e}")
        return 1

    result = power_tune(
        wf,
        request=args.request,
        provider=args.provider,
        log=print,
    )
    payload = result.to_dict()
    payload["variant"] = args.variant
    payload["patch_meta"] = {k: meta.get(k) for k in ("seed", "steps", "cfg", "width", "height", "frames") if k in meta}

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.workflow, indent=1) + "\n", encoding="utf-8")
        print(f"wrote workflow: {out}")

    if args.json:
        print(json.dumps(payload, indent=1, default=str))
    else:
        print(f"reason:  {result.reason or '(none)'}")
        print(f"ops:     {len(result.ops)} proposed, {len(result.applied)} applied")
        print(f"valid:   {result.valid}")
        print(f"rag:     {result.rag_used}")
        if result.error:
            print(f"error:   {result.error}")
        for e in result.validation_errors[:8]:
            print(f"  ERR {e}")
        for o in result.applied[:12]:
            print(f"  + {o}")
    return 0 if result.valid or not result.ops else 2


def cmd_diagnose(args: argparse.Namespace) -> int:
    from master_agent.comfy.diagnose import DiagnoseFailed, ScaleRefused, run_diagnose

    try:
        rec = run_diagnose(
            prompt=args.prompt,
            variant=args.variant,
            steps=args.steps,
            seed=args.seed,
            prepare_only=bool(args.prepare),
            width=args.width,
            height=args.height,
        )
    except ScaleRefused as e:
        print(f"FAIL  {e}")
        return 2
    except DiagnoseFailed as e:
        print(f"FAIL  {e}")
        return 1
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    if args.json:
        print(json.dumps({k: v for k, v in rec.items() if k != "meta"}, indent=1, default=str))
    return 0 if rec.get("ok") else 1


def cmd_comfy_attach(args: argparse.Namespace) -> int:
    """Patch a WorkflowPatchPlan / buddy.comfy.attach/v1 recipe onto a graph."""
    from master_agent.comfy.attach import AttachError, load_attach_recipe, run_attach
    from master_agent.comfy.cli_run import prepare_run, unwrap_workflow

    recipe_path = getattr(args, "recipe", None)
    if not recipe_path:
        print("FAIL  comfy attach requires --recipe PATH")
        return 2
    try:
        recipe = load_attach_recipe(recipe_path)
    except AttachError as e:
        print(f"FAIL  {e}")
        return 1
    try:
        workflow = None
        if args.workflow_json:
            p = Path(args.workflow_json)
            raw = p.read_text(encoding="utf-8") if p.is_file() else args.workflow_json
            workflow = unwrap_workflow(json.loads(raw))
        else:
            preferred = recipe.preferred_variants[0] if recipe.preferred_variants else "base"
            workflow = prepare_run(
                getattr(args, "mode", None) or "generate",
                template_path=args.template,
                variant=args.variant or preferred,
                prompt=args.prompt or "",
            )
    except Exception as e:
        print(f"FAIL  workflow: {e}")
        return 1
    submit = bool(getattr(args, "submit", False))
    if submit and getattr(args, "dry_run", False):
        print("FAIL  --submit and --dry-run are mutually exclusive")
        return 2
    runs_dir = getattr(args, "runs_dir", None)
    try:
        rec = run_attach(
            recipe=recipe,
            workflow=workflow,
            client=comfy_client(),
            submit=submit,
            runs_dir=Path(runs_dir) if runs_dir else None,
            variant=args.variant,
        )
    except AttachError as e:
        print(f"FAIL  {e}")
        return 1
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec.get("workflow") or {}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    printable = {k: v for k, v in rec.items() if k != "workflow"}
    print(json.dumps(printable, indent=1, default=str))
    return 0 if rec.get("ok") else 1


def _cmd_comfy_blaze_remake(args: argparse.Namespace) -> int:
    """Named concert / pier remakes. Prepare does not queue and does not invent a clip."""
    from master_agent.comfy.blaze_remake import (
        prepare_blaze_remake,
        provenance_for_clip,
        upload_remake_stills,
    )
    from master_agent.comfy.cli_run import LintError, execute_prepared, lint_or_raise
    from master_agent.provenance import missing_required, write_clip_provenance

    if getattr(args, "mode", "generate") != "generate":
        print("FAIL  blaze remake uses --mode generate")
        return 1
    scott = getattr(args, "scott", None)
    blaze = getattr(args, "blaze", None)
    if not scott or not blaze:
        print("FAIL  blaze remake requires --scott and --blaze")
        return 1
    variant = getattr(args, "variant", None) or "base"
    if variant not in {"base", "ltx25_msr", "msr"}:
        print(f"WARN  blaze remake locks variant ltx25_msr (ignored {variant})")
    try:
        prepared = prepare_blaze_remake(
            args.recipe,
            scott=scott,
            blaze=blaze,
            prompt=getattr(args, "prompt", "") or "",
            negative=getattr(args, "negative_prompt", None),
            width=getattr(args, "width", None),
            height=getattr(args, "height", None),
            frames=getattr(args, "frames", None),
            seed=getattr(args, "seed", None),
        )
    except ValueError as exc:
        print(f"FAIL  {exc}")
        return 1
    if prepared.warning:
        print(prepared.warning)
    workflow = prepared.workflow
    missing = missing_required(prepared.provenance)
    if missing:
        print(f"FAIL  provenance missing {missing}")
        return 1
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(workflow, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    if args.prepare:
        try:
            client = comfy_client()
            object_info, _src = client.load_object_info(prefer_live=True)
            lint_or_raise(workflow, object_info)
        except LintError as exc:
            print(f"FAIL  {exc}")
            return 1
        except Exception as exc:
            print(f"WARN  linter skipped ({exc})")
        sidecar = ""
        if args.out:
            planned = Path(args.out).with_suffix(".mp4")
            payload = provenance_for_clip(prepared, planned)
            dest = write_clip_provenance(planned, payload)
            sidecar = str(dest)
            print(f"provenance: {sidecar}")
        body = {
            "ok": True,
            "nodes": len(workflow),
            "recipe": prepared.recipe_id,
            "variant": prepared.variant,
            "width": prepared.width,
            "height": prepared.height,
            "frames": prepared.frames,
            "warning": prepared.warning,
            "provenance": prepared.provenance,
        }
        if sidecar:
            body["provenance_sidecar"] = sidecar
        print(json.dumps(body, indent=1))
        return 0
    try:
        client = comfy_client()
        upload_remake_stills(workflow, client.upload_image, scott, blaze)
        rec = execute_prepared(workflow, variant=prepared.variant)
    except FileNotFoundError as exc:
        print(f"FAIL  {exc}")
        return 1
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1
    clip = rec.get("video_path")
    if clip:
        payload = provenance_for_clip(prepared, clip)
        dest = write_clip_provenance(clip, payload)
        rec["provenance_sidecar"] = str(dest)
        rec["provenance"] = payload
        print(f"provenance: {dest}")
    print(json.dumps(rec, indent=1))
    return 0


def cmd_comfy(args: argparse.Namespace) -> int:
    command = getattr(args, "comfy_command", "run")
    if command in {"start", "stop", "status", "restart"}:
        from master_agent.comfy.tower import cmd_tower

        return cmd_tower(args)
    if command == "update":
        from master_agent.comfy.updates import cmd_update

        return cmd_update(args)
    if command == "attach":
        return cmd_comfy_attach(args)
    recipe = getattr(args, "recipe", None)
    if recipe:
        from master_agent.comfy.blaze_remake import is_blaze_remake

        if is_blaze_remake(recipe):
            return _cmd_comfy_blaze_remake(args)
        print(
            "FAIL  comfy run --recipe expects blaze-concert or blaze-pier. "
            "Attach recipes use: comfy attach --recipe FILE"
        )
        return 1
    from master_agent.comfy.cli_run import LintError, execute_prepared, lint_or_raise, prepare_run

    try:
        overrides = _parse_override_flags(args.set)
        workflow = None
        if args.workflow_json:
            p = Path(args.workflow_json)
            raw = p.read_text(encoding="utf-8") if p.is_file() else args.workflow_json
            workflow = json.loads(raw)
        gen_extra: dict = {}
        if getattr(args, "mode", "generate") == "generate":
            plan = _inoutpaint_plan(args)
            if plan:
                gen_extra["inoutpaint"] = {
                    "mode": "outpaint",
                    "pad": plan["pad"],
                    "mask_png": plan["mask_png"],
                }
                gen_extra["width"] = plan["width"]
                gen_extra["height"] = plan["height"]
                print(
                    "outpaint canvas "
                    f"{plan['canvas_w']}x{plan['canvas_h']} "
                    f"stage {plan['width']}x{plan['height']}"
                )
            else:
                if getattr(args, "width", None):
                    gen_extra["width"] = args.width
                if getattr(args, "height", None):
                    gen_extra["height"] = args.height
            if getattr(args, "duration", None) is not None:
                gen_extra["duration_s"] = args.duration
            if getattr(args, "frames", None) is not None:
                gen_extra["frames"] = args.frames
            if getattr(args, "seed", None) is not None:
                gen_extra["seed"] = args.seed
            if getattr(args, "negative_prompt", None):
                gen_extra["negative_prompt"] = args.negative_prompt
            video = getattr(args, "video", None)
            if video and not getattr(args, "prepare", False):
                gen_extra["video_name"] = Path(video).name
            mask = getattr(args, "mask", None)
            if mask and plan is None:
                gen_extra["mask_name"] = Path(mask).name
        prepared = prepare_run(
            args.mode,
            workflow=workflow,
            template_path=args.template,
            variant=args.variant,
            prompt=args.prompt,
            overrides=overrides,
            vae=getattr(args, "vae", None),
            **gen_extra,
        )
        if getattr(args, "mode", "generate") == "generate" and not getattr(args, "prepare", False):
            from master_agent.comfy.inoutpaint import stash_local_mask, stash_local_video

            video = getattr(args, "video", None)
            if video:
                stash_local_video(prepared, video)
            mask = getattr(args, "mask", None)
            if mask and "inoutpaint" not in gen_extra:
                stash_local_mask(prepared, mask)
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(prepared, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    if args.prepare:
        try:
            client = comfy_client()
            object_info, _src = client.load_object_info(prefer_live=True)
            lint_or_raise(prepared, object_info)
        except LintError as e:
            print(f"FAIL  {e}")
            return 1
        except Exception as e:
            print(f"WARN  linter skipped ({e})")
        if not args.out:
            print(json.dumps({"ok": True, "nodes": len(prepared)}, indent=1))
        return 0
    try:
        rec = execute_prepared(prepared, variant=args.variant)
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    print(json.dumps(rec, indent=1))
    return 0

