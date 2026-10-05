"""Weight download, workflow catalog, and model selector commands."""

from __future__ import annotations

import argparse
import json

def cmd_download_flux(args: argparse.Namespace) -> int:
    from master_agent.models.download import download_flux_weights

    try:
        paths = download_flux_weights()
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    print(f"OK    {len(paths)} flux weight file(s) ready")
    return 0


def cmd_download_models(args: argparse.Namespace) -> int:
    """Scan first; download missing weights only after consent."""
    if getattr(args, "selector", False):
        from master_agent.models.selector import cmd_models_select

        if getattr(args, "heartmula", False):
            args.soundtrack = True
        return cmd_models_select(args)
    from master_agent.setup import print_inventory_preamble

    print_inventory_preamble()
    if getattr(args, "heartmula", False):
        from master_agent.heartmula.doctor import cmd_download_heartmula

        return cmd_download_heartmula(args)
    from master_agent.models.weights import (
        WEIGHT_FILES,
        MissingWeightsError,
        download_missing_bundle,
        download_named_file,
        ic_ingredients_placement,
        is_ltx25_bundle,
        scan_bundle,
    )

    if args.bundle:
        bundle = args.bundle
    elif getattr(args, "wan", False):
        bundle = "wan22"
    elif getattr(args, "vace", False):
        bundle = "vace"
    elif getattr(args, "krea", False):
        bundle = "krea2"
    elif getattr(args, "qwen", False):
        bundle = "qwen_edit"
    elif getattr(args, "flux_pack", False):
        bundle = "flux"
    elif args.h3:
        bundle = "h3_all"
    elif args.ltx25:
        bundle = "ltx25_all"
    else:
        bundle = "ltx25_core"
    status = scan_bundle(bundle)
    print(json.dumps(status.to_dict(), indent=1) if args.json else (
        "OK    all required weights present" if status.ok else status.to_dict()["ask"]
    ))
    if is_ltx25_bundle(bundle):
        place = ic_ingredients_placement()
        if place["ok"]:
            print(f"OK    ic-lora {place['detail']}")
        else:
            print(place["detail"])
            if place.get("fix"):
                print(f"        → {place['fix']}")
            # Pixel-spatial IC-LoRA satisfies the slot scan. msr / v2v still
            # name the Ingredients file, so list it even when the bundle is OK.
            # --yes fetches that exact filename only when no copy exists.
            # A wrong-folder copy is a move, not a download.
            allow_ic = (
                args.yes
                and not getattr(args, "scan_only", False)
                and not getattr(args, "use_existing", False)
            )
            if allow_ic and not place.get("path"):
                try:
                    got = download_named_file(WEIGHT_FILES["ic_lora"], yes=True)
                except Exception as exc:
                    print(f"FAIL  ic-lora: {exc}")
                    return 1
                if got is not None:
                    print(f"OK    downloaded {got.name}")
    if getattr(args, "scan_only", False):
        print("\n--scan-only: nothing downloaded.")
        return 0 if status.ok else 2
    if getattr(args, "use_existing", False):
        print("\n--use-existing: keeping files already on disk. Nothing downloaded.")
        return 0
    if status.ok:
        return 0
    if not args.yes and not getattr(args, "download", False):
        print("\nNothing downloaded. Re-run with --yes after you agree, or pass --download to confirm.")
        return 2
    if not args.yes:
        from master_agent.setup import confirm_prompt

        if not confirm_prompt("Download the confirmed-missing set? [y/N] "):
            print("\nNothing downloaded.")
            return 2
    try:
        _status, paths = download_missing_bundle(
            bundle,
            yes=True,
            include_optional=bool(args.optional),
        )
    except MissingWeightsError as e:
        print(f"FAIL  {e}")
        return 1
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    print(f"OK    {len(paths)} file(s) downloaded")
    return 0


def _print_director_recipes(recipes: list[dict]) -> None:
    if not recipes:
        return
    print(f"{len(recipes)} director recipe(s):")
    for row in recipes:
        delivery = " crop" if row.get("delivery") else ""
        print(
            f"  {row['id']:<28} variant={row['variant']:<6} "
            f"{row['width']}x{row['height']} {row['frames']}f  "
            f"{row.get('description', '')}{delivery}"
        )


def cmd_workflows(args: argparse.Namespace) -> int:
    from master_agent.comfy.catalog import list_catalog_items
    from master_agent.comfy.partner_pointers import POINTERS
    from master_agent.orchestrator.director_presets import list_recipe_rows

    items = list_catalog_items()
    pointers = [p.as_list_item() for p in POINTERS]
    recipes = list_recipe_rows()
    variants = [i for i in items if i.get("kind") == "variant"]
    if args.json:
        print(json.dumps([*items, *pointers, *recipes], indent=1))
        return 0
    if getattr(args, "vram", False):
        from master_agent.models.vram_policy import format_vram_table, workflow_row

        print(format_vram_table())
        print(f"{len(variants)} default catalog variant(s) — 16GB class:")
        for item in variants:
            row = workflow_row(item["id"])
            alt = f" → {row.safer_alternate}" if row.safer_alternate else ""
            print(
                f"  {item['id']:<28} {row.vram_class:<8} ~{row.expected_vram_gb:4.1f}G  "
                f"{row.default_pack}{alt}"
            )
        print(
            f"{len(pointers)} Partner pointer(s) omitted from the VRAM table "
            "(field-shape records, not executable)."
        )
        _print_director_recipes(recipes)
        return 0
    print(f"{len(variants)} default catalog variant(s):")
    for item in variants:
        desc = item.get("description") or ""
        suffix = f"  {desc}" if desc else ""
        print(f"  {item['id']:<28} {item.get('path', '')}{suffix}")
    print(
        f"{len(pointers)} Partner pointer(s) (field-shape record, not executable):"
    )
    for item in pointers:
        print(
            f"  {item['id']:<28} {item.get('template', '')}  {item.get('description', '')}"
        )
    _print_director_recipes(recipes)
    return 0


def cmd_models_manifest(args: argparse.Namespace) -> int:
    from master_agent.models.selector import cmd_models_manifest as _manifest

    return _manifest(args)


def cmd_models_select(args: argparse.Namespace) -> int:
    from master_agent.models.selector import cmd_models_select as _select

    return _select(args)

