"""Character sheet — render the bible's shot grid through the flux workflow.

Every shot is rendered at 1024x1024 with the appearance block prefixed,
copied into the character's sheet dir, and (optionally) identity-checked
against the hero image by the vision judge. Failing shots are retried once
with a new seed. A ``sheet.json`` manifest persists kept files + scores for
``dataset.build_dataset``.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

from master_agent.character.bible import CharacterBible
from master_agent.comfy.client import ComfyClient
from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.config import CHARACTERS_DIR, COMFYUI_OUTPUT_DIR, JUDGE_SCORE_THRESHOLD
from master_agent.judge.vision import vision_review


def _history_images(history_entry: dict[str, Any]) -> list[dict[str, str]]:
    """All image outputs from a history entry."""
    out: list[dict[str, str]] = []
    for node_out in (history_entry.get("outputs") or {}).values():
        for item in node_out.get("images") or []:
            if isinstance(item, dict) and item.get("filename"):
                out.append(
                    {
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder") or "",
                        "type": item.get("type") or "output",
                    }
                )
    return out


def _render_one(
    client,
    bible: CharacterBible,
    shot: str,
    index: int,
    seed: int | None = None,
) -> Path | None:
    """Queue one shot and return the produced PNG path (None on failure)."""
    workflow, _meta = load_and_patch_workflow(
        "flux",
        prompt=f"{bible.appearance}, {shot}",
        seed=seed,
        width=1024,
        height=1024,
        filename_prefix=f"ccc_{bible.name}_{index:02d}",
    )
    prompt_id = client.queue_prompt(workflow)
    entry = client.wait_for_prompt(prompt_id)
    images = _history_images(entry)
    if not images:
        return None
    src = ComfyClient.resolve_output_path(images[-1], COMFYUI_OUTPUT_DIR)
    return src if src.is_file() else None


def generate_character_sheet(
    bible: CharacterBible,
    client=None,
    shots: int | None = None,
    out_dir: Path | None = None,
    vision: bool = True,
) -> dict:
    """Render + filter the character sheet.

    Returns {"sheet_dir", "hero", "kept", "scores"}; also writes sheet.json.
    """
    if client is None:
        client = ComfyClient()
    sheet_dir = Path(out_dir) if out_dir else CHARACTERS_DIR / bible.name / "sheet"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    shot_list = list(bible.shots)[: shots or len(bible.shots)]
    kept: list[str] = []
    scores: dict[str, float] = {}
    hero: Path | None = None

    for i, shot in enumerate(shot_list):
        src = _render_one(client, bible, shot, i)
        if src is None:
            continue
        dest = sheet_dir / f"{i:02d}_{src.name}"
        shutil.copy2(src, dest)

        if hero is None:
            hero = dest  # first rendered image is the identity reference
            kept.append(dest.name)
            scores[dest.name] = 1.0
            continue

        if not vision:
            kept.append(dest.name)
            continue

        review = vision_review(
            dest,
            user_request=f"character sheet for {bible.name}: {shot}",
            reference_paths=[hero],
        )
        if review is None:
            # vision unavailable/failed: cannot judge, keep the shot
            kept.append(dest.name)
            continue
        score = float(review.get("identity_score", review.get("score", 0.0)))
        if score < JUDGE_SCORE_THRESHOLD:
            # retry once with a new seed
            retry_src = _render_one(
                client, bible, shot, i, seed=random.randint(0, 2**32 - 1)
            )
            if retry_src is not None:
                shutil.copy2(retry_src, dest)
                retry = vision_review(
                    dest,
                    user_request=f"character sheet for {bible.name}: {shot} (retry)",
                    reference_paths=[hero],
                )
                if retry is not None:
                    score = float(retry.get("identity_score", retry.get("score", 0.0)))
                else:
                    score = JUDGE_SCORE_THRESHOLD  # unjudgeable retry: keep
        scores[dest.name] = score
        if score >= JUDGE_SCORE_THRESHOLD:
            kept.append(dest.name)
        elif dest.is_file():
            dest.unlink()  # dropped shot does not pollute the dataset

    manifest = {
        **bible.to_dict(),
        "hero": hero.name if hero else None,
        "kept": kept,
        "scores": scores,
    }
    (sheet_dir / "sheet.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "sheet_dir": str(sheet_dir),
        "hero": str(hero) if hero else None,
        "kept": kept,
        "scores": scores,
    }
