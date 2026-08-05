"""Validate a trained character LoRA — identity-scored held-out renders.

Renders 4 held-out prompts (poses/backgrounds NOT in the dataset) through
the flux workflow with the trained LoRA injected, scores each against the
hero sheet image via the vision judge, and proposes the next training
overrides via ``retry_ladder`` when the score is below threshold.

LoRA injection: the flux workflow template may not ship a lora node, so we
insert a ``LoraLoaderModelOnly`` between the UNETLoader and every consumer
of its MODEL output (or set ``lora_name`` on an existing lora node).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from master_agent.comfy.client import ComfyClient
from master_agent.comfy.workflow_patcher import _ensure_lora_node, load_and_patch_workflow
from master_agent.config import (
    CHARACTERS_DIR,
    COMFYUI_OUTPUT_DIR,
    LORA_DEFAULTS,
    LORA_SCORE_THRESHOLD,
    MODELS_DIR,
)
from master_agent.judge.vision import vision_review

# Held-out scenes: deliberately absent from the sheet/dataset grid.
HELD_OUT_SCENES = [
    "walking through a neon-lit city street at night, full body shot",
    "sitting in a sunlit cafe, laughing, medium shot",
    "standing on a windy beach at dawn, three-quarter view",
    "close-up portrait in a cozy library, soft window light",
]

_LORA_CLASSES = ("LoraLoaderModelOnly", "LoraLoader")


def retry_ladder(attempt: int) -> dict:
    """Overrides for the NEXT training attempt after ``attempt`` failed.

    1 -> +500 steps; 2 -> lr 2e-4; 3 -> rank 32; 4 -> add 1024 resolution;
    >=5 -> {} (give up).
    """
    if attempt <= 1:
        return {"steps": int(LORA_DEFAULTS["steps"]) + 500}
    if attempt == 2:
        return {"lr": 2e-4}
    if attempt == 3:
        return {"rank": 32}
    if attempt == 4:
        return {"resolution": [512, 768, 1024]}
    return {}


def _inject_lora(workflow: dict[str, Any], lora_name: str) -> None:
    """Insert/set a LoraLoaderModelOnly carrying the trained character LoRA."""
    # Existing lora node: just set the name.
    for _nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") in _LORA_CLASSES:
            node.setdefault("inputs", {})["lora_name"] = lora_name
            return
    # Template has no lora node: insert one via the patcher helper.
    if _ensure_lora_node(workflow, lora_name) is None:
        raise RuntimeError("flux workflow has no UNETLoader node to attach the LoRA to")


def _newest_lora(name: str) -> Path | None:
    loras_dir = MODELS_DIR / "loras"
    if not loras_dir.is_dir():
        return None
    candidates = sorted(
        loras_dir.glob(f"{name}_r*.safetensors"), key=lambda p: p.stat().st_mtime
    )
    return candidates[-1] if candidates else None


def _hero_path(char_dir: Path) -> Path | None:
    sheet_dir = char_dir / "sheet"
    manifest = sheet_dir / "sheet.json"
    if manifest.is_file():
        try:
            hero = (json.loads(manifest.read_text(encoding="utf-8")) or {}).get("hero")
            if hero and (sheet_dir / hero).is_file():
                return sheet_dir / hero
        except Exception:
            pass
    if sheet_dir.is_dir():
        pngs = sorted(sheet_dir.glob("*.png"))
        return pngs[0] if pngs else None
    return None


def validate_lora(name: str, attempt: int = 1, client=None) -> dict:
    """Score the trained LoRA on held-out scenes.

    Returns {"score", "pass", "images", "next_overrides"}.
    """
    char_dir = CHARACTERS_DIR / name
    char_json = char_dir / "character.json"
    if not char_json.is_file():
        return {"score": 0.0, "pass": False, "images": [], "next_overrides": {}, "error": "no character.json"}
    character = json.loads(char_json.read_text(encoding="utf-8"))
    lora_path = _newest_lora(name)
    if lora_path is None:
        return {"score": 0.0, "pass": False, "images": [], "next_overrides": retry_ladder(attempt), "error": "no trained lora found"}
    hero = _hero_path(char_dir)
    if client is None:
        client = ComfyClient()

    trigger = character.get("trigger_word") or ""
    appearance = character.get("appearance") or name
    out_dir = char_dir / "validate"
    out_dir.mkdir(parents=True, exist_ok=True)

    images: list[str] = []
    scores: list[float] = []
    for i, scene in enumerate(HELD_OUT_SCENES):
        workflow, _meta = load_and_patch_workflow(
            "flux",
            prompt=f"{trigger}, {appearance}, {scene}".lstrip(", "),
            width=1024,
            height=1024,
            filename_prefix=f"val_{name}_{i:02d}",
        )
        _inject_lora(workflow, lora_path.name)
        prompt_id = client.queue_prompt(workflow)
        entry = client.wait_for_prompt(prompt_id)
        produced = []
        for node_out in (entry.get("outputs") or {}).values():
            for item in node_out.get("images") or []:
                if isinstance(item, dict) and item.get("filename"):
                    produced.append(item)
        if not produced:
            continue
        src = ComfyClient.resolve_output_path(produced[-1], COMFYUI_OUTPUT_DIR)
        dest = out_dir / src.name
        shutil.copy2(src, dest)
        images.append(str(dest))
        review = vision_review(
            dest,
            user_request=f"lora validation for {name}: {scene}",
            reference_paths=[hero] if hero else None,
        )
        if review:
            scores.append(float(review.get("identity_score", review.get("score", 0.0))))

    score = sum(scores) / len(scores) if scores else 0.0
    passed = bool(scores) and score >= LORA_SCORE_THRESHOLD
    return {
        "score": score,
        "pass": passed,
        "images": images,
        "next_overrides": {} if passed else retry_ladder(attempt),
    }
