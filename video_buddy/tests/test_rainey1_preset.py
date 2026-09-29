"""Rainey1 director preset and recipe pack. No GPU, no weight downloads.

Run: python -m pytest tests/test_rainey1_preset.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace

from master_agent.__main__ import cmd_run, cmd_workflows
from master_agent.config import DOWNSCALE_LADDER, OBJECT_INFO_CACHE, snap_ltx_frames
from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.orchestrator.director import choose_variant, rule_based_variant
from master_agent.orchestrator.director_presets import (
    load_rainey1_pack,
    recipe_by_id,
    recipe_frames,
    select_recipe,
)


_LADDER = set(DOWNSCALE_LADDER)
_RECIPE_IDS = (
    "rainey1_lock_open",
    "rainey1_breach",
    "rainey1_density",
    "rainey1_myth_16x9",
    "rainey1_story_9x16",
)


def test_pack_loads_and_frames_are_legal():
    pack = load_rainey1_pack()
    assert pack.id == "rainey1"
    assert pack.variant == "base"
    assert "identity drift" in pack.negative
    assert "glow soup" in pack.negative
    assert "text burned in" in pack.negative
    for phrase in (
        "emissive multi-light",
        "identity lock",
        "zero slop",
        "music-video hold",
        "no text overlay",
    ):
        assert phrase in pack.additives
    ids = [recipe.id for recipe in pack.recipes]
    assert ids == list(_RECIPE_IDS)
    for recipe in pack.recipes:
        assert recipe.frames == snap_ltx_frames(recipe.frames)
        assert recipe.frames >= 9
        assert (recipe.frames - 1) % 8 == 0
        assert recipe.variant == "base"
        assert "jason" not in recipe.prompt_pattern.lower()
        assert "jason" not in recipe.anchor.lower()


def test_length_eight_snaps_to_nine():
    assert recipe_frames(8) == 9
    assert recipe_frames(1) == 9
    assert recipe_frames(17) == 17
    assert recipe_frames(25) == 25


def test_ladder_and_story_crop():
    for recipe_id in ("rainey1_lock_open", "rainey1_breach", "rainey1_density", "rainey1_story_9x16"):
        recipe = recipe_by_id(recipe_id)
        assert (recipe.width, recipe.height, recipe.frames) in _LADDER
    myth = recipe_by_id("rainey1_myth_16x9")
    assert (myth.width, myth.height, myth.frames) == (768, 512, 25)
    assert "DOWNSCALE_LADDER" in myth.notes
    story = recipe_by_id("rainey1_story_9x16")
    assert story.vertical_latent is False
    assert "crop" in story.delivery.lower()
    assert "512x768" in story.delivery or "512×768" in story.delivery


def test_cues_select_recipes_and_pin_base():
    assert select_recipe("a garden") is None
    assert select_recipe("rainey1").id == "rainey1_lock_open"
    assert select_recipe("rainey1 lock open: photoreal desk").id == "rainey1_lock_open"
    assert select_recipe("rainey1 breach at night").id == "rainey1_breach"
    assert select_recipe("rainey1 density umbilicus look up").id == "rainey1_density"
    assert select_recipe("rainey1 myth 16:9 set-piece").id == "rainey1_myth_16x9"
    assert select_recipe("rainey1 story 9:16 reels").id == "rainey1_story_9x16"
    # 9:16 must not be read as 16:9.
    assert select_recipe("rainey1 9:16 vertical").id == "rainey1_story_9x16"
    assert rule_based_variant("rainey1 lock open: photoreal locked character") == "base"
    assert rule_based_variant("wan 2.2 photoreal film grain") == "wan22"
    assert choose_variant("rainey1 lock open: photoreal desk") == ("base", "preset")
    assert choose_variant("rainey1", force="eros") == ("eros", "forced")
    assert choose_variant("rainey1 dub this clip", has_video=True) == ("lipsync", "input")


def test_llm_cannot_steal_rainey1_to_wan(monkeypatch):
    monkeypatch.setattr(
        "master_agent.orchestrator.director._llm_variant",
        lambda *_a, **_k: "wan22",
    )
    assert choose_variant("rainey1 breach photoreal film grain") == ("base", "preset")


def test_prompt_and_negative_land_in_the_graph():
    pack = load_rainey1_pack()
    recipe = recipe_by_id("rainey1_lock_open")
    from master_agent.orchestrator.director_presets import compose_positive

    prompt = compose_positive(
        "rainey1 lock open: photoreal locked character at a cluttered desk",
        recipe,
        pack.additives,
    )
    for phrase in pack.additives:
        assert phrase in prompt
    wf, meta = load_and_patch_workflow(
        "base",
        prompt=prompt,
        negative_prompt=pack.negative,
        width=recipe.width,
        height=recipe.height,
        frames=recipe.frames,
        seed=1,
    )
    assert meta["frames"] == 9
    assert wf["20"]["inputs"]["length"] == 9
    assert wf["20"]["inputs"]["width"] == 768
    assert wf["20"]["inputs"]["height"] == 512
    assert "identity drift" in wf["11"]["inputs"]["text"]
    assert "emissive multi-light" in wf["10"]["inputs"]["text"]
    assert "jason" not in wf["10"]["inputs"]["text"].lower()


def test_workflows_lists_rainey1_recipes(capsys):
    assert cmd_workflows(Namespace(json=False, vram=False)) == 0
    text = capsys.readouterr().out
    assert "director recipe" in text
    for recipe_id in _RECIPE_IDS:
        assert recipe_id in text
    assert cmd_workflows(Namespace(json=True, vram=False)) == 0
    items = json.loads(capsys.readouterr().out)
    recipes = [item for item in items if item.get("kind") == "recipe"]
    assert {item["id"] for item in recipes} == set(_RECIPE_IDS)
    assert all(item["variant"] == "base" for item in recipes)


def _run_ns(**overrides) -> Namespace:
    base = dict(
        request="rainey1 lock open: photoreal locked character at desk",
        variant=None,
        duration=5.0,
        duration_set=False,
        quality=None,
        seed=1,
        width=768,
        height=512,
        width_set=False,
        height_set=False,
        video=None,
        image=None,
        audio=None,
        mask=None,
        line=None,
        words=None,
        no_judge=True,
        max_judge_rounds=1,
        max_full_judge_rounds=1,
        storyboard="off",
        llm_panel=None,
        panel_judge=None,
        dry_run=True,
        self_improve_dry=False,
        power_mode=False,
        no_power_mode=True,
        upscale=None,
        no_interview=True,
        attach=None,
        tripod=False,
        lipdub_max_s=None,
        silence_min_s=None,
        lipdub_overlap=None,
        silence_mode=None,
        anchor=None,
        reframe=None,
        max_piece_seconds=None,
        pause_reset_strength=None,
        pause_reset_min_s=None,
        outpaint=False,
        aspect=None,
        negative_prompt=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_dry_run_uses_recipe_and_does_not_queue(monkeypatch, capsys):
    queued: list[bool] = []

    def _queue(self, workflow):
        queued.append(True)
        raise AssertionError("dry-run queued a prompt")

    def _object_info(self, cache_path=OBJECT_INFO_CACHE, prefer_live=True):
        return json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8")), "cache"

    monkeypatch.setattr("master_agent.comfy.client.ComfyClient.queue_prompt", _queue)
    monkeypatch.setattr("master_agent.comfy.client.ComfyClient.load_object_info", _object_info)

    rc = cmd_run(_run_ns())
    out = capsys.readouterr().out
    assert queued == []
    assert "preset: rainey1_lock_open" in out
    assert "variant=base" in out
    assert "frames=9" in out
    assert "variant: base" in out
    assert "nothing queued" in out
    assert rc == 0
