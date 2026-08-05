"""Mocked tests for the CCC character stage (no network/GPU/ComfyUI).

Run: .venv/Scripts/python.exe -m pytest tests/test_character.py -q
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from master_agent.character import bible as bible_mod
from master_agent.character import dataset as dataset_mod
from master_agent.character import sheet as sheet_mod
from master_agent.character.bible import CharacterBible, make_character_bible


# --- bible --------------------------------------------------------------------


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=self.replies[min(self.calls - 1, len(self.replies) - 1)])


def _valid_payload(**kw):
    payload = {
        "name": "mira voss",
        "trigger_word": "zxc_mira",
        "appearance": "woman in her 30s, short red hair, green eyes, freckles, denim jacket",
        "shots": [f"shot {i}, simple background" for i in range(12)],
    }
    payload.update(kw)
    return json.dumps(payload)


class TestBible:
    def test_valid_json(self):
        b = make_character_bible("a red-haired woman", llm=FakeLLM([_valid_payload()]))
        assert b.name == "mira_voss"
        assert b.trigger_word == "zxc_mira"
        assert "red hair" in b.appearance
        assert len(b.shots) == 12

    def test_markdown_fenced_json(self):
        fenced = "```json\n" + _valid_payload() + "\n```"
        b = make_character_bible("a red-haired woman", llm=FakeLLM([fenced]))
        assert b.name == "mira_voss"
        assert b.trigger_word == "zxc_mira"

    def test_malformed_retries_then_falls_back(self):
        llm = FakeLLM(["sorry, here is your character:", "still not json"])
        b = make_character_bible("A brave knight with a scar", llm=llm)
        assert llm.calls == 2  # retried once
        assert b.name == "a_brave"
        assert re.fullmatch(r"[a-z0-9_]{3,20}", b.trigger_word)
        assert b.trigger_word.startswith("zxc_")
        assert len(b.shots) >= 12
        assert b.appearance == "A brave knight with a scar"

    def test_trigger_sanitized(self):
        bad = _valid_payload(trigger_word="My Trigger!! 2024")
        b = make_character_bible("x", llm=FakeLLM([bad]))
        assert re.fullmatch(r"[a-z0-9_]{3,20}", b.trigger_word)
        assert b.trigger_word == "my_trigger_2024"

    def test_empty_trigger_gets_zxc_prefix(self):
        bad = _valid_payload(trigger_word="")
        b = make_character_bible("x", llm=FakeLLM([bad]))
        assert b.trigger_word.startswith("zxc_")


# --- sheet --------------------------------------------------------------------


class FakeComfyClient:
    """Canned queue/wait that 'produces' a PNG per prompt in a fake output dir."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.count = 0

    def queue_prompt(self, workflow):
        self.count += 1
        return f"pid-{self.count}"

    def wait_for_prompt(self, prompt_id):
        fname = f"{prompt_id}.png"
        (self.out_dir / fname).write_bytes(b"fakepng")
        return {
            "outputs": {
                "9": {"images": [{"filename": fname, "subfolder": "", "type": "output"}]}
            }
        }


@pytest.fixture
def sheet_env(tmp_path, monkeypatch):
    comfy_out = tmp_path / "comfy_out"
    comfy_out.mkdir()
    sheet_dir = tmp_path / "sheet"
    monkeypatch.setattr(sheet_mod, "COMFYUI_OUTPUT_DIR", comfy_out)
    monkeypatch.setattr(
        sheet_mod, "load_and_patch_workflow", lambda *a, **k: ({"fake": True}, {})
    )
    return comfy_out, sheet_dir


def _test_bible(n=4):
    return CharacterBible(
        name="mira",
        trigger_word="zxc_mira",
        appearance="short red hair, green eyes",
        shots=[f"shot {i}" for i in range(n)],
    )


class TestSheet:
    def test_kept_dropped_retry_and_hero(self, sheet_env, monkeypatch):
        comfy_out, sheet_dir = sheet_env
        # shot1: 0.90 kept; shot2: 0.50 -> retry 0.85 kept; shot3: 0.50 -> 0.40 dropped
        vision_scores = iter([0.90, 0.50, 0.85, 0.50, 0.40])
        seen_refs = []

        def fake_review(path, *, user_request, reference_paths=None, **kw):
            seen_refs.append(reference_paths)
            s = next(vision_scores)
            return {"score": s, "identity_score": s, "pass": s >= 0.75}

        monkeypatch.setattr(sheet_mod, "vision_review", fake_review)
        client = FakeComfyClient(comfy_out)
        result = sheet_mod.generate_character_sheet(
            _test_bible(4), client=client, out_dir=sheet_dir
        )

        kept = result["kept"]
        assert len(kept) == 3  # hero + 2 kept, 1 dropped
        assert result["hero"] == str(Path(result["sheet_dir"]) / kept[0])
        assert Path(result["hero"]).is_file()
        # hero is the FIRST rendered image (no vision call on it)
        assert kept[0].startswith("00_")
        # every vision call referenced the hero
        assert all(refs == [Path(result["hero"])] for refs in seen_refs)
        # 1 (hero) + 1 (shot1) + 2 (shot2 + retry) + 2 (shot3 + retry)
        assert client.count == 6
        # dropped shot file was removed, kept ones exist
        for fname in kept:
            assert (sheet_dir / fname).is_file()
        assert len(list(sheet_dir.glob("*.png"))) == 3
        # scores recorded per judged image
        assert result["scores"][kept[1]] == pytest.approx(0.90)
        assert result["scores"][kept[2]] == pytest.approx(0.85)
        # manifest persisted for build_dataset
        manifest = json.loads((sheet_dir / "sheet.json").read_text(encoding="utf-8"))
        assert manifest["kept"] == kept
        assert manifest["trigger_word"] == "zxc_mira"

    def test_vision_unavailable_keeps_shots(self, sheet_env, monkeypatch):
        comfy_out, sheet_dir = sheet_env
        monkeypatch.setattr(sheet_mod, "vision_review", lambda *a, **k: None)
        client = FakeComfyClient(comfy_out)
        result = sheet_mod.generate_character_sheet(
            _test_bible(3), client=client, out_dir=sheet_dir
        )
        assert len(result["kept"]) == 3
        assert client.count == 3  # no retries when unjudgeable

    def test_vision_disabled_no_review_calls(self, sheet_env, monkeypatch):
        comfy_out, sheet_dir = sheet_env

        def boom(*a, **k):
            raise AssertionError("vision_review must not be called")

        monkeypatch.setattr(sheet_mod, "vision_review", boom)
        client = FakeComfyClient(comfy_out)
        result = sheet_mod.generate_character_sheet(
            _test_bible(3), client=client, out_dir=sheet_dir, vision=False
        )
        assert len(result["kept"]) == 3


# --- dataset ------------------------------------------------------------------


def _make_sheet(char_dir: Path, n: int):
    sheet_dir = char_dir / "sheet"
    sheet_dir.mkdir(parents=True)
    kept = []
    for i in range(n):
        fname = f"{i:02d}_img.png"
        (sheet_dir / fname).write_bytes(b"fakepng")
        kept.append(fname)
    manifest = {
        "name": char_dir.name,
        "trigger_word": "zxc_mira",
        "appearance": "short red hair, green eyes",
        "shots": [f"shot {i}" for i in range(12)],
        "hero": kept[0],
        "kept": kept,
        "scores": {k: 0.9 for k in kept},
    }
    (sheet_dir / "sheet.json").write_text(json.dumps(manifest), encoding="utf-8")
    return kept


class TestDataset:
    def test_writes_pairs_captions_and_character_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dataset_mod, "CHARACTERS_DIR", tmp_path)
        monkeypatch.setattr("master_agent.kb.store.kb_available", lambda: False)
        kept = _make_sheet(tmp_path / "mira", 3)

        summary = dataset_mod.build_dataset("mira", min_images=3)
        assert summary["ok"] is True
        assert summary["dataset_count"] == 3

        ds = tmp_path / "mira" / "dataset"
        for fname in kept:
            assert (ds / fname).is_file()
            caption = (ds / Path(fname).with_suffix(".txt")).read_text(encoding="utf-8")
            assert caption.startswith("[trigger], ")
            assert "short red hair" in caption

        character = json.loads(
            (tmp_path / "mira" / "character.json").read_text(encoding="utf-8")
        )
        assert character["name"] == "mira"
        assert character["trigger_word"] == "zxc_mira"
        assert character["dataset_count"] == 3
        assert character["scores"][kept[0]] == 0.9
        assert character["created"]

    def test_min_images_flag_still_writes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dataset_mod, "CHARACTERS_DIR", tmp_path)
        monkeypatch.setattr("master_agent.kb.store.kb_available", lambda: False)
        _make_sheet(tmp_path / "mira", 2)

        summary = dataset_mod.build_dataset("mira", min_images=10)
        assert summary["ok"] is False
        assert "min_images" in summary["reason"]
        assert summary["dataset_count"] == 2
        assert len(list((tmp_path / "mira" / "dataset").glob("*.png"))) == 2
        assert (tmp_path / "mira" / "character.json").is_file()
