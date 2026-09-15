"""Hermes profile register + facade. No live SOS/GPU.

Run: python -m pytest tests/test_hermes_adapter.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

from master_agent.a2a.protocol import TaskStore
from master_agent.hermes.adapter import hermes_complete, openai_models, wants_pitch
from master_agent.hermes.gateways import LTX_RESEARCH_SYSTEM
from master_agent.hermes.pitch import hermes_pitch
from master_agent.hermes.profile import register_ltx_profile


def test_register_ltx_profile_no_env_and_soul_merge(tmp_path: Path):
    root = tmp_path / "buddy"
    root.mkdir()
    (root / "master_agent").mkdir()
    py = tmp_path / "python"
    py.write_text("", encoding="utf-8")
    home = tmp_path / "hermes"
    first = register_ltx_profile(home=home, video_buddy_root=root, python=str(py), force=False)
    soul = first.profile_dir / "SOUL.md"
    assert soul.read_text(encoding="utf-8").strip() == LTX_RESEARCH_SYSTEM.strip()
    assert not (first.profile_dir / ".env").exists()
    assert first.soul_written is True
    cfg = (first.profile_dir / "config.yaml").read_text(encoding="utf-8")
    assert "master-agent" in cfg
    assert str(root) in cfg

    soul.write_text("Custom Ltx soul. Do not clobber.\n", encoding="utf-8")
    second = register_ltx_profile(home=home, video_buddy_root=root, python=str(py), force=False)
    assert second.soul_skipped is True
    assert "Custom Ltx soul" in soul.read_text(encoding="utf-8")
    assert not (second.profile_dir / ".env").exists()

    third = register_ltx_profile(home=home, video_buddy_root=root, python=str(py), force=True)
    assert third.soul_written is True
    assert soul.read_text(encoding="utf-8").strip() == LTX_RESEARCH_SYSTEM.strip()


def test_wants_pitch_only_on_explicit_ask():
    assert wants_pitch("please pitch three shots") is True
    assert wants_pitch("storyboard only, no render") is True
    assert wants_pitch("rain on a window") is False
    assert wants_pitch("rain on a window", {"pitch": True}) is True


def test_facade_raw_brief_submits_and_pitch_does_not():
    store = TaskStore()
    submitted = []

    def submit(task_id, body):
        submitted.append((task_id, body["request"]))

    raw = hermes_complete(
        {"messages": [{"role": "user", "content": "rain on a window"}]},
        store=store,
        submit=submit,
    )
    assert raw["buddy"]["task_id"]
    assert submitted and submitted[0][1] == "rain on a window"

    pitch = hermes_complete(
        {"messages": [{"role": "user", "content": "pitch a 3-shot board"}], "metadata": {"pitch": True}},
        store=store,
        submit=submit,
        request=lambda *a, **k: {
            "ok": True,
            "status": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "title": "Latch",
                                    "logline": "A hand stops.",
                                    "shots": [
                                        {"title": "Open", "prompt": "wide tent", "motion": "push"},
                                        {"title": "Turn", "prompt": "hook", "motion": "drift"},
                                        {"title": "Close", "prompt": "hand", "motion": "hold"},
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        },
    )
    assert "pitch" in pitch
    assert pitch["pitch"]["title"] == "Latch"
    assert len(submitted) == 1
    models = openai_models()
    assert models["data"][0]["id"] == "ltx"


def test_hermes_pitch_make_video_submits_via_shared_hook():
    store = TaskStore()
    submitted = []

    def submit(task_id, body):
        submitted.append((task_id, body["request"]))
        store.set(task_id, state="working")

    story = json.dumps(
        {
            "title": "Brass Latch",
            "logline": "A hand almost takes it.",
            "shots": [
                {"title": "Open", "prompt": "wide tent night", "motion": "push"},
                {"title": "Turn", "prompt": "MCU brass hook", "motion": "drift"},
                {"title": "Close", "prompt": "hand stops", "motion": "hold"},
            ],
        }
    )

    def request(method, url, headers=None, json=None):
        return {"ok": True, "status": 200, "body": {"choices": [{"message": {"content": story}}]}, "error": ""}

    out = hermes_pitch(
        {"brief": "Ada wants the coat."},
        token="t",
        request=request,
        make_video=True,
        store=store,
        submit=submit,
    )
    assert out["ok"] is True
    assert out["video"]["task_id"]
    assert submitted
    assert "wide tent night" in submitted[0][1]
