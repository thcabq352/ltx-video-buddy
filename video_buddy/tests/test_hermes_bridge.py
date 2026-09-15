"""Hermes multi-profile gateway + pitch. No live SOS/FastAPI.

Run: .venv/Scripts/python.exe -m pytest tests/test_hermes_bridge.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

from master_agent.hermes.gateways import discover_gateways, seat_system_prompt
from master_agent.hermes.pitch import PITCH_SYSTEM, hermes_pitch


def test_discover_profiles_and_ltx_system(tmp_path: Path):
    (tmp_path / "config.yaml").write_text("platforms:\n  api_server:\n    extra:\n      port: 8642\n", encoding="utf-8")
    forge = tmp_path / "profiles" / "forge"
    forge.mkdir(parents=True)
    (forge / "config.yaml").write_text("platforms:\n  api_server:\n    extra:\n      port: 8644\n", encoding="utf-8")
    rows = discover_gateways(home=tmp_path, host="127.0.0.1", probe=False)
    names = {g.profile for g in rows}
    assert "default" in names
    assert "forge" in names
    ltx = next((g for g in rows if g.profile == "ltx"), None)
    if ltx:
        prompt = seat_system_prompt(ltx)
        assert "Ltx" in prompt or "ltx" in prompt.lower()
    forge_gw = next(g for g in rows if g.profile == "forge")
    assert "first person" in seat_system_prompt(forge_gw)
    adapters = [g for g in rows if g.source == "buddy-adapter"]
    assert adapters and adapters[0].port == 8189
    assert "8642" not in adapters[0].chat_url
    ghosts = [g for g in rows if g.source == "sos-mux" and g.profile == "ltx"]
    assert ghosts, "SOS mux ghost ltx rows stay for compat"


def test_probe_unhealthy_ghosts_do_not_win_primary(tmp_path: Path):
    from master_agent.hermes.gateways import discover_primary_seat

    (tmp_path / "config.yaml").write_text("platforms:\n  api_server:\n    extra:\n      port: 8642\n", encoding="utf-8")

    def request(method, url, **_k):
        if ":8189/" in url:
            return {"ok": True, "status": 200}
        return {"ok": False, "status": 503}

    primary = discover_primary_seat(home=tmp_path, host="127.0.0.1", request=request)
    assert primary is not None
    assert primary.source == "buddy-adapter"
    assert primary.port == 8189
    assert primary.healthy is True


def test_healthy_hermes_ltx_wins_over_adapter(tmp_path: Path):
    from master_agent.hermes.gateways import discover_gateways, discover_primary_seat

    (tmp_path / "config.yaml").write_text("platforms:\n  api_server:\n    extra:\n      port: 8642\n", encoding="utf-8")
    ltx = tmp_path / "profiles" / "ltx"
    ltx.mkdir(parents=True)
    (ltx / "config.yaml").write_text("platforms:\n  api_server:\n    extra:\n      port: 8642\n", encoding="utf-8")

    def request(method, url, **_k):
        return {"ok": True, "status": 200}

    rows = discover_gateways(home=tmp_path, host="127.0.0.1", request=request)
    assert not any(g.source == "buddy-adapter" for g in rows)
    primary = discover_primary_seat(home=tmp_path, host="127.0.0.1", request=request)
    assert primary is not None
    assert primary.profile == "ltx"
    assert primary.source == "hermes-gateway"


def test_hermes_pitch_returns_three_shots():
    story_json = json.dumps(
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
        body = {"choices": [{"message": {"content": story_json}}]}
        return {"ok": True, "status": 200, "body": body, "error": ""}

    out = hermes_pitch({"brief": "Ada wants the coat."}, token="t", request=request, make_video=False)
    assert out["ok"] is True
    assert out["title"] == "Brass Latch"
    assert len(out["shots"]) == 3
    assert out["shots"][0]["prompt"] == "wide tent night"
    assert "JSON only" in PITCH_SYSTEM
