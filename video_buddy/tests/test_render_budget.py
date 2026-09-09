"""Render budget. Run: .venv/Scripts/python.exe -m pytest tests/test_render_budget.py -q"""

from master_agent.control.budget import RenderBudget, vram_minutes
from master_agent.control.cost import estimate_cost


def test_exceeding_cap_pauses_queue_and_logs_scene(tmp_path):
    cheap = estimate_cost("base", frames=81)
    add = vram_minutes(cheap)
    assert add > 0
    budget = RenderBudget(tmp_path / "budget.json", cap=add * 1.5, used=0.0)
    first = budget.consider("shot-a", cheap)
    assert first["decision"] == "admit"
    assert first["scene_id"] == "shot-a"
    assert first["ts"]
    assert budget.used == first["used_after"]
    assert budget.paused is False

    plan = budget.apply_queue(
        [
            {"id": "shot-b", "cost": cheap},
            {"id": "shot-c", "cost": cheap},
        ]
    )
    assert plan["paused"] is True
    assert [s["id"] for s in plan["admitted"]] == []
    assert [s["id"] for s in plan["held"]] == ["shot-b", "shot-c"]
    assert budget.used < budget.cap
    assert {p["id"] for p in budget.pending} == {"shot-b", "shot-c"}
    logged = [row for row in budget.log if row["scene_id"] in {"shot-a", "shot-b", "shot-c"}]
    assert len(logged) == 3
    assert logged[1]["decision"] == "hold"
    assert logged[1]["ts"] and logged[1]["scene_id"] == "shot-b"
