"""Priority queue. Run: .venv/Scripts/python.exe -m pytest tests/test_priority_queue.py -q"""

from master_agent.control.queue import order_scenes


def test_cheap_high_confidence_first():
    scenes = [
        {"id": "dear", "vram_gb": 14.0, "confidence": 0.9, "flagged": False},
        {"id": "easy", "vram_gb": 9.0, "confidence": 0.95, "flagged": False},
        {"id": "hot", "vram_gb": 8.0, "confidence": 0.99, "flagged": True},
        {"id": "mid", "vram_gb": 9.0, "confidence": 0.5, "flagged": False},
    ]
    ordered = [s["id"] for s in order_scenes(scenes)]
    assert ordered[0] == "easy"
    assert ordered[1] == "mid"
    assert ordered[-1] == "hot"
