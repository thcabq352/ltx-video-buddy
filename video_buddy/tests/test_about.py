"""About card. Run: .venv/Scripts/python.exe -m pytest tests/test_about.py -q"""

from master_agent.about import format_about, studio_about


def test_studio_about_shape():
    card = studio_about()
    assert card["name"] == "VIDEO BUDDY"
    assert card["package"] == "master_agent"
    assert card["mcp_id"] == "master-agent"
    assert card["drive"]["first"] == "cli"
    assert "comfy run" in card["drive"]["graph"]
    assert "zod" in card["personas"]
    text = format_about(card)
    assert "VIDEO BUDDY" in text
    assert "comfy run" in text
    assert "CLI first" in card["tagline"]
