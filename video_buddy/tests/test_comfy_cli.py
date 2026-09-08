"""comfy run modes. Run: .venv/Scripts/python.exe -m pytest tests/test_comfy_cli.py -q"""

from pathlib import Path

import pytest

from master_agent.comfy.cli_run import apply_overrides, prepare_run


def test_raw_and_template_overrides(tmp_path: Path):
    wf = {"12": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20}}}
    out = apply_overrides(wf, {"12": {"steps": 8}})
    assert out["12"]["inputs"]["steps"] == 8
    assert wf["12"]["inputs"]["steps"] == 20
    raw = prepare_run("raw", workflow=wf, overrides={"12": {"seed": 99}})
    assert raw["12"]["inputs"]["seed"] == 99
    path = tmp_path / "t.json"
    path.write_text('{"7": {"class_type": "CLIPTextEncode", "inputs": {"text": "a"}}}', encoding="utf-8")
    templ = prepare_run("template", template_path=path, overrides={"7": {"text": "b"}})
    assert templ["7"]["inputs"]["text"] == "b"
    with pytest.raises(KeyError):
        apply_overrides(wf, {"99": {"seed": 0}})
