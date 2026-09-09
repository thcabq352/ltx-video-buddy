"""comfy run modes. Run: .venv/Scripts/python.exe -m pytest tests/test_comfy_cli.py -q"""

from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.comfy.cli_run import apply_overrides, prepare_run
from master_agent.__main__ import cmd_comfy


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


def _comfy_args(workflow_json: str, **kwargs) -> Namespace:
    values = {
        "set": [],
        "workflow_json": workflow_json,
        "mode": "raw",
        "template": None,
        "variant": "base",
        "prompt": "",
        "out": None,
        "prepare": False,
    }
    values.update(kwargs)
    return Namespace(**values)


def test_cmd_comfy_queues_by_default(monkeypatch, tmp_path: Path, capsys):
    path = tmp_path / "w.json"
    path.write_text('{"12": {"class_type": "KSampler", "inputs": {"steps": 4}}}', encoding="utf-8")
    seen: list[dict] = []

    def fake_exec(workflow, **kwargs):
        seen.append(workflow)
        return {
            "status": "done",
            "prompt_id": "abc",
            "video_path": "outputs/x.mp4",
            "outputs": ["outputs/x.mp4"],
            "nodes": 1,
        }

    monkeypatch.setattr("master_agent.comfy.cli_run.execute_prepared", fake_exec)
    rc = cmd_comfy(_comfy_args(str(path)))
    assert rc == 0
    assert len(seen) == 1
    assert seen[0]["12"]["inputs"]["steps"] == 4
    assert "outputs/x.mp4" in capsys.readouterr().out


def test_cmd_comfy_prepare_does_not_queue(monkeypatch, tmp_path: Path, capsys):
    path = tmp_path / "w.json"
    path.write_text('{"12": {"class_type": "KSampler", "inputs": {"steps": 4}}}', encoding="utf-8")

    def boom(*_a, **_k):
        raise AssertionError("execute_prepared must not run in --prepare")

    monkeypatch.setattr("master_agent.comfy.cli_run.execute_prepared", boom)
    monkeypatch.setattr("master_agent.comfy.cli_run.lint_or_raise", lambda *_a, **_k: None)

    class FakeClient:
        def load_object_info(self, prefer_live=True):
            return {}, "cache"

    monkeypatch.setattr("master_agent.__main__.ComfyClient", FakeClient)
    rc = cmd_comfy(_comfy_args(str(path), prepare=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out.lower() or '"ok": True' in out or '"nodes"' in out
