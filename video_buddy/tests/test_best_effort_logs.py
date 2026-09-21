"""Best-effort paths log the exception at DEBUG instead of swallowing it.

Run: python -m pytest tests/test_best_effort_logs.py -q
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import httpx

from master_agent.comfy.client import ComfyClient
from master_agent.kb.ingest import _workflow_digest, ingest_run_file
from master_agent.orchestrator.pipeline import PipelineResult, _write_record
from master_agent.orchestrator.state import LOOP_PASSED, RunState


def test_unreadable_workflow_digest_logs(tmp_path, caplog):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="master_agent.kb.ingest"):
        digest = _workflow_digest(path)
    assert "unreadable" in digest
    assert any(r.exc_info and r.exc_info[1] is not None for r in caplog.records)


def test_unreadable_run_file_logs_and_returns_zero(tmp_path, caplog):
    path = tmp_path / "run.json"
    path.write_text("{", encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="master_agent.kb.ingest"):
        assert ingest_run_file(path) == 0
    assert any(r.exc_info and r.exc_info[1] is not None for r in caplog.records)


def test_pipeline_kb_ingest_failure_logs(tmp_path, caplog):
    result = PipelineResult("rid", "brief")
    with (
        patch("master_agent.orchestrator.pipeline.RUNS_DIR", tmp_path),
        patch(
            "master_agent.kb.ingest.ingest_run_record",
            side_effect=RuntimeError("chroma down"),
        ),
        caplog.at_level(logging.DEBUG, logger="master_agent.orchestrator.pipeline"),
    ):
        _write_record(result)
    assert any(
        r.exc_info and isinstance(r.exc_info[1], RuntimeError) for r in caplog.records
    )


def test_machine_kb_ingest_failure_logs(tmp_path, caplog):
    from master_agent.orchestrator.machine import Orchestrator

    st = RunState(
        request="brief",
        run_id="abc",
        provenance={"schema": "test"},
        state="DONE",
        loop_status=LOOP_PASSED,
        judge_decision="accept",
    )
    with (
        patch("master_agent.orchestrator.machine.RUNS_DIR", tmp_path),
        patch(
            "master_agent.kb.ingest.ingest_run_record",
            side_effect=RuntimeError("chroma down"),
        ),
        caplog.at_level(logging.DEBUG, logger="master_agent.orchestrator.machine"),
    ):
        Orchestrator()._finish(st)
    assert any(
        r.exc_info and isinstance(r.exc_info[1], RuntimeError) for r in caplog.records
    )


def test_audio_upload_fallback_logs(tmp_path, caplog, monkeypatch):
    audio = tmp_path / "vo.wav"
    audio.write_bytes(b"RIFF")

    class _Resp:
        status_code = 500
        text = "upload endpoint missing"

        def json(self):
            return {}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("master_agent.comfy.client.httpx.Client", lambda *a, **k: _Client())
    with caplog.at_level(logging.DEBUG, logger="master_agent.comfy.client"):
        name = ComfyClient("http://comfy.test").upload_audio(audio)
    assert name == "vo.wav"
    assert any("audio upload failed (500)" in r.message for r in caplog.records)


def test_free_memory_logs_http_error(caplog, monkeypatch):
    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("master_agent.comfy.client.httpx.Client", lambda *a, **k: _Client())
    with caplog.at_level(logging.DEBUG, logger="master_agent.comfy.client"):
        ComfyClient("http://comfy.test").free_memory()
    assert any(
        r.exc_info and isinstance(r.exc_info[1], httpx.ConnectError) for r in caplog.records
    )


def test_music_and_fractal_kb_ingest_logs(tmp_path, caplog):
    from master_agent.fractal.pipeline import _write_record as write_fractal
    from master_agent.music.pipeline import _write_record as write_music

    with (
        patch("master_agent.music.pipeline.RUNS_DIR", tmp_path),
        patch("master_agent.fractal.pipeline.RUNS_DIR", tmp_path),
        patch(
            "master_agent.kb.ingest.ingest_run_record",
            side_effect=RuntimeError("chroma down"),
        ),
    ):
        with caplog.at_level(logging.DEBUG, logger="master_agent.music.pipeline"):
            write_music("m1", {"run_id": "m1"})
        with caplog.at_level(logging.DEBUG, logger="master_agent.fractal.pipeline"):
            write_fractal("f1", {"run_id": "f1"})
    messages = [r.message for r in caplog.records if r.exc_info]
    assert any("music" in m for m in messages)
    assert any("fractal" in m for m in messages)
