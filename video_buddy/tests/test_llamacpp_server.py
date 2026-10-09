"""Managed llama.cpp lifecycle. No GPU, no real llama-server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from master_agent import config
from master_agent import llamacpp_server as srv


@pytest.fixture(autouse=True)
def _clean():
    srv.reset_state()
    yield
    srv.reset_state()


def test_argv_points_at_models_dir():
    argv = srv.server_argv(
        "/opt/llama-server",
        host="127.0.0.1",
        port=8080,
        models_dir=config.MODELS_DIR,
    )
    assert argv[0] == "/opt/llama-server"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8080"
    assert argv[argv.index("--models-dir") + 1] == str(config.MODELS_DIR)
    assert str(config.MODELS_DIR).endswith("models")


def test_external_server_is_not_spawned():
    with patch.object(srv, "_listening", return_value=True), patch.object(
        srv, "_popen", side_effect=AssertionError("spawned")
    ):
        assert srv.ensure_started() == "external"


def test_autostart_off_never_spawns(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_AUTOSTART", False)
    with patch.object(srv, "_listening", return_value=False), patch.object(
        srv, "resolve_binary", return_value="/opt/llama-server"
    ), patch.object(srv, "_popen", side_effect=AssertionError("spawned")):
        assert srv.ensure_started() == "disabled"
    with patch.object(srv, "_listening", return_value=True):
        assert srv.ensure_started() == "external"


def test_missing_binary_warns_and_does_not_raise(capsys):
    with patch.object(srv, "_listening", return_value=False), patch.object(
        srv, "resolve_binary", return_value=None
    ), patch.object(srv, "_popen", side_effect=AssertionError("spawned")):
        assert srv.ensure_started(fall_through=True) == "missing"
    err = capsys.readouterr().err
    assert "llama.cpp binary not found" in err
    assert "Falling through to Ollama" in err
    assert srv.ensure_started(fall_through=True) == "missing"
    assert capsys.readouterr().err == ""


def test_start_uses_models_dir_and_stop_signals_only_owned():
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None
    argv_box: dict = {}

    def fake_popen(argv):
        argv_box["argv"] = argv
        return proc

    with patch.object(srv, "_listening", return_value=False), patch.object(
        srv, "resolve_binary", return_value="/usr/bin/llama-server"
    ), patch.object(srv, "_popen", side_effect=fake_popen), patch.object(
        srv, "_wait_until_listening", return_value=True
    ), patch.object(srv, "_terminate") as term:
        assert srv.ensure_started() == "started"
        assert "--models-dir" in argv_box["argv"]
        assert argv_box["argv"][argv_box["argv"].index("--models-dir") + 1] == str(
            config.MODELS_DIR
        )
        srv.stop()
        term.assert_called_once_with(proc)
        srv.stop()
        term.assert_called_once()


def test_failed_start_does_not_raise():
    proc = MagicMock()
    proc.pid = 7
    proc.poll.return_value = 1
    with patch.object(srv, "_listening", return_value=False), patch.object(
        srv, "resolve_binary", return_value="/usr/bin/llama-server"
    ), patch.object(srv, "_popen", return_value=proc), patch.object(
        srv, "_wait_until_listening", return_value=False
    ), patch.object(srv, "_terminate") as term:
        assert srv.ensure_started(fall_through=True) == "failed"
    term.assert_called_once_with(proc)


def test_pinned_missing_binary_does_not_say_fall_through(capsys):
    with patch.object(srv, "_listening", return_value=False), patch.object(
        srv, "resolve_binary", return_value=None
    ):
        assert srv.ensure_started(fall_through=False) == "missing"
    err = capsys.readouterr().err
    assert "LLM_PROVIDER=llamacpp stays on llama.cpp" in err
    assert "Falling through" not in err
