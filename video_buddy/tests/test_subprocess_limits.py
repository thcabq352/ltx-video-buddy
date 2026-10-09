"""Media tools time out and trainers never outlive the caller (U6)."""

from __future__ import annotations

import os
import sys
import time

import pytest

from master_agent import proc
from master_agent.lora import trainer


def test_run_media_times_out_with_failure_code():
    t0 = time.monotonic()
    res = proc.run_media([sys.executable, "-c", "import time; time.sleep(30)"], text=True, timeout=0.5)
    assert time.monotonic() - t0 < 10
    assert res.returncode == 124
    assert "timed out" in res.stderr


def test_run_media_does_not_inherit_stdin():
    res = proc.run_media([sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"], text=True)
    assert res.returncode == 0
    assert res.stdout.strip() == "''"


def test_trainer_timeout_kills_process(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "AI_TOOLKIT_DIR", tmp_path)
    monkeypatch.setenv("LORA_TRAIN_TIMEOUT_S", "0.5")
    t0 = time.monotonic()
    rc = trainer._default_runner(
        [sys.executable, "-c", "import time; print('step', flush=True); time.sleep(30)"],
        tmp_path / "train.log",
    )
    assert time.monotonic() - t0 < 10
    assert rc != 0
    assert "step" in (tmp_path / "train.log").read_text()


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX pids")
def test_trainer_killed_when_caller_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "AI_TOOLKIT_DIR", tmp_path)
    pid_file = tmp_path / "pid"
    script = (
        "import os, time; "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
        "print('started', flush=True); time.sleep(30)"
    )

    def boom(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.print", boom)
    with pytest.raises(KeyboardInterrupt):
        trainer._default_runner([sys.executable, "-c", script], tmp_path / "train.log")
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
