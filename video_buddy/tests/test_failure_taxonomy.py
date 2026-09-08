"""Failure taxonomy. Run: .venv/Scripts/python.exe -m pytest tests/test_failure_taxonomy.py -q"""

from master_agent.control.failures import classify_failure


def test_classifies_five_tags():
    assert classify_failure("CUDA OOM allocation on device")["tag"] == "oom"
    assert classify_failure("ReadTimeout: timed out")["tag"] == "timeout"
    assert classify_failure("empty prompt / unrenderable type")["tag"] == "bad_prompt"
    assert classify_failure("class_type MissingNode")["tag"] == "model_bug"
    assert classify_failure("judge score below threshold, rewrite")["tag"] == "judge_rejection"
