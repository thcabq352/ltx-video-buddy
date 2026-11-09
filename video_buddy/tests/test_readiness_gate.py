"""Readiness must judge the Klein still image and must not disrupt servers.

No GPU, no weights, and no live Comfy or llama-server. nvidia-smi, process
signals, and HTTP are injected. These tests describe the repaired contract.
"""

from __future__ import annotations

import base64
import hashlib
import json

import httpx
import pytest

from master_agent.comfy.client import ComfyClient
from master_agent.gpu_seat import (
    SeatRefused,
    ServerView,
    acquire_for_submit,
    release_after_render,
    reset_state,
)
from master_agent.llamacpp_server import os as llamacpp_os

IMAGE = b"\x89PNG\r\n\x1a\nklein-still-exact"
IMAGE_B64 = base64.b64encode(IMAGE).decode("ascii")
GOOD_VERDICT = json.dumps(
    {
        "score": 0.91,
        "pass": True,
        "issues": [],
        "temporal_consistency": "single frame",
        "subject_lock": "the ceramic cup stays one object",
        "artifacts": "none visible",
        "reason": "soft window light on a ceramic cup sitting on a wood table",
    }
)
REJECTED = json.dumps(
    {
        "score": 0.2,
        "pass": False,
        "reason": "the cup is missing",
        "artifacts": "smeared table",
        "subject_lock": "no cup",
    }
)
READY_JSON = json.dumps({"score": 0.99, "pass": True, "reason": "ready"})
LOW_SCORE = json.dumps(
    {
        "score": 0.5,
        "pass": True,
        "reason": "a cup is partly visible but the table is unfinished",
        "artifacts": "soft smear",
        "subject_lock": "cup",
    }
)
_CUP = {
    "reason": "soft window light on a ceramic cup sitting on a wood table",
    "artifacts": "none visible",
    "subject_lock": "the ceramic cup stays one object",
}
NAN_SCORE = json.dumps({"score": "NaN", "pass": True, **_CUP})
READY_ISSUES = json.dumps({"score": 0.99, "pass": True, "issues": ["ready"]})
READY_COMMA = json.dumps({"score": 0.99, "pass": True, "issues": ["ready,"]})
READY_QUESTION = json.dumps({"score": 0.99, "pass": True, "reason": "ready?"})
MISSING_PASS = json.dumps({"score": 0.91, **_CUP})
PASS_AS_STRING = json.dumps({"score": 0.91, "pass": "true", **_CUP})


class _LLM:
    def __init__(self, sink: list):
        self.sink = sink

    def invoke(self, text):
        self.sink.append(text)
        return "ready"


@pytest.fixture
def seat(monkeypatch):
    reset_state()
    monkeypatch.setenv("SEAT_MIN_FREE_GB", "6")
    monkeypatch.delenv("SEAT_REQUIRE_PROBE", raising=False)
    stopped: list[str] = []
    monkeypatch.setattr(
        "master_agent.llamacpp_server.stop", lambda *a, **k: stopped.append("stop")
    )
    monkeypatch.setattr(
        "master_agent.llamacpp_server.stop_pid", lambda *a, **k: stopped.append("stop_pid")
    )
    monkeypatch.setattr(
        "master_agent.llamacpp_server.start_with_layers",
        lambda *a, **k: stopped.append("start"),
    )
    monkeypatch.setattr(llamacpp_os, "kill", lambda *a, **k: stopped.append("kill"))
    yield stopped
    reset_state()


def _gpu(owned: bool = True, layers: int | None = 32) -> ServerView:
    return ServerView(
        running=True,
        gpu_layers=layers,
        owned=owned,
        pid=4242,
        argv=["llama-server", "-ngl", "32"],
    )


@pytest.mark.parametrize("owned", [True, False])
@pytest.mark.parametrize("layers", [32, None])
@pytest.mark.parametrize("free", [1.5, 12.0, None])
def test_acquire_and_release_never_stop_or_restart(seat, owned, layers, free):
    stopped = seat
    hooks: list = []
    view = _gpu(owned=owned, layers=layers)
    if free is not None and free < 6:
        with pytest.raises(SeatRefused, match="No server was stopped"):
            acquire_for_submit(
                purpose="ComfyUI render",
                view=view,
                free_gb=free,
                stop=lambda current: hooks.append(("stop", current.pid)),
            )
    elif free is None:
        with pytest.raises(SeatRefused, match="No server was stopped"):
            acquire_for_submit(
                purpose="ComfyUI render",
                view=view,
                free_gb=None,
                stop=lambda current: hooks.append("stop"),
            )
    else:
        line = acquire_for_submit(
            purpose="ComfyUI render",
            view=view,
            free_gb=free,
            stop=lambda current: hooks.append("stop"),
        )
        assert "stopped llama-server" not in line
        note = release_after_render(start=lambda ngl: hooks.append(("start", ngl)))
        assert note and "no server was restarted" in note
    if free is None or (free is not None and free < 6):
        release_after_render(start=lambda ngl: hooks.append(("start", ngl)))
    assert stopped == []
    assert hooks == []


def test_probe_error_refuses_without_stopping(seat):
    def boom():
        raise RuntimeError("smi failed")

    with pytest.raises(SeatRefused, match="smi failed"):
        acquire_for_submit(view=_gpu(), free_probe=boom, stop=lambda _v: seat.append("injected"))
    release_after_render(start=lambda ngl: seat.append("start"))
    assert seat == []


def test_cpu_server_and_idle_card_are_left_alone(seat):
    cpu = ServerView(running=True, gpu_layers=0, owned=True, pid=7)
    with pytest.raises(SeatRefused, match="No server was stopped"):
        acquire_for_submit(view=cpu, free_gb=1.5, stop=lambda _v: seat.append("stop"))
    line = acquire_for_submit(view=ServerView(running=False), free_gb=None)
    assert "not running" in line
    release_after_render(start=lambda ngl: seat.append("start"))
    assert seat == []


def test_ordinary_submit_occupied_gpu_does_not_disrupt(monkeypatch, seat):
    monkeypatch.setenv("FIRST_GATE", "passed")
    posts: list[str] = []
    monkeypatch.setattr("master_agent.gpu_seat._default_view", lambda: _gpu(owned=False))
    monkeypatch.setattr("master_agent.gpu_seat.probe_free_vram_gb", lambda: 1.5)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            posts.append("prompt")
            return httpx.Response(200, json={"prompt_id": "p1"})
        return httpx.Response(200, json={})

    _install_http(monkeypatch, handler)
    try:
        with pytest.raises(Exception, match="No server was stopped"):
            ComfyClient("http://comfy.test").queue_prompt({"1": {"class_type": "KSampler", "inputs": {}}})
    finally:
        ComfyClient.close_pool()
    assert posts == []
    assert seat == []


def test_ordinary_submit_probe_error_does_not_disrupt(monkeypatch, seat):
    monkeypatch.setenv("FIRST_GATE", "passed")
    monkeypatch.setattr("master_agent.gpu_seat._default_view", lambda: _gpu())

    def boom():
        raise RuntimeError("smi failed")

    monkeypatch.setattr("master_agent.gpu_seat.probe_free_vram_gb", boom)
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request.url.path)
        return httpx.Response(200, json={"prompt_id": "p1"})

    _install_http(monkeypatch, handler)
    try:
        with pytest.raises(Exception, match="smi failed"):
            ComfyClient("http://comfy.test").queue_prompt({"1": {"class_type": "KSampler", "inputs": {}}})
    finally:
        ComfyClient.close_pool()
    assert posts == []
    assert seat == []


def test_ordinary_submit_with_room_does_not_restart(monkeypatch, seat):
    monkeypatch.setenv("FIRST_GATE", "passed")
    monkeypatch.setattr("master_agent.gpu_seat._default_view", lambda: _gpu())
    monkeypatch.setattr("master_agent.gpu_seat.probe_free_vram_gb", lambda: 12.0)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json={"p1": {"outputs": {"1": {"images": []}}}})
        return httpx.Response(200, json={})

    _install_http(monkeypatch, handler)
    try:
        client = ComfyClient("http://comfy.test")
        prompt_id = client.queue_prompt({"1": {"class_type": "KSampler", "inputs": {}}})
        client.wait_for_prompt(prompt_id, poll_interval=0.01, timeout=2)
    finally:
        ComfyClient.close_pool()
    assert seen[0] == "/prompt"
    assert seat == []


def test_text_fired_flag_does_not_pass_or_queue_the_clip(tmp_path, monkeypatch):
    from master_agent import first_gate

    monkeypatch.delenv("FIRST_GATE", raising=False)
    monkeypatch.setattr(first_gate, "state_path", lambda: tmp_path / "first_gate.json")
    queued: list[str] = []
    with pytest.raises(first_gate.FirstGateBlocked):
        first_gate.run_first_gate(
            still_runner=lambda: queued.append("still") or {"judge_fired": True},
            clip_runner=lambda: queued.append("clip") or {"ok": True},
            judge_probe=lambda: {"fired": True, "ready": True},
            vram_free_gb=12,
        )
    assert queued == ["still"]
    assert first_gate.is_passed() is False
    assert _passed_flag(tmp_path) is not True


def test_ready_probe_without_image_does_not_start_the_still(tmp_path, monkeypatch):
    from master_agent import first_gate

    monkeypatch.delenv("FIRST_GATE", raising=False)
    monkeypatch.setattr(first_gate, "state_path", lambda: tmp_path / "first_gate.json")
    queued: list[str] = []
    with pytest.raises(first_gate.FirstGateBlocked, match="no mmproj"):
        first_gate.run_first_gate(
            still_runner=lambda: queued.append("still") or {"judge_fired": True},
            clip_runner=lambda: queued.append("clip") or {"ok": True},
            judge_probe=lambda: {"fired": True, "ready": False, "reason": "no mmproj"},
            vram_free_gb=12,
        )
    assert queued == []
    assert first_gate.is_passed() is False


def test_visual_acceptance_then_clip_persists(tmp_path, monkeypatch):
    from master_agent import first_gate

    monkeypatch.delenv("FIRST_GATE", raising=False)
    path = tmp_path / "first_gate.json"
    monkeypatch.setattr(first_gate, "state_path", lambda: path)
    result = first_gate.run_first_gate(
        still_runner=lambda: {
            "visual": {
                "accepted": True,
                "saw_image": True,
                "score": 0.91,
                "image_sha256": "abc",
                "reason": "ceramic cup on a wood table",
            }
        },
        clip_runner=lambda: {"ok": True},
        judge_probe=lambda: {"ready": True, "fired": False},
        vram_free_gb=12,
    )
    assert result["status"] == "passed"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["judge"]["saw_image"] is True
    assert saved["judge"]["image_sha256"] == "abc"


def test_clip_failure_leaves_gate_closed(tmp_path, monkeypatch):
    from master_agent import first_gate

    monkeypatch.delenv("FIRST_GATE", raising=False)
    monkeypatch.setattr(first_gate, "state_path", lambda: tmp_path / "first_gate.json")
    with pytest.raises(first_gate.FirstGateBlocked, match="LTX"):
        first_gate.run_first_gate(
            still_runner=lambda: {
                "visual": {"accepted": True, "saw_image": True, "score": 0.9, "reason": "cup"}
            },
            clip_runner=lambda: {"ok": False, "error": "smoke failed"},
            judge_probe=lambda: {"ready": True, "fired": False},
            vram_free_gb=12,
        )
    assert first_gate.is_passed() is False
    assert _passed_flag(tmp_path) is not True
    with pytest.raises(first_gate.FirstGateBlocked):
        first_gate.assert_submit_allowed({"1": {"class_type": "KSampler", "inputs": {}}})


@pytest.mark.parametrize(
    "reply",
    [
        "ready",
        "Ready.",
        "",
        "not-json",
        READY_JSON,
        REJECTED,
        LOW_SCORE,
        NAN_SCORE,
        READY_ISSUES,
        READY_COMMA,
        READY_QUESTION,
        MISSING_PASS,
        PASS_AS_STRING,
    ],
)
def test_non_visual_judge_never_passes_or_queues_ltx(tmp_path, monkeypatch, seat, reply):
    outcome = _run_command(tmp_path, monkeypatch, seat, vision=reply)
    assert outcome["code"] == 2
    assert outcome["passed"] is False
    assert outcome["clips"] == 0
    assert outcome["stills"] == 1
    assert outcome["llm"] == []
    assert outcome["images"] == [("image/png", IMAGE_B64)]
    assert "single word ready" not in outcome["user_text"]
    assert outcome["backend"] == "llamacpp"
    assert seat == []
    assert _passed_flag(tmp_path) is not True
    _assert_later_submit_blocked(monkeypatch)


def test_unavailable_judge_does_not_pass_or_queue_ltx(tmp_path, monkeypatch, seat):
    def explode(*_a, **_k):
        raise ConnectionError("connection refused")

    outcome = _run_command(tmp_path, monkeypatch, seat, vision=explode)
    assert outcome["code"] == 2
    assert outcome["passed"] is False
    assert outcome["stills"] == 1
    assert outcome["clips"] == 0
    assert outcome["images"] == [("image/png", IMAGE_B64)]
    assert seat == []


def test_missing_mmproj_does_not_queue(tmp_path, monkeypatch, seat):
    outcome = _run_command(tmp_path, monkeypatch, seat, vision=GOOD_VERDICT, mmproj=False)
    assert outcome["code"] == 2
    assert outcome["passed"] is False
    assert outcome["stills"] == 0
    assert outcome["clips"] == 0
    assert outcome["images"] is None
    assert outcome["llm"] == []
    assert seat == []


def test_unreachable_llamacpp_does_not_queue_or_autostart(tmp_path, monkeypatch, seat):
    outcome = _run_command(tmp_path, monkeypatch, seat, vision=GOOD_VERDICT, llamacpp=False)
    assert outcome["code"] == 2
    assert outcome["passed"] is False
    assert outcome["stills"] == 0
    assert outcome["llm"] == []
    assert outcome["started"] == []
    assert seat == []


def test_exact_still_bytes_reach_llamacpp_before_ltx_and_pass(tmp_path, monkeypatch, seat):
    outcome = _run_command(tmp_path, monkeypatch, seat, vision=GOOD_VERDICT)
    assert outcome["code"] == 0
    assert outcome["passed"] is True
    assert outcome["stills"] == 1
    assert outcome["clips"] == 1
    assert outcome["images"] == [("image/png", IMAGE_B64)]
    assert outcome["backend"] == "llamacpp"
    assert outcome["view_name"] == "gate.png"
    decoded = base64.b64decode(outcome["images"][0][1])
    assert decoded == IMAGE
    assert hashlib.sha256(IMAGE).hexdigest() in outcome["user_text"]
    assert "ceramic cup" in outcome["user_text"]
    assert "single word ready" not in outcome["user_text"]
    assert outcome["llm"] == []
    assert outcome["prompts"][0][0] == "flux"
    assert "ceramic cup" in outcome["prompts"][0][1]["prompt"]
    assert outcome["prompts"][1][0] == "ltx25_t2v_i2v"
    saved = json.loads((tmp_path / "first_gate.json").read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["klein_still"] is True
    assert saved["ltx_clip"] is True
    assert saved["judge"]["saw_image"] is True
    assert saved["judge"]["image_sha256"] == hashlib.sha256(IMAGE).hexdigest()
    assert saved["judge"]["score"] == pytest.approx(0.91)
    assert seat == []


def test_history_without_an_image_does_not_pass(tmp_path, monkeypatch, seat):
    outcome = _run_command(tmp_path, monkeypatch, seat, vision=GOOD_VERDICT, history_image=False)
    assert outcome["code"] == 2
    assert outcome["passed"] is False
    assert outcome["stills"] == 1
    assert outcome["clips"] == 0
    assert outcome["images"] is None
    assert seat == []


def test_low_vram_readiness_queues_nothing(tmp_path, monkeypatch, seat):
    outcome = _run_command(tmp_path, monkeypatch, seat, vision=GOOD_VERDICT, free=1.0)
    assert outcome["code"] == 2
    assert outcome["passed"] is False
    assert outcome["stills"] == 0
    assert outcome["clips"] == 0
    assert seat == []


def test_unknown_vram_with_gpu_llm_refuses_readiness(tmp_path, monkeypatch, seat):
    outcome = _run_command(tmp_path, monkeypatch, seat, vision=GOOD_VERDICT, free=None)
    assert outcome["code"] == 2
    assert outcome["passed"] is False
    assert outcome["stills"] == 0
    assert outcome["clips"] == 0
    assert "No server was stopped" in outcome["output"]
    assert seat == []


def _passed_flag(tmp_path):
    path = tmp_path / "first_gate.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("passed")


def _install_http(monkeypatch, handler):
    real = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr("master_agent.comfy.client.httpx.Client", factory)
    ComfyClient.close_pool()


def _run_command(tmp_path, monkeypatch, seat, *, vision, mmproj=True, llamacpp=True, free=12.0, history_image=True):
    del seat
    from master_agent import first_gate

    monkeypatch.delenv("FIRST_GATE", raising=False)
    monkeypatch.setenv("SEAT_MIN_FREE_GB", "6")
    monkeypatch.setattr(first_gate, "state_path", lambda: tmp_path / "first_gate.json")
    monkeypatch.setattr("master_agent.gpu_seat._default_view", lambda: _gpu())
    monkeypatch.setattr(
        "master_agent.gpu_seat.probe_free_vram_gb",
        lambda: free,
    )
    llm: list = []
    monkeypatch.setattr("master_agent.llm.get_llm", lambda *a, **k: _LLM(llm))
    started: list = []
    monkeypatch.setattr(
        "master_agent.llamacpp_server.ensure_started",
        lambda *a, **k: started.append("ensure") or "external",
    )
    monkeypatch.setattr(
        "master_agent.llm.provider_available",
        lambda spec: llamacpp and str(spec).startswith("llamacpp"),
    )
    monkeypatch.setattr(
        "master_agent.models.llm_gguf.mmproj_present",
        lambda slot, root=None: (tmp_path / "mmproj.gguf") if mmproj else None,
    )
    captured: dict = {"images": None, "user": "", "backend": None}
    prompts: list = []

    def fake_load(kind, **kwargs):
        prompts.append((kind, kwargs))
        if kind == "flux":
            graph = {
                "1": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "flux-2-klein-4b.safetensors"},
                }
            }
            return graph, {}
        graph = {
            "2": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"length": 9}},
            "3": {"class_type": "Note", "inputs": {"text": "ltx smoke"}},
        }
        return graph, {"frames": 9}

    monkeypatch.setattr("master_agent.comfy.workflow_patcher.load_and_patch_workflow", fake_load)

    def chat(backend, system, user_text, images):
        captured["backend"] = backend
        captured["user"] = user_text
        captured["system"] = system
        captured["images"] = list(images)
        if callable(vision):
            return vision(backend, system, user_text, images)
        return vision

    monkeypatch.setattr("master_agent.judge.vision._vision_chat", chat)
    posts: list[dict] = []
    views: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/prompt":
            posts.append(json.loads(request.content.decode()))
            return httpx.Response(200, json={"prompt_id": f"p{len(posts)}"})
        if request.url.path.startswith("/history/"):
            pid = request.url.path.rsplit("/", 1)[-1]
            if pid == "p1" and history_image:
                body = {
                    pid: {
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "gate.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        }
                    }
                }
            elif pid == "p1":
                body = {pid: {"outputs": {"9": {"images": []}}}}
            else:
                body = {pid: {"outputs": {"2": {"gifs": [{"filename": "gate.mp4", "type": "output"}]}}}}
            return httpx.Response(200, json=body)
        if request.url.path == "/view":
            views.append(request.url.params.get("filename") or "")
            return httpx.Response(200, content=IMAGE)
        return httpx.Response(404)

    _install_http(monkeypatch, handler)
    from io import StringIO
    from contextlib import redirect_stdout

    buf = StringIO()
    try:
        with redirect_stdout(buf):
            code = first_gate.command()
    finally:
        ComfyClient.close_pool()
    stills = sum(1 for post in posts if "flux-2-klein-4b" in json.dumps(post))
    clips = sum(1 for post in posts if "ltx" in json.dumps(post).lower())
    return {
        "code": code,
        "passed": first_gate.is_passed(),
        "stills": stills,
        "clips": clips,
        "images": captured["images"],
        "user_text": captured["user"],
        "backend": captured["backend"],
        "llm": llm,
        "started": started,
        "prompts": prompts,
        "view_name": views[0] if views else None,
        "output": buf.getvalue(),
        "posts": posts,
    }


def _assert_later_submit_blocked(monkeypatch):
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request.url.path)
        return httpx.Response(200, json={"prompt_id": "later"})

    _install_http(monkeypatch, handler)
    try:
        with pytest.raises(Exception, match="FIRST GATE"):
            ComfyClient("http://comfy.test").queue_prompt(
                {"1": {"class_type": "KSampler", "inputs": {}}}
            )
    finally:
        ComfyClient.close_pool()
    assert posts == []
