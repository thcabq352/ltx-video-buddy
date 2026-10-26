"""Run state machine end to end over a mocked Comfy HTTP server.

SELECT_VARIANT → PATCH → VALIDATE → SUBMIT → POLL → RESOLVE → JUDGE → DONE | ERROR
plus PLAN_OOM_RETRY and the judge revise loop.

The real ``ComfyClient`` runs against ``httpx.MockTransport``. The patcher,
linter, OOM ladder, resolve copy, judge loop, provenance and ``_finish`` are
real. Only the LLM judge, weight presence, video probing and KB ingest are
stubbed. No network, no GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from master_agent.comfy.client import ComfyClient
from master_agent.config import DOWNSCALE_LADDER, MAX_RETRIES, OBJECT_INFO_CACHE
from master_agent.orchestrator.machine import Orchestrator
from master_agent.orchestrator.state import (
    LOOP_ERROR,
    LOOP_EXHAUSTED,
    LOOP_HUMAN_VETO,
    LOOP_PASSED,
    STATES,
    RunState,
)
from master_agent.provenance import CLIP_PROVENANCE_SCHEMA, missing_required, sidecar_path

BASE_URL = "http://127.0.0.1:18188"
REQUEST = "a red fox trotting through fresh snow, cinematic"

# Every edge the orchestrator may take. The revise loop and the OOM retry
# re-run patch/validate/submit without re-labelling those states, so they
# go straight to POLL.
LEGAL_EDGES = {
    "SELECT_VARIANT": {"PATCH", "JUDGE", "DONE", "ERROR"},
    "PATCH": {"VALIDATE", "POLL", "PLAN_OOM_RETRY", "ERROR"},
    "VALIDATE": {"SUBMIT", "ERROR"},
    "SUBMIT": {"POLL", "PLAN_OOM_RETRY", "ERROR"},
    "POLL": {"RESOLVE", "PLAN_OOM_RETRY", "ERROR"},
    "PLAN_OOM_RETRY": {"POLL", "PLAN_OOM_RETRY", "ERROR"},
    "RESOLVE": {"JUDGE", "DONE", "ERROR"},
    "JUDGE": {"DONE", "PATCH", "JUDGE", "ERROR"},
    "DONE": set(),
    "ERROR": set(),
}


def state_path(st: RunState) -> list[str]:
    return ["SELECT_VARIANT"] + [m[3:] for m in st.messages if m.startswith("-> ")]


def assert_legal_path(st: RunState) -> list[str]:
    path = state_path(st)
    for src, dst in zip(path, path[1:]):
        assert dst in STATES, f"unknown state {dst!r} in {path}"
        assert dst in LEGAL_EDGES[src], f"illegal edge {src} -> {dst} in {path}"
    assert path[-1] in ("DONE", "ERROR"), f"run did not end terminal: {path}"
    assert path.count("DONE") + path.count("ERROR") == 1, f"terminal re-entered: {path}"
    return path


class FakeComfy:
    """Minimal ComfyUI HTTP surface: /object_info, /free, /prompt, /history, /upload."""

    def __init__(self, out_dir: Path, object_info: dict[str, Any]):
        self.out_dir = out_dir
        self.object_info = object_info
        self.object_info_status = 200
        self.queued: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str]] = []
        # One entry per /prompt call; the last one repeats.
        #   "ok"            → prompt_id, history completes with a written mp4
        #   "missing_file"  → history lists an mp4 that is not on disk
        #   "no_video"      → history completes with only a .txt file
        #   "pending"       → history never shows the prompt (poll timeout)
        #   "job_error"     → history status_str=error
        #   "job_oom"       → history status_str=error with a CUDA OOM message
        #   "submit_oom"    → /prompt 500 with a CUDA OOM body
        #   "submit_400"    → /prompt 400 with node_errors
        self.script: list[str] = ["ok"]
        self._behaviour: dict[str, str] = {}

    def _next(self) -> str:
        idx = min(len(self.queued) - 1, len(self.script) - 1)
        return self.script[idx]

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if path == "/object_info":
            if self.object_info_status != 200:
                return httpx.Response(self.object_info_status, text="down")
            return httpx.Response(200, json=self.object_info)
        if path == "/free":
            return httpx.Response(200)
        if path == "/upload/image":
            return httpx.Response(200, json={"name": "uploaded.png"})
        if path == "/prompt":
            body = json.loads(request.content)
            self.queued.append(body["prompt"])
            mode = self._next()
            if mode == "submit_oom":
                return httpx.Response(500, text="torch.cuda.OutOfMemoryError: CUDA out of memory")
            if mode == "submit_400":
                return httpx.Response(
                    400, json={"error": "invalid prompt", "node_errors": {"20": "bad length"}}
                )
            pid = f"p{len(self.queued)}"
            self._behaviour[pid] = mode
            return httpx.Response(200, json={"prompt_id": pid, "number": len(self.queued)})
        if path.startswith("/history/"):
            pid = path.rsplit("/", 1)[-1]
            mode = self._behaviour.get(pid, "pending")
            if mode == "pending":
                return httpx.Response(200, json={})
            if mode in ("job_error", "job_oom"):
                msg = "CUDA out of memory" if mode == "job_oom" else "KSampler exploded"
                return httpx.Response(
                    200,
                    json={
                        pid: {
                            "outputs": {},
                            "status": {
                                "status_str": "error",
                                "completed": False,
                                "messages": [["execution_error", {"exception_message": msg}]],
                            },
                        }
                    },
                )
            if mode == "no_video":
                outputs = {"90": {"files": [{"filename": "log.txt", "type": "output"}]}}
            else:
                name = f"master_agent_{pid}.mp4"
                if mode == "ok":
                    self.out_dir.mkdir(parents=True, exist_ok=True)
                    (self.out_dir / name).write_bytes(f"video-bytes-{pid}".encode())
                outputs = {
                    "90": {"images": [{"filename": name, "subfolder": "", "type": "output"}]}
                }
            return httpx.Response(
                200,
                json={pid: {"outputs": outputs, "status": {"status_str": "success", "completed": True}}},
            )
        return httpx.Response(404, text=f"no route {path}")


class FastClient(ComfyClient):
    poll_timeout = 2.0

    def wait_for_prompt(self, prompt_id, **_kw):
        return super().wait_for_prompt(prompt_id, poll_interval=0.0, timeout=self.poll_timeout)


@pytest.fixture(scope="module")
def object_info() -> dict[str, Any]:
    return json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))


@pytest.fixture
def rig(tmp_path, monkeypatch, object_info):
    """Wire a FakeComfy, isolate outputs/runs, stub judge LLM + weights + probe."""
    comfy = FakeComfy(tmp_path / "comfy_out", object_info)
    ComfyClient.close_pool()
    ComfyClient._pool[BASE_URL] = httpx.Client(transport=httpx.MockTransport(comfy.handler))

    real_resolve = ComfyClient.resolve_output_path.__func__ if hasattr(
        ComfyClient.resolve_output_path, "__func__"
    ) else ComfyClient.resolve_output_path
    monkeypatch.setattr(
        ComfyClient,
        "resolve_output_path",
        staticmethod(lambda info, output_dir=comfy.out_dir: real_resolve(info, output_dir)),
    )
    monkeypatch.setattr("master_agent.orchestrator.director.DIRECTOR_LLM", False)
    monkeypatch.setattr("master_agent.orchestrator.machine.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr("master_agent.models.weights.require_weights", lambda *_a, **_k: None)
    monkeypatch.setattr("master_agent.models.weights.resolve_ltx23_gguf", lambda *_a, **_k: None)
    models = tmp_path / "models"
    q4 = models / "unet" / "LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf"
    q4.parent.mkdir(parents=True)
    q4.write_bytes(b"gguf")
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [models])
    # base/eros/directors now load 10Eros through UnetLoaderGGUF. That name is
    # not in the tower object_info cache, so the validator asks the disk.
    # Weight presence is stubbed for this machine: a named file counts.
    monkeypatch.setattr(
        "master_agent.models.weights.find_weight_file",
        lambda name, *_a, **_k: Path(str(name)) if name else None,
    )
    monkeypatch.setattr("master_agent.judge.probe.probe_video", lambda *_a, **_k: {})
    monkeypatch.setattr("master_agent.orchestrator.machine.analyze", lambda *a, **k: (0.95, []))
    monkeypatch.setattr("master_agent.judge.judge._vision_review_safe", lambda *a, **k: None)
    ingested: list[dict] = []
    monkeypatch.setattr(
        "master_agent.kb.ingest.ingest_run_record", lambda rec: ingested.append(rec) or True
    )
    verdicts: list[dict] = [_PASS]
    llm_calls = {"n": 0}

    def _llm(**_k):
        llm_calls["n"] += 1
        return dict(verdicts[min(llm_calls["n"] - 1, len(verdicts) - 1)])

    monkeypatch.setattr("master_agent.judge.judge._llm_judge", _llm)

    class Rig:
        pass

    r = Rig()
    r.comfy = comfy
    r.tmp = tmp_path
    r.verdicts = verdicts
    r.llm_calls = llm_calls
    r.ingested = ingested
    r.client = FastClient(base_url=BASE_URL)
    r.orch = Orchestrator(client=r.client)
    yield r
    ComfyClient.close_pool()


_PASS = {
    "pass": True,
    "score": 0.93,
    "brief_adherence": 0.9,
    "issues": [],
    "prompt_rewrite": "",
    "reason": "on brief",
}
_FAIL_REWRITE = {
    "pass": False,
    "score": 0.35,
    "brief_adherence": 0.8,
    "issues": ["fox barely moves"],
    "prompt_rewrite": "a red fox bounding through deep snow, motion blur on the legs",
    "param_hints": {"seed": 4242},
    "reason": "needs more motion",
}


def run(r, **kw) -> RunState:
    args = dict(
        variant="base",
        duration_s=2.0,
        width=640,
        height=384,
        seed=7,
        judge_enabled=True,
        power_mode=False,
        max_judge_rounds=3,
    )
    args.update(kw)
    return r.orch.run(REQUEST, **args)


def run_records(r) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted((r.tmp / "runs").glob("*.json"))
    ]


def prompts_posted(r) -> int:
    return sum(1 for m, p in r.comfy.calls if m == "POST" and p == "/prompt")


# --------------------------------------------------------------------------- happy path


def test_happy_path_walks_every_stage_in_order(rig):
    st = run(rig)
    assert st.state == "DONE", st.error
    assert assert_legal_path(st) == [
        "SELECT_VARIANT",
        "PATCH",
        "VALIDATE",
        "SUBMIT",
        "POLL",
        "RESOLVE",
        "JUDGE",
        "DONE",
    ]
    assert st.error is None
    assert st.loop_status == LOOP_PASSED
    assert st.judge_decision == "accept"
    assert st.prompt_id == "p1"
    assert prompts_posted(rig) == 1
    assert ("POST", "/free") in rig.comfy.calls


def test_happy_path_queues_the_patched_graph(rig):
    run(rig)
    wf = rig.comfy.queued[0]
    assert wf["10"]["inputs"]["text"] == REQUEST
    assert wf["42"]["inputs"]["noise_seed"] == 7
    assert (wf["20"]["inputs"]["width"], wf["20"]["inputs"]["height"]) == (640, 384)
    length = wf["20"]["inputs"]["length"]
    assert length == 49 and (length - 1) % 8 == 0
    assert wf["21"]["inputs"]["frames_number"] == length
    assert wf["90"]["inputs"]["filename_prefix"] == "master_agent"


def test_resolve_copies_render_and_finish_writes_provenance_and_record(rig):
    st = run(rig)
    clip = Path(st.video_path)
    assert clip.is_file()
    assert clip.name == "shot-1.mp4"
    assert clip.read_bytes() == b"video-bytes-p1"
    sidecar = sidecar_path(clip)
    assert sidecar.is_file()
    assert st.provenance_sidecar == str(sidecar)
    prov = json.loads(sidecar.read_text(encoding="utf-8"))
    assert prov["schema"] == CLIP_PROVENANCE_SCHEMA
    assert missing_required(prov) == []
    assert prov["hash"] == hashlib.sha256(b"video-bytes-p1").hexdigest()
    assert prov["output_path"] == str(clip)
    assert prov["engine"]["variant"] == "base"
    assert prov["params"]["seed"] == 7
    assert prov["lineage"]["attempt_id"] == "shot-1.a1"
    assert prov["judge"]["score"] > 0

    records = run_records(rig)
    assert len(records) == 1
    rec = records[0]
    assert rec["run_id"] == st.run_id
    assert rec["state"] == "DONE"
    assert rec["loop_status"] == LOOP_PASSED
    assert rec["provenance"]["hash"] == prov["hash"]
    assert any(m.startswith("run record: ") for m in st.messages)
    assert len(rigs_ingested := rig.ingested) == 1 and rigs_ingested[0]["run_id"] == st.run_id


def test_no_judge_skips_judge_state_and_still_finalizes(rig):
    st = run(rig, judge_enabled=False, revise_enabled=False)
    assert assert_legal_path(st) == [
        "SELECT_VARIANT",
        "PATCH",
        "VALIDATE",
        "SUBMIT",
        "POLL",
        "RESOLVE",
        "DONE",
    ]
    assert st.judge_decision == "skipped"
    assert st.loop_status == LOOP_PASSED
    assert rig.llm_calls["n"] == 0
    assert st.provenance["judge"]["fail_reasons"] == [{"kind": "judge", "detail": "skipped"}]
    assert st.provenance["hash"]


# --------------------------------------------------------------------------- ERROR per stage


def _assert_error(st: RunState, *, last_stage: str, contains: str) -> None:
    path = assert_legal_path(st)
    assert st.state == "ERROR"
    assert path[-2] == last_stage, path
    assert st.error and contains in st.error, st.error
    assert st.loop_status == LOOP_ERROR
    assert f"FAILED: {st.error}" in st.messages


def test_error_at_select_variant_partner_refusal(rig, monkeypatch):
    monkeypatch.setattr(
        "master_agent.comfy.partner_pointers.pack_c_video_url_allowed", lambda: False
    )
    st = run(rig, variant="seedance25_draft_t2v")
    _assert_error(st, last_stage="SELECT_VARIANT", contains="Local-only")
    assert prompts_posted(rig) == 0
    assert len(run_records(rig)) == 1
    assert st.provenance and st.provenance["schema"] == CLIP_PROVENANCE_SCHEMA


def test_error_at_select_variant_media_route(rig):
    st = run(rig, variant="lipsync")
    _assert_error(st, last_stage="SELECT_VARIANT", contains="lipsync needs a source video")
    assert prompts_posted(rig) == 0
    assert len(run_records(rig)) == 1


def test_error_at_select_variant_unexpected_exception_is_captured(rig, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("director crashed")

    monkeypatch.setattr("master_agent.orchestrator.director.choose_variant", boom)
    st = run(rig, variant=None)
    _assert_error(
        st, last_stage="SELECT_VARIANT", contains="variant selection failed: director crashed"
    )
    assert prompts_posted(rig) == 0
    assert len(run_records(rig)) == 1


def test_error_at_patch_exception(rig, monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("template exploded")

    monkeypatch.setattr("master_agent.orchestrator.machine.load_and_patch_workflow", boom)
    st = run(rig)
    _assert_error(st, last_stage="PATCH", contains="patch failed: template exploded")
    assert prompts_posted(rig) == 0
    assert not any(p == "/object_info" for _m, p in rig.comfy.calls)


def test_error_at_patch_media_wiring_refuses_dropped_image(rig, monkeypatch):
    monkeypatch.setattr(
        "master_agent.orchestrator.talking.media_route_error", lambda *_a, **_k: None
    )
    st = run(rig, image_name="still.png")
    _assert_error(st, last_stage="PATCH", contains="does not feed the sampler")
    assert prompts_posted(rig) == 0


def test_error_at_validate_missing_node_class(rig, object_info):
    rig.comfy.object_info = {k: v for k, v in object_info.items() if k != "LTXVScheduler"}
    st = run(rig)
    _assert_error(st, last_stage="VALIDATE", contains="workflow validation failed")
    assert "LTXVScheduler" in st.error
    assert prompts_posted(rig) == 0


def test_error_at_validate_no_object_info(rig, monkeypatch):
    from master_agent.comfy.client import ComfyClientError

    def down(*_a, **_k):
        raise ComfyClientError("ComfyUI is down and no cache exists")

    monkeypatch.setattr(rig.client, "load_object_info", down)
    st = run(rig)
    _assert_error(st, last_stage="VALIDATE", contains="cannot load /object_info")
    assert prompts_posted(rig) == 0


def test_error_at_submit_missing_weights(rig, monkeypatch):
    from master_agent.models.weights import MissingWeightsError

    def missing(_variant, **_k):
        raise MissingWeightsError("base needs LTX2.3 checkpoint (missing)", [])

    monkeypatch.setattr("master_agent.models.weights.require_weights", missing)
    st = run(rig)
    _assert_error(st, last_stage="SUBMIT", contains="base needs LTX2.3 checkpoint")
    assert prompts_posted(rig) == 0


def test_error_at_submit_comfy_rejects_prompt(rig):
    rig.comfy.script = ["submit_400"]
    st = run(rig)
    _assert_error(st, last_stage="SUBMIT", contains="submit failed: ComfyUI /prompt failed (400)")
    assert "bad length" in st.error
    assert st.retries == 0
    assert prompts_posted(rig) == 1


def test_error_at_poll_job_error(rig):
    rig.comfy.script = ["job_error"]
    st = run(rig)
    _assert_error(st, last_stage="POLL", contains="poll failed: ComfyUI job error")
    assert "KSampler exploded" in st.error
    assert st.prompt_id == "p1"


def test_error_at_poll_timeout(rig):
    rig.comfy.script = ["pending"]
    rig.client.poll_timeout = 0.05
    st = run(rig)
    _assert_error(st, last_stage="POLL", contains="poll failed: Timed out after 0.05s")
    assert "prompt_id=p1" in st.error
    assert st.retries == 0


def test_error_at_poll_completed_without_video(rig):
    rig.comfy.script = ["no_video"]
    st = run(rig)
    _assert_error(st, last_stage="POLL", contains="produced no video files")


def test_error_at_resolve_output_missing_on_disk(rig):
    rig.comfy.script = ["missing_file"]
    st = run(rig)
    _assert_error(st, last_stage="RESOLVE", contains="output files missing on disk")
    assert "master_agent_p1.mp4" in st.error
    assert not Path(st.planned_clip).is_file()
    assert st.provenance["hash"] is None


def test_error_at_judge_unhandled_exception(rig, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("judge backend fell over")

    monkeypatch.setattr("master_agent.orchestrator.judge_loop.judge_segment", boom)
    st = run(rig)
    _assert_error(
        st, last_stage="JUDGE", contains="unhandled orchestrator error: judge backend fell over"
    )
    assert Path(st.video_path).is_file()
    assert len(run_records(rig)) == 1


def test_error_run_record_carries_error_and_terminal_state(rig):
    rig.comfy.script = ["job_error"]
    st = run(rig)
    (rec,) = run_records(rig)
    assert rec["state"] == "ERROR"
    assert rec["error"] == st.error
    assert rec["loop_status"] == LOOP_ERROR
    assert rec["provenance"]["hash"] is None
    assert missing_required(rec["provenance"]) == []


# --------------------------------------------------------------------------- OOM retry


def test_submit_oom_walks_one_rung_and_retries(rig):
    rig.comfy.script = ["submit_oom", "ok"]
    st = run(rig)
    path = assert_legal_path(st)
    assert st.state == "DONE", st.error
    assert path[:6] == ["SELECT_VARIANT", "PATCH", "VALIDATE", "SUBMIT", "PLAN_OOM_RETRY", "POLL"]
    assert st.retries == 1
    assert st.downscale_level == 1
    w, h, f = DOWNSCALE_LADDER[1]
    retry = rig.comfy.queued[1]["20"]["inputs"]
    assert (retry["width"], retry["height"], retry["length"]) == (w, h, f)
    assert rig.comfy.queued[1]["21"]["inputs"]["frames_number"] == f
    assert (st.width, st.height, st.frames) == (w, h, f)
    assert st.duration_s == pytest.approx(f / 24)


def test_poll_oom_also_retries(rig):
    rig.comfy.script = ["job_oom", "ok"]
    st = run(rig)
    path = assert_legal_path(st)
    assert st.state == "DONE", st.error
    assert ["POLL", "PLAN_OOM_RETRY", "POLL"] == path[4:7]
    assert st.retries == 1
    assert prompts_posted(rig) == 2


def test_oom_exhausts_ladder_then_errors(rig):
    rig.comfy.script = ["submit_oom"]
    st = run(rig)
    path = assert_legal_path(st)
    assert st.state == "ERROR"
    rungs = len(DOWNSCALE_LADDER) - 1
    assert st.retries == min(rungs, MAX_RETRIES) + 1
    assert path.count("PLAN_OOM_RETRY") == min(rungs, MAX_RETRIES)
    assert st.error.startswith("submit OOM after")
    assert "out of memory" in st.error.lower()
    assert prompts_posted(rig) == min(rungs, MAX_RETRIES) + 1
    assert st.loop_status == LOOP_ERROR


# --------------------------------------------------------------------------- judge revise loop


def test_judge_fail_repatches_with_rewrite_and_reseed_then_passes(rig):
    rig.verdicts[:] = [_FAIL_REWRITE, _PASS]
    st = run(rig)
    path = assert_legal_path(st)
    assert st.state == "DONE", st.error
    assert path == [
        "SELECT_VARIANT", "PATCH", "VALIDATE", "SUBMIT", "POLL", "RESOLVE", "JUDGE",
        "PATCH", "POLL", "RESOLVE", "JUDGE", "DONE",
    ]
    assert st.loop_status == LOOP_PASSED
    assert st.attempt == 2
    assert prompts_posted(rig) == 2
    first, second = rig.comfy.queued
    assert first["10"]["inputs"]["text"] == REQUEST
    assert second["10"]["inputs"]["text"] == _FAIL_REWRITE["prompt_rewrite"]
    assert second["42"]["inputs"]["noise_seed"] == 4242
    assert len(st.revise_history) == 1 and st.revise_history[0]["attempt"] == 1
    assert st.parent_attempt_id == "shot-1.a1"
    assert Path(st.video_path).read_bytes() == b"video-bytes-p2"
    assert st.provenance["lineage"]["attempt_id"] == "shot-1.a2"
    assert st.provenance["hash"] == hashlib.sha256(b"video-bytes-p2").hexdigest()


def test_revise_loop_never_reselects_variant(rig, monkeypatch):
    calls = {"n": 0}
    real = Orchestrator._select_variant

    def counting(self, st, force):
        calls["n"] += 1
        return real(self, st, force)

    monkeypatch.setattr(Orchestrator, "_select_variant", counting)
    rig.verdicts[:] = [_FAIL_REWRITE, _PASS]
    st = run(rig)
    assert st.state == "DONE"
    assert calls["n"] == 1
    assert st.variant == "base"
    assert "SELECT_VARIANT" not in state_path(st)[1:]


def test_judge_fail_until_budget_is_exhausted_not_accept(rig):
    rig.verdicts[:] = [_FAIL_REWRITE]
    st = run(rig, max_judge_rounds=2)
    path = assert_legal_path(st)
    assert st.state == "DONE"
    assert st.loop_status == LOOP_EXHAUSTED
    assert st.judge_decision == "exhausted"
    assert st.attempt == 2
    assert prompts_posted(rig) == 2
    assert path.count("JUDGE") == 2


def test_human_veto_stops_without_requeue(rig):
    rig.verdicts[:] = [
        {
            "pass": False,
            "score": 0.95,
            "brief_adherence": 0.1,
            "issues": ["pretty but off brief"],
            "prompt_rewrite": "",
            "reason": "not what was asked",
        }
    ]
    st = run(rig)
    assert_legal_path(st)
    assert st.state == "DONE"
    assert st.loop_status == LOOP_HUMAN_VETO
    assert prompts_posted(rig) == 1


def test_revise_rerun_that_fails_validation_ends_in_error(rig, object_info, monkeypatch):
    rig.verdicts[:] = [_FAIL_REWRITE, _PASS]
    real_validate = Orchestrator._validate
    seen = {"n": 0}

    def flaky_validate(self, st):
        seen["n"] += 1
        if seen["n"] == 2:
            rig.comfy.object_info = {k: v for k, v in object_info.items() if k != "RandomNoise"}
        return real_validate(self, st)

    monkeypatch.setattr(Orchestrator, "_validate", flaky_validate)
    st = run(rig)
    path = assert_legal_path(st)
    assert st.state == "ERROR"
    assert path[-3:] == ["JUDGE", "PATCH", "ERROR"]
    assert "workflow validation failed" in st.error and "RandomNoise" in st.error
    assert st.loop_status == LOOP_ERROR
    assert prompts_posted(rig) == 1
    (rec,) = run_records(rig)
    assert rec["state"] == "ERROR" and rec["attempt"] == 2


def test_revise_rerun_poll_timeout_ends_in_error(rig):
    rig.verdicts[:] = [_FAIL_REWRITE, _PASS]
    rig.comfy.script = ["ok", "pending"]
    rig.client.poll_timeout = 0.05
    st = run(rig)
    path = assert_legal_path(st)
    assert st.state == "ERROR"
    assert path[-3:] == ["PATCH", "POLL", "ERROR"]
    assert "Timed out" in st.error
    assert st.loop_status == LOOP_ERROR


# --------------------------------------------------------------------------- _finish / re-entry


@pytest.mark.parametrize(
    "state, judge_decision, loop_in, loop_out",
    [
        ("ERROR", "", "started", LOOP_ERROR),
        ("ERROR", "", "", LOOP_ERROR),
        ("DONE", "accept", "started", LOOP_PASSED),
        ("DONE", "rewrite", "started", LOOP_EXHAUSTED),
        ("DONE", "accept", LOOP_HUMAN_VETO, LOOP_HUMAN_VETO),
        ("DONE", "exhausted", LOOP_EXHAUSTED, LOOP_EXHAUSTED),
        ("ERROR", "", LOOP_ERROR, LOOP_ERROR),
    ],
)
def test_finish_maps_loop_status(rig, state, judge_decision, loop_in, loop_out):
    st = RunState(request=REQUEST, run_id="fin1", variant="base")
    st.state = state
    st.judge_decision = judge_decision
    st.loop_status = loop_in
    out = rig.orch._finish(st)
    assert out is st
    assert st.loop_status == loop_out
    assert st.provenance["schema"] == CLIP_PROVENANCE_SCHEMA
    assert Path(st.provenance_sidecar).is_file()
    (rec,) = run_records(rig)
    assert rec["loop_status"] == loop_out


def test_finish_is_idempotent_on_reentry(rig):
    st = run(rig)
    before = json.loads(json.dumps(st.to_dict(), default=str))
    path_before = state_path(st)
    sidecar_before = Path(st.provenance_sidecar).read_text(encoding="utf-8")
    rig.orch._finish(st)
    after = json.loads(json.dumps(st.to_dict(), default=str))
    assert after["state"] == before["state"] == "DONE"
    assert after["loop_status"] == before["loop_status"]
    assert after["provenance"] == before["provenance"]
    assert after["provenance_history"] == before["provenance_history"]
    assert Path(st.provenance_sidecar).read_text(encoding="utf-8") == sidecar_before
    assert state_path(st) == path_before
    assert prompts_posted(rig) == 1
    assert {r["run_id"] for r in run_records(rig)} == {st.run_id}


def test_orchestrator_reuse_keeps_runs_independent(rig):
    first = run(rig)
    rig.comfy.script = ["missing_file"]
    second = run(rig)
    assert first.run_id != second.run_id
    assert first.state == "DONE"
    assert second.state == "ERROR"
    assert "master_agent_p2.mp4" in second.error
    assert Path(first.video_path).read_bytes() == b"video-bytes-p1"
    assert Path(first.output_dir) != Path(second.output_dir)
    assert not Path(second.planned_clip).is_file()
    assert len(run_records(rig)) == 2


# --------------------------------------------------------------------------- RunState itself


def test_state_catalog_matches_documented_flow():
    assert STATES[:7] == (
        "SELECT_VARIANT", "PATCH", "VALIDATE", "SUBMIT", "POLL", "RESOLVE", "JUDGE",
    )
    assert set(STATES) == set(LEGAL_EDGES)
    assert RunState(request="x").state == "SELECT_VARIANT"


def test_fail_records_error_and_moves_to_error():
    st = RunState(request="x")
    st.transition("PATCH")
    st.fail("nope")
    assert st.state == "ERROR"
    assert st.error == "nope"
    assert st.messages[-2:] == ["-> ERROR", "FAILED: nope"]


@pytest.mark.xfail(
    strict=True,
    reason="RunState.transition does not guard edges; illegal moves are only caught "
    "by the orchestrator-path assertions in this file",
)
@pytest.mark.parametrize(
    "src, dst",
    [("SELECT_VARIANT", "RESOLVE"), ("DONE", "PATCH"), ("ERROR", "JUDGE"), ("PATCH", "BOGUS")],
)
def test_runstate_rejects_illegal_transition(src, dst):
    st = RunState(request="x")
    st.state = src
    with pytest.raises((ValueError, RuntimeError)):
        st.transition(dst)
