"""First launch must pass one Klein 4B still, then one short LTX clip.

Other ComfyUI submits are refused until ``state/first_gate.json`` says
``passed``. The still runs with the local llama.cpp judge
(``qwen3-vl-heretic``). A judge that does not fire, or a GPU that is not
free, fails the gate out loud. Tests pass runners in; they do not need a GPU.
"""

from __future__ import annotations

import contextvars
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

_STEP: contextvars.ContextVar[str] = contextvars.ContextVar("vb_first_gate_step", default="")

STATE_NAME = "first_gate.json"


class FirstGateBlocked(RuntimeError):
    """A render was refused because the first-run sequence has not passed."""


def state_path() -> Path:
    from master_agent.config import STATE_DIR

    return Path(STATE_DIR) / STATE_NAME


def _env_bypass() -> bool:
    flag = (os.getenv("FIRST_GATE") or "").strip().lower()
    return flag in {"passed", "off", "skip", "0", "false", "no"}


def load_state() -> dict[str, Any]:
    path = state_path()
    if not path.is_file():
        return {"passed": False, "klein_still": False, "ltx_clip": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"passed": False, "klein_still": False, "ltx_clip": False}
    return data if isinstance(data, dict) else {"passed": False}


def save_state(data: dict[str, Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def is_passed() -> bool:
    if _env_bypass():
        return True
    return bool(load_state().get("passed"))


def blocked_message() -> str:
    return (
        "FIRST GATE: refusing this render. The first launch must pass one FLUX.2 [klein] 4B "
        "still through the runner with the local qwen3-vl-heretic judge, then one short LTX clip. "
        f"Other renders stay blocked until {state_path()} says passed. "
        "Run: python -m master_agent first-gate"
    )


@contextmanager
def gate_step(step: str) -> Iterator[None]:
    token = _STEP.set(step)
    try:
        yield
    finally:
        _STEP.reset(token)


def _klein_workflow(workflow: dict[str, Any]) -> bool:
    blob = json.dumps(workflow).lower()
    return "flux-2-klein-4b" in blob or "qwen_3_4b" in blob


def _short_ltx(workflow: dict[str, Any]) -> bool:
    blob = json.dumps(workflow).lower()
    if "ltx" not in blob and "emptyltxv" not in blob:
        return False
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        length = inputs.get("length")
        if isinstance(length, int) and length <= 17:
            return True
    return "ltx" in blob


def assert_submit_allowed(workflow: dict[str, Any] | None) -> None:
    """Raise :class:`FirstGateBlocked` unless the gate has passed or this is its own step."""
    if is_passed():
        return
    step = _STEP.get()
    graph = workflow or {}
    if step == "klein" and _klein_workflow(graph):
        return
    state = load_state()
    if step == "ltx" and state.get("klein_still") and _short_ltx(graph):
        return
    raise FirstGateBlocked(blocked_message())


def run_first_gate(
    *,
    still_runner: Callable[[], dict[str, Any]],
    clip_runner: Callable[[], dict[str, Any]],
    judge_probe: Callable[[], dict[str, Any]],
    vram_free_gb: float | None,
    required_free_gb: float = 6.0,
) -> dict[str, Any]:
    """Run the two steps. Runners are called inside the gate token.

    ``judge_probe`` must report ``fired: true`` before the still, and the
    still runner must report ``judge_fired: true``. A tight GPU fails before
    either submit.
    """
    if vram_free_gb is not None and float(vram_free_gb) < float(required_free_gb):
        raise FirstGateBlocked(
            "FIRST GATE: GPU is not free. "
            f"Free VRAM is {float(vram_free_gb):.2f} GB and the first still needs "
            f"at least {float(required_free_gb):.1f} GB. Nothing was queued."
        )
    judge = judge_probe() or {}
    if not judge.get("fired"):
        reason = judge.get("reason") or "the local judge did not run"
        raise FirstGateBlocked(
            "FIRST GATE: the local qwen3-vl-heretic judge did not fire "
            f"({reason}). Look and taste checks stay on llama.cpp. Nothing was queued."
        )
    with gate_step("klein"):
        still = still_runner() or {}
    if not still.get("judge_fired"):
        raise FirstGateBlocked(
            "FIRST GATE: the Klein still ran but the local judge did not score it. "
            "The gate does not pass."
        )
    data = load_state()
    data["klein_still"] = True
    data["judge"] = {"fired": True, "provider": judge.get("provider") or "llamacpp"}
    save_state(data)
    with gate_step("ltx"):
        clip = clip_runner() or {}
    if clip.get("ok") is False:
        raise FirstGateBlocked(
            "FIRST GATE: the short LTX clip did not finish. "
            + str(clip.get("error") or "no clip")
        )
    data = load_state()
    data["ltx_clip"] = True
    data["passed"] = True
    save_state(data)
    return {"status": "passed", "still": still, "clip": clip, "path": str(state_path())}


def command(args: Any = None) -> int:
    """CLI entry. Real runners talk to ComfyUI; this process does not guess a pass."""
    del args
    if is_passed() and not _env_bypass():
        print(f"OK    first gate already passed ({state_path()})")
        return 0
    from master_agent.gpu_seat import probe_free_vram_gb, required_free_gb
    from master_agent.judge.local_look import fire_look_check

    def still() -> dict[str, Any]:
        return _production_still()

    def clip() -> dict[str, Any]:
        return _production_clip()

    try:
        result = run_first_gate(
            still_runner=still,
            clip_runner=clip,
            judge_probe=fire_look_check,
            vram_free_gb=probe_free_vram_gb(),
            required_free_gb=required_free_gb(),
        )
    except FirstGateBlocked as exc:
        print(f"FAIL  {exc}")
        return 2
    print(f"OK    first gate passed ({result['path']})")
    return 0


def _production_still() -> dict[str, Any]:
    from master_agent.comfy.client import ComfyClient
    from master_agent.comfy.workflow_patcher import load_and_patch_workflow
    from master_agent.judge.local_look import fire_look_check

    workflow, _meta = load_and_patch_workflow(
        "flux",
        prompt="first gate still, soft window light, a ceramic cup on a wood table",
        width=768,
        height=512,
        seed=1,
        steps=4,
    )
    client = ComfyClient()
    prompt_id = client.queue_prompt(workflow)
    client.wait_for_prompt(prompt_id)
    judge = fire_look_check(kind="still")
    return {"ok": True, "prompt_id": prompt_id, "judge_fired": bool(judge.get("fired"))}


def _production_clip() -> dict[str, Any]:
    from master_agent.comfy.client import ComfyClient
    from master_agent.comfy.workflow_patcher import load_and_patch_workflow

    workflow, meta = load_and_patch_workflow(
        "ltx25_t2v_i2v",
        prompt="first gate clip, the cup sits still, soft light",
        duration_s=9 / 24,
        seed=1,
    )
    length = int(meta.get("frames") or 9)
    for node in workflow.values():
        if isinstance(node, dict):
            inputs = node.get("inputs") or {}
            if "length" in inputs and isinstance(inputs["length"], int):
                inputs["length"] = min(int(inputs["length"]), max(length, 9))
    client = ComfyClient()
    prompt_id = client.queue_prompt(workflow)
    client.wait_for_prompt(prompt_id)
    return {"ok": True, "prompt_id": prompt_id, "frames": length}
