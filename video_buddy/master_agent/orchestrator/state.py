"""Orchestrator run state — one video generation run end to end."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# States in normal flow order; ERROR / DONE are terminal
STATES = (
    "SELECT_VARIANT",
    "PATCH",
    "VALIDATE",
    "SUBMIT",
    "POLL",
    "RESOLVE",
    "JUDGE",
    "PLAN_OOM_RETRY",
    "DONE",
    "ERROR",
)

# Closed judge→revise→rerun loop. DONE is the machine state; loop_status
# says why we stopped (accept used to hide budget exhaustion).
LOOP_STARTED = "started"
LOOP_PASSED = "passed"
LOOP_EXHAUSTED = "exhausted"
LOOP_HUMAN_VETO = "human_veto"
LOOP_ERROR = "error"


@dataclass
class RunState:
    request: str
    run_id: str = ""

    # Selection
    variant: Optional[str] = None

    # Generation params (judge rewrite/retune mutates these between rounds)
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 768
    height: int = 512
    duration_s: float = 5.0
    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    quality: Optional[str] = None
    stg_scale: Optional[float] = None
    stg_blocks: Optional[list[int]] = None
    sampler_name: Optional[str] = None

    # Input media (ComfyUI input-dir filenames, post-upload)
    video_name: Optional[str] = None
    image_name: Optional[str] = None
    audio_name: Optional[str] = None
    audio_path: Optional[str] = None
    kind: str = ""
    music_bed_attached: bool = False
    has_audio: bool = False

    # Storyboard shot card this run belongs to (multi-segment pipeline)
    shot: Optional[dict] = None

    # Workflow / ComfyUI
    workflow_meta: dict[str, Any] = field(default_factory=dict)
    power_mode: bool = False
    power_meta: dict[str, Any] = field(default_factory=dict)
    prompt_id: Optional[str] = None
    video_path: Optional[str] = None

    # Previs attach (buddy.comfy.attach/v1) — judge rules (c)/(d)
    attach_recipe: Optional[dict[str, Any]] = None
    previs_source: str = ""
    control_pack_present: bool = False
    control_pack_used: dict[str, bool] = field(default_factory=dict)

    # Judge
    judge_enabled: bool = True
    max_judge_rounds: int = 3
    judge_round: int = 0
    attempt: int = 1
    judge_score: float = 0.0
    judge_decision: str = ""
    judge_issues: list[str] = field(default_factory=list)
    judge_reason: str = ""
    judge_history: list[dict[str, Any]] = field(default_factory=list)
    quality_bar: dict[str, Any] = field(default_factory=dict)
    revise_history: list[dict[str, Any]] = field(default_factory=list)
    loop_status: str = LOOP_STARTED
    provenance: dict[str, Any] = field(default_factory=dict)
    provenance_history: list[dict[str, Any]] = field(default_factory=list)
    provenance_sidecar: str = ""

    # Closed loop without Comfy / GPU (tests + --self-improve-dry)
    dry_run: bool = False

    # Control flow
    state: str = "SELECT_VARIANT"
    retries: int = 0
    downscale_level: int = 0
    error: Optional[str] = None
    messages: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.messages.append(msg)
        print(f"[{self.state}] {msg}")

    def transition(self, new_state: str) -> None:
        self.messages.append(f"-> {new_state}")
        self.state = new_state

    def fail(self, error: str) -> None:
        self.error = error
        self.transition("ERROR")
        self.log(f"FAILED: {error}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
