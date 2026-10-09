"""PartnerPointerError still writes a run record; OOM duration uses the variant fps."""

from __future__ import annotations

import json
from unittest.mock import patch

from master_agent.config import get_variant_gen
from master_agent.orchestrator.machine import Orchestrator
from master_agent.orchestrator.oom import apply_oom_downscale
from master_agent.orchestrator.state import RunState


def test_partner_pointer_refusal_writes_run_record(tmp_path):
    from master_agent.comfy.partner_pointers import PartnerPointerError

    runs = tmp_path / "runs"

    def refuse(self, st, variant):
        raise PartnerPointerError("api_seedance is not executable")

    with patch("master_agent.orchestrator.machine.RUNS_DIR", runs), patch(
        "master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs"
    ), patch.object(Orchestrator, "_select_variant", refuse):
        st = Orchestrator().run("one take", dry_run=True, judge_enabled=False)
    assert st.state == "ERROR"
    records = list(runs.glob("*.json"))
    assert len(records) == 1
    assert "not executable" in json.dumps(json.loads(records[0].read_text(encoding="utf-8")))


def test_oom_downscale_duration_uses_variant_fps():
    ladder = [(768, 512, 97), (640, 384, 49)]
    st = RunState(request="x", variant="wan22", downscale_level=0)
    assert apply_oom_downscale(st, ladder) is True
    fps = get_variant_gen("wan22")["fps"]
    assert fps == 16
    assert st.duration_s == 49 / fps
