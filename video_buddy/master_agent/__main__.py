"""CLI entry: python -m master_agent <command>

Commands:
  health              Check ComfyUI reachability + GPU stats
  fetch-object-info   Fetch /object_info from ComfyUI and cache it to state/
  scan-models         Scan models/ into state/model_inventory.json and print summary
  validate [file|-a]  Validate workflow JSON against /object_info + model inventory
  run                 Orchestrated video generation (storyboard -> judge -> stitch)
  fractal             Procedural fractal deep-zoom video (CPU only, no ComfyUI)
  music               Beat-synced music video from an audio track (ffmpeg mux)
  mv plan|render      Comfy/LTX → Remotion MTV mode (beat plan, unique burns)
  heartmula           generate lyrics+tags → wav, or transcribe → words.json
  persona list|show|set   Intake voice (default: ara)
  soul list|show|set      Standing values (default: studio)
  brief               Interview-only: rough idea -> creative brief (--go to generate)
  character create|list   CCC stage: bible -> Flux sheet -> captioned dataset
  lora setup|train|validate  Flux LoRA training via ai-toolkit + vision validation
  download-flux       One-time Flux fp8 weights download (~17GB)
  inventory           List discovered weights (paths + roles) before doctor
  download-models     Scan packs; --scan-only / --use-existing / --download; --yes to fetch
  models manifest     Print the capability-grouped selector catalog
  models select       LTX 2.3|2.5 checklist, totals, disk gate, Soundtrack Studio
  automatic-install   Automatic install: pre-flight, then ComfyUI + LTX 2.3 + llama.cpp (--yes)
  setup | doctor      Scan deps + weights (fetch rules: repo docs/WEIGHTS.md)
  workflows           List default catalog variants (no env flags)
  comfy run           Drive ComfyUI from the CLI (prepare + lint + queue)
  comfy ingest        Store an API workflow under state/ingested/<slug>/ (no queue)
  comfy learn         Refresh and print the learned field map for a slug
  comfy dry-run       Print the patched map; do not queue
  comfy run --ingested SLUG
                      Queue a learned graph (vae_guard on). Not a catalog variant.
  comfy attach        Apply previs buddy.comfy.attach/v1 (dry-run; --submit to /prompt)
  comfy start|stop|status|restart
                      Managed local ComfyUI via comfy-cli (not attach / not generate)
  comfy update        Report stale pins; update only with --yes, snapshot first
  diagnose            9-frame hull fire (sec/step); does not spend shift budget
  rainey1-batch       Rainey1 seeds → junk filter → judge → top-K (.buddy.json)
  budget              status | reset-shift  (VRAM-min shift ledger)
  hermes              status | register  (profile ltx + discovery)
  capabilities        Gap matrix: tower-ish Comfy nodes vs Buddy wiring
  curriculum          Print LESSON_BUDDY_WORKS_HERE (L0→L5) and Part 2 gate
  about               Print the studio identity card

Command bodies live in master_agent.cli. This module is the dispatcher
and the patch point for ComfyClient.
"""

from __future__ import annotations

import sys

from master_agent.comfy.client import ComfyClient, ComfyClientError
from master_agent.cli.common import _music_intent
from master_agent.cli.comfy import (
    cmd_capabilities,
    cmd_comfy,
    cmd_comfy_attach,
    cmd_diagnose,
    cmd_fetch_object_info,
    cmd_power_tune,
    cmd_validate,
)
from master_agent.cli.doctor import cmd_health, cmd_inventory, cmd_scan_models, cmd_setup
from master_agent.cli.fractal import cmd_fractal
from master_agent.cli.models import (
    cmd_download_flux,
    cmd_download_models,
    cmd_models_manifest,
    cmd_models_select,
    cmd_workflows,
)
from master_agent.cli.music import cmd_heartmula, cmd_music, cmd_mv
from master_agent.cli.parser import main
from master_agent.cli.persona import cmd_character, cmd_lora, cmd_persona, cmd_soul
from master_agent.cli.run import cmd_brief, cmd_run
from master_agent.cli.studio import (
    cmd_about,
    cmd_budget,
    cmd_curriculum,
    cmd_hermes,
    cmd_kb,
    cmd_ui,
)

__all__ = [
    "ComfyClient",
    "ComfyClientError",
    "_music_intent",
    "cmd_about",
    "cmd_brief",
    "cmd_budget",
    "cmd_capabilities",
    "cmd_character",
    "cmd_comfy",
    "cmd_comfy_attach",
    "cmd_curriculum",
    "cmd_diagnose",
    "cmd_download_flux",
    "cmd_download_models",
    "cmd_fetch_object_info",
    "cmd_fractal",
    "cmd_health",
    "cmd_heartmula",
    "cmd_hermes",
    "cmd_inventory",
    "cmd_kb",
    "cmd_lora",
    "cmd_models_manifest",
    "cmd_models_select",
    "cmd_music",
    "cmd_mv",
    "cmd_persona",
    "cmd_power_tune",
    "cmd_run",
    "cmd_scan_models",
    "cmd_setup",
    "cmd_soul",
    "cmd_ui",
    "cmd_validate",
    "cmd_workflows",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
