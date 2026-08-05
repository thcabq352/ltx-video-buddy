"""CLI entry: python -m master_agent <command>

Commands:
  health              Check ComfyUI reachability + GPU stats
  fetch-object-info   Fetch /object_info from ComfyUI and cache it to state/
  scan-models         Scan models/ into state/model_inventory.json and print summary
  validate [file|-a]  Validate workflow JSON against /object_info + model inventory
  run                 Orchestrated video generation (storyboard -> judge -> stitch)
  fractal             Procedural fractal deep-zoom video (CPU only, no ComfyUI)
  music               Beat-synced music video from an audio track
  persona list|show   Intake personas (default: ara)
  brief               Interview-only: rough idea -> creative brief (--go to generate)
  character create|list   CCC stage: bible -> Flux sheet -> captioned dataset
  lora setup|train|validate  Flux LoRA training via ai-toolkit + vision validation
  download-flux       One-time Flux fp8 weights download (~17GB)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from master_agent.comfy.client import ComfyClient, ComfyClientError
from master_agent.comfy.validator import format_report, validate_workflow_file
from master_agent.config import WORKFLOWS_DIR, ensure_dirs
from master_agent.models.inventory import format_summary, scan_inventory


def cmd_health(args: argparse.Namespace) -> int:
    client = ComfyClient()
    try:
        stats = client.health()
    except ComfyClientError as e:
        print(f"FAIL  {e}")
        return 1
    system = stats.get("system") or {}
    devices = stats.get("devices") or []
    print(f"OK    ComfyUI at {client.base_url}")
    print(f"      os={system.get('os')} python={system.get('python_version')}")
    for dev in devices:
        name = dev.get("name") or "?"
        vram_total = (dev.get("vram_total") or 0) / 1e9
        vram_free = (dev.get("vram_free") or 0) / 1e9
        print(f"      gpu={name} vram={vram_free:.1f}G free / {vram_total:.1f}G")
    return 0


def cmd_fetch_object_info(args: argparse.Namespace) -> int:
    client = ComfyClient()
    try:
        info = client.refresh_object_info_cache()
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    print(f"OK    cached {len(info)} node classes to state/object_info.json")
    return 0


def cmd_scan_models(args: argparse.Namespace) -> int:
    inv = scan_inventory()
    print(format_summary(inv))
    bad = [v for v, i in inv.bundles.items() if not i.get("runnable")]
    return 1 if bad and args.strict else 0


def cmd_validate(args: argparse.Namespace) -> int:
    targets: list[Path] = []
    if args.all:
        if not WORKFLOWS_DIR.is_dir():
            print(f"FAIL  workflows dir not found: {WORKFLOWS_DIR}")
            return 1
        targets = sorted(WORKFLOWS_DIR.glob("*.json"))
        if not targets:
            print(f"FAIL  no *.json in {WORKFLOWS_DIR}")
            return 1
    elif args.file:
        targets = [Path(args.file)]
    else:
        print("FAIL  pass a workflow file or --all")
        return 2

    client = ComfyClient()
    reports = []
    for path in targets:
        if not path.is_file():
            print(f"FAIL  {path} not found")
            return 1
        try:
            report = validate_workflow_file(
                path, client=client, prefer_live=not args.offline
            )
        except Exception as e:
            print(f"FAIL  {path}: {e}")
            return 1
        reports.append(report)

    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=1))
    else:
        for report in reports:
            print(format_report(report))
            print()
    return 0 if all(r.ok for r in reports) else 1


def cmd_run(args: argparse.Namespace) -> int:
    from master_agent.config import (
        JUDGE_ENABLED,
        MAX_FULL_JUDGE_ROUNDS,
        MAX_JUDGE_ROUNDS,
    )
    from master_agent.orchestrator.pipeline import dry_run_pipeline, run_pipeline

    client = ComfyClient()
    if not client.is_up():
        print(f"FAIL  ComfyUI is not reachable. Start: ComfyUI_windows_portable\\run_api_8188.bat")
        return 1

    args.request = _maybe_interview(args.request, no_interview=args.no_interview)

    if args.dry_run:
        return dry_run_pipeline(
            args.request,
            variant=args.variant,
            duration_s=args.duration,
            quality=args.quality,
            width=args.width,
            height=args.height,
            storyboard_mode=args.storyboard,
            llm_panel=args.llm_panel,
            panel_judge=args.panel_judge,
            client=client,
        )

    # Auto-route music videos to the beat-synced pipeline (explicit --variant wins)
    if args.audio and not args.variant and _music_intent(args.request, args.audio, args.quality):
        from master_agent.music.pipeline import run_music_video

        print("auto-route: music video detected -> beat-synced music pipeline")
        rec = run_music_video(
            args.request,
            args.audio,
            visual="shots",
            quality=args.quality,
            seed=args.seed,
            width=args.width,
            height=args.height,
            judge_enabled=False if args.no_judge else JUDGE_ENABLED,
            max_judge_rounds=args.max_judge_rounds or MAX_JUDGE_ROUNDS,
            llm_panel=args.llm_panel,
            panel_judge=args.panel_judge,
            upscale=args.upscale,
            client=client,
        )
        print()
        if rec.get("status") in ("done", "done_with_warnings"):
            print(f"{rec['status'].upper()}   video: {rec.get('video_path')}")
            return 0 if rec["status"] == "done" else 2
        print(f"ERROR  {rec.get('error')}")
        return 1

    # Upload input media into ComfyUI's input dir first
    video_name = image_name = audio_name = None
    try:
        if args.video:
            video_name = client.upload_image(Path(args.video))
            print(f"uploaded video: {video_name}")
        if args.image:
            image_name = client.upload_image(Path(args.image))
            print(f"uploaded image: {image_name}")
        if args.audio:
            audio_name = client.upload_audio(Path(args.audio))
            print(f"uploaded audio: {audio_name}")
    except Exception as e:
        print(f"FAIL  upload: {e}")
        return 1

    result = run_pipeline(
        args.request,
        variant=args.variant,
        duration_s=args.duration,
        quality=args.quality,
        seed=args.seed,
        width=args.width,
        height=args.height,
        video_name=video_name,
        image_name=image_name,
        audio_name=audio_name,
        storyboard_mode=args.storyboard,
        judge_enabled=False if args.no_judge else JUDGE_ENABLED,
        max_judge_rounds=args.max_judge_rounds or MAX_JUDGE_ROUNDS,
        max_full_judge_rounds=args.max_full_judge_rounds or MAX_FULL_JUDGE_ROUNDS,
        llm_panel=args.llm_panel,
        panel_judge=args.panel_judge,
        client=client,
    )
    print()
    if result.status in ("done", "done_with_warnings"):
        print(f"{result.status.upper()}   video: {result.video_path}")
        print(f"       segments: {len(result.segment_paths)} scores={[round(s, 2) for s in result.segment_scores]}")
        print(f"       full judge: score={result.full_judge_score:.2f} pass={result.full_judge_pass}")
        if result.full_judge_notes:
            print(f"       notes: {result.full_judge_notes}")
        return 0 if result.status == "done" else 2
    print(f"ERROR  {result.error}")
    return 1


def _music_intent(request: str, audio_path: str, quality: str | None) -> bool:
    """Auto-routing rule: music keywords, or audio longer than one segment."""
    from master_agent.config import get_quality_profile
    from master_agent.music.beats import audio_duration
    from master_agent.music.pipeline import MUSIC_KEYWORDS

    text = request.lower()
    if any(kw in text for kw in MUSIC_KEYWORDS):
        return True
    duration = audio_duration(audio_path)
    cap = float(get_quality_profile(quality)["segment_max_s"])
    return duration > cap


def _maybe_interview(request: str, *, no_interview: bool = False) -> str:
    """Pre-generation intake interview (default on for interactive sessions)."""
    from master_agent.config import INTAKE_ENABLED

    if no_interview or not INTAKE_ENABLED or not request.strip():
        return request
    if not sys.stdin.isatty():
        return request
    from master_agent.persona.intake import run_interview_cli

    brief = run_interview_cli(request)
    return brief.to_request()


def cmd_persona(args: argparse.Namespace) -> int:
    from master_agent.config import PERSONA
    from master_agent.persona.persona import list_personas, load_persona

    if args.persona_command == "list":
        for p in list_personas():
            mark = "*" if p.slug == PERSONA else " "
            print(f"{mark} {p.slug:10s} {p.name:10s} {p.path}")
        print(f"\n* = active (PERSONA env, default 'ara'). Add your own: state/personas/<name>.md")
        return 0
    # show
    try:
        persona = load_persona(args.name)
    except ValueError as e:
        print(f"FAIL  {e}")
        return 1
    print(persona.system_prompt)
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    from master_agent.persona.intake import run_interview_cli

    brief = run_interview_cli(args.request)
    if not args.go:
        return 0

    client = ComfyClient()
    if not client.is_up():
        print("FAIL  ComfyUI is not reachable. Start: ComfyUI_windows_portable\\run_api_8188.bat")
        return 1
    from master_agent.orchestrator.pipeline import run_pipeline

    duration = brief.duration_s or args.duration
    print(f"generating: {duration}s @ {args.quality or 'default'} quality")
    result = run_pipeline(
        brief.to_request(),
        duration_s=duration,
        quality=args.quality,
        client=client,
    )
    print()
    if result.status in ("done", "done_with_warnings"):
        print(f"{result.status.upper()}   video: {result.video_path}")
        return 0 if result.status == "done" else 2
    print(f"ERROR  {result.error}")
    return 1


def cmd_fractal(args: argparse.Namespace) -> int:
    from master_agent.config import FRACTAL_DEFAULTS
    from master_agent.fractal.pipeline import run_fractal

    request = _maybe_interview(args.request or "", no_interview=args.no_interview)
    rec = run_fractal(
        request,
        duration_s=args.duration,
        fps=args.fps or FRACTAL_DEFAULTS["fps"],
        width=args.width,
        height=args.height,
        target=args.target,
        palette=args.palette,
        seed=args.seed,
        julia=args.julia,
        audio_path=args.audio,
    )
    video = rec["video_path"]
    if args.upscale:
        try:
            from master_agent.upscale import upscale_video

            video = str(upscale_video(video, method=args.upscale, run_id=rec["run_id"]))
            print(f"upscaled: {video}")
        except Exception as e:
            print(f"WARN  upscale failed: {e}")
    print(f"DONE   video: {video}")
    return 0


def cmd_music(args: argparse.Namespace) -> int:
    from master_agent.config import JUDGE_ENABLED, MAX_JUDGE_ROUNDS
    from master_agent.music.pipeline import run_music_video

    if not args.audio:
        print("FAIL  music needs --audio <file>")
        return 2
    args.request = _maybe_interview(args.request, no_interview=args.no_interview)
    client = ComfyClient()
    if args.visual == "shots" and not client.is_up():
        print("FAIL  ComfyUI is not reachable. Start: ComfyUI_windows_portable\\run_api_8188.bat")
        print("      (or use --visual fractal — CPU only, no ComfyUI needed)")
        return 1
    rec = run_music_video(
        args.request,
        args.audio,
        visual=args.visual,
        variant=args.variant,
        quality=args.quality,
        seed=args.seed,
        width=args.width,
        height=args.height,
        judge_enabled=False if args.no_judge else JUDGE_ENABLED,
        max_judge_rounds=args.max_judge_rounds or MAX_JUDGE_ROUNDS,
        llm_panel=args.llm_panel,
        panel_judge=args.panel_judge,
        upscale=args.upscale,
        client=client,
    )
    print()
    if rec.get("status") in ("done", "done_with_warnings"):
        print(f"{rec['status'].upper()}   video: {rec.get('video_path')}")
        bmap = rec.get("beat_map") or {}
        print(f"       {bmap.get('bpm')} BPM, {len(rec.get('shot_windows') or [])} shots")
        if rec.get("full_judge_score") is not None:
            print(f"       full judge: score={rec.get('full_judge_score', 0):.2f} pass={rec.get('full_judge_pass')}")
        return 0 if rec["status"] == "done" else 2
    print(f"ERROR  {rec.get('error')}")
    return 1


def cmd_kb(args: argparse.Namespace) -> int:
    from master_agent.kb.ingest import ingest_all_runs, ingest_workflows
    from master_agent.kb.store import (
        COLLECTION_RUNS,
        COLLECTION_WORKFLOWS,
        collection_count,
        kb_available,
        search,
    )

    if not kb_available():
        print("FAIL  knowledge base unavailable (KB_ENABLED=0 or chromadb missing)")
        return 1

    if args.kb_command == "ingest":
        n_wf = ingest_workflows()
        n_runs = ingest_all_runs()
        print(f"OK    ingested {n_wf} workflow(s), {n_runs} run record(s)")
        print(f"      workflows={collection_count(COLLECTION_WORKFLOWS)} runs={collection_count(COLLECTION_RUNS)}")
        return 0

    if args.kb_command == "search":
        if not args.query:
            print("FAIL  pass a query: kb search \"coffee ad\"")
            return 2
        coll = COLLECTION_WORKFLOWS if args.workflows else COLLECTION_RUNS
        hits = search(coll, args.query, k=args.k)
        if not hits:
            print("no hits")
            return 0
        for h in hits:
            m = h.get("metadata") or {}
            dist = h.get("distance")
            print(f"- [{dist:.3f}] {h.get('id')} {m.get('request') or m.get('name') or ''}")
            first = (h.get("text") or "").splitlines()[0] if h.get("text") else ""
            print(f"    {first[:160]}")
        return 0

    if args.kb_command == "stats":
        print(f"workflows={collection_count(COLLECTION_WORKFLOWS)} runs={collection_count(COLLECTION_RUNS)}")
        return 0

    return 2


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("FAIL  uvicorn not installed (pip install uvicorn fastapi)")
        return 1
    from master_agent.web.app import app

    print(f"Master Agent studio: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_download_flux(args: argparse.Namespace) -> int:
    from master_agent.models.download import download_flux_weights

    try:
        paths = download_flux_weights()
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    print(f"OK    {len(paths)} flux weight file(s) ready")
    return 0


def cmd_character(args: argparse.Namespace) -> int:
    from master_agent.config import CHARACTERS_DIR

    if args.character_command == "list":
        if not CHARACTERS_DIR.is_dir():
            print("no characters yet")
            return 0
        rows = []
        for char_dir in sorted(CHARACTERS_DIR.iterdir()):
            cj = char_dir / "character.json"
            if not cj.is_file():
                continue
            try:
                data = json.loads(cj.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(data)
        if not rows:
            print("no characters yet")
            return 0
        for data in rows:
            print(
                f"- {data.get('name')}  trigger={data.get('trigger_word')}  "
                f"dataset={data.get('dataset_count')}  created={data.get('created', '')[:10]}"
            )
        return 0

    # create
    if not args.description:
        print('FAIL  pass a description: character create "a grizzled dwarven smith"')
        return 2
    from master_agent.character import (
        build_dataset,
        generate_character_sheet,
        make_character_bible,
    )

    client = ComfyClient()
    if not client.is_up():
        print("FAIL  ComfyUI is not reachable. Start: ComfyUI_windows_portable\\run_api_8188.bat")
        return 1

    print(f"bible: {args.description!r} ...")
    bible = make_character_bible(args.description)
    if args.name:
        bible.name = args.name
    print(f"       name={bible.name} trigger={bible.trigger_word} shots={len(bible.shots)}")

    sheet = generate_character_sheet(
        bible, client=client, shots=args.shots, vision=not args.no_vision
    )
    kept = sheet.get("kept") or []
    print(f"sheet: kept {len(kept)}/{len(sheet.get('scores') or kept)} hero={sheet.get('hero')}")

    dataset = build_dataset(bible.name)
    if not dataset.get("ok"):
        print(f"WARN  dataset: {dataset.get('reason')} ({dataset.get('dataset_count')} images)")
    else:
        print(f"dataset: {dataset.get('dataset_count')} images -> {dataset.get('dataset_dir')}")

    if not args.train:
        print(f"next:   python -m master_agent lora train {bible.name}")
        return 0 if dataset.get("ok") else 2
    if not dataset.get("ok"):
        print("FAIL  dataset too small to train")
        return 1
    return _train_and_validate_loop(bible.name)


def _train_and_validate_loop(name: str, start_attempt: int = 1) -> int:
    from master_agent.config import LORA_MAX_ATTEMPTS
    from master_agent.lora import train_lora, validate_lora

    overrides = None
    for attempt in range(start_attempt, LORA_MAX_ATTEMPTS + 1):
        print(f"train:  attempt {attempt}/{LORA_MAX_ATTEMPTS} overrides={overrides or 'defaults'}")
        result = train_lora(name, overrides=overrides)
        if not result.get("ok"):
            print(f"FAIL  training: {result.get('error')}")
            return 1
        print(f"train:  done -> {result.get('lora_path')}")
        val = validate_lora(name, attempt=attempt)
        print(f"judge:  score={val.get('score', 0):.2f} pass={val.get('pass')}")
        if val.get("pass"):
            print(f"OK    lora validated: {result.get('lora_path')}")
            return 0
        overrides = val.get("next_overrides") or {}
        if not overrides:
            print("FAIL  retry budget exhausted, lora did not pass validation")
            return 1
    return 1


def cmd_lora(args: argparse.Namespace) -> int:
    from master_agent.config import AI_TOOLKIT_DIR

    if args.lora_command == "setup":
        return _lora_setup(AI_TOOLKIT_DIR)

    if not args.name:
        print(f"FAIL  pass a character name: lora {args.lora_command} <name>")
        return 2

    if args.lora_command == "train":
        overrides = {
            k: v
            for k, v in (("steps", args.steps), ("lr", args.lr), ("rank", args.rank))
            if v
        }
        result = train_lora_impl(args.name, overrides or None, resume=not args.no_resume)
        if not result.get("ok"):
            print(f"FAIL  {result.get('error')}")
            return 1
        print(f"OK    lora: {result.get('lora_path')} log: {result.get('log')}")
        if args.validate:
            val = validate_lora_impl(args.name)
            print(f"judge: score={val.get('score', 0):.2f} pass={val.get('pass')}")
            return 0 if val.get("pass") else 2
        return 0

    if args.lora_command == "validate":
        val = validate_lora_impl(args.name, attempt=args.attempt)
        if val.get("error"):
            print(f"FAIL  {val['error']}")
            return 1
        print(f"score={val.get('score', 0):.2f} pass={val.get('pass')}")
        if val.get("next_overrides"):
            print(f"next overrides: {val['next_overrides']}")
        return 0 if val.get("pass") else 2

    return 2


def train_lora_impl(name, overrides, resume=True):
    from master_agent.lora import train_lora

    client = ComfyClient()
    return train_lora(name, overrides=overrides, resume=resume, client=client)


def validate_lora_impl(name, attempt=1):
    from master_agent.lora import validate_lora

    return validate_lora(name, attempt=attempt)


def _lora_setup(toolkit_dir) -> int:
    """Clone Ostris ai-toolkit and create its private venv (one-time, long)."""
    import subprocess
    import sys

    if not (toolkit_dir / "run.py").is_file():
        if toolkit_dir.exists() and any(toolkit_dir.iterdir()):
            print(f"FAIL  {toolkit_dir} exists but has no run.py — fix or remove it")
            return 1
        print(f"cloning ai-toolkit -> {toolkit_dir}")
        rc = subprocess.call(
            ["git", "clone", "https://github.com/ostris/ai-toolkit", str(toolkit_dir)]
        )
        if rc != 0:
            print("FAIL  git clone failed")
            return 1
    venv_dir = toolkit_dir / "venv"
    venv_py = venv_dir / "Scripts" / "python.exe"
    if not venv_py.is_file():
        print(f"creating venv -> {venv_dir}")
        rc = subprocess.call([sys.executable, "-m", "venv", str(venv_dir)])
        if rc != 0:
            print("FAIL  venv creation failed")
            return 1
    reqs = toolkit_dir / "requirements.txt"
    steps = [
        [str(venv_py), "-m", "pip", "install", "--upgrade", "pip"],
        # torch cu130 wheels per ai-toolkit README
        [
            str(venv_py), "-m", "pip", "install",
            "torch==2.13.0", "torchvision==0.28.0", "torchaudio==2.11.0",
            "--index-url", "https://download.pytorch.org/whl/cu130",
        ],
    ]
    if reqs.is_file():
        steps.append([str(venv_py), "-m", "pip", "install", "-r", str(reqs)])
    for cmd in steps:
        print(f"$ {' '.join(cmd[:6])} ...")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"FAIL  setup step failed (rc={rc}); re-run `lora setup` to resume")
            return 1
    print("OK    ai-toolkit ready. Train with: python -m master_agent lora train <character>")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows console is cp1252 — never crash on LLM-emitted unicode (e.g. →)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    ensure_dirs()
    parser = argparse.ArgumentParser(prog="master_agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("health", help="check ComfyUI reachability")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("fetch-object-info", help="cache /object_info to state/")
    p.set_defaults(func=cmd_fetch_object_info)

    p = sub.add_parser("scan-models", help="scan models/ into inventory")
    p.add_argument("--strict", action="store_true", help="exit 1 if any bundle is not runnable")
    p.set_defaults(func=cmd_scan_models)

    p = sub.add_parser("validate", help="validate workflow JSON")
    p.add_argument("file", nargs="?", help="workflow JSON path")
    p.add_argument("--all", "-a", action="store_true", help="validate all workflows/*.json")
    p.add_argument("--offline", action="store_true", help="use cached object_info only")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("kb", help="knowledge base: ingest | search | stats")
    p.add_argument("kb_command", choices=["ingest", "search", "stats"])
    p.add_argument("query", nargs="?", default="", help="search query")
    p.add_argument("-k", type=int, default=3, help="max hits (default 3)")
    p.add_argument("--workflows", action="store_true", help="search workflows instead of runs")
    p.set_defaults(func=cmd_kb)

    p = sub.add_parser("ui", help="web dashboard (FastAPI) on 127.0.0.1:8189")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8189)
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("run", help="orchestrated generation: patch -> validate -> submit -> judge")
    p.add_argument("request", help="what to generate (natural language)")
    p.add_argument("--variant", choices=["base", "eros", "directors", "lipsync", "wan22"], help="force variant")
    p.add_argument("--duration", type=float, default=5.0, help="seconds (default 5)")
    p.add_argument("--quality", choices=["draft", "balanced", "quality"], help="quality profile")
    p.add_argument("--seed", type=int, help="fixed seed (default: random)")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--video", help="source video file (lipsync); uploaded to ComfyUI first")
    p.add_argument("--image", help="source image file (i2v); uploaded to ComfyUI first")
    p.add_argument("--audio", help="source audio file; uploaded to ComfyUI first")
    p.add_argument("--no-judge", action="store_true", help="skip the judge loop")
    p.add_argument("--max-judge-rounds", type=int, default=None, help="judge retry budget")
    p.add_argument("--storyboard", choices=["smart", "always", "multi_only", "off"],
                   help="storyboard mode (default: env STORYBOARD_MODE or smart)")
    p.add_argument("--llm-panel",
                   help="storyboard LLM panel: preset (default | duo) or comma list, "
                        "e.g. ollama:qwen3.6-27b-fable,ollama:gemma4:latest")
    p.add_argument("--panel-judge",
                   help="provider that picks the winning storyboard (default: env PANEL_JUDGE or ollama)")
    p.add_argument("--max-full-judge-rounds", type=int, default=None,
                   help="full-video judge re-gen budget (multi-segment)")
    p.add_argument("--dry-run", action="store_true",
                   help="storyboard + patch + validate all segments, no GPU queue")
    p.add_argument("--upscale", choices=["rtx", "seedvr2"], default=None,
                   help="post-stage upscale of the final video")
    p.add_argument("--no-interview", action="store_true",
                   help="skip the persona intake interview")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("persona", help="personas: list | show <name>")
    p.add_argument("persona_command", choices=["list", "show"])
    p.add_argument("name", nargs="?", default="", help="persona slug (show)")
    p.set_defaults(func=cmd_persona)

    p = sub.add_parser("brief", help="interview-only: turn a rough idea into a creative brief")
    p.add_argument("request", help="your rough idea (natural language)")
    p.add_argument("--go", action="store_true", help="generate immediately after the interview")
    p.add_argument("--duration", type=float, default=8.0, help="seconds if the brief doesn't say (--go)")
    p.add_argument("--quality", choices=["draft", "balanced", "quality"], help="quality profile (--go)")
    p.set_defaults(func=cmd_brief)

    p = sub.add_parser("fractal", help="procedural fractal deep-zoom video (CPU only)")
    p.add_argument("request", nargs="?", default="", help="optional brief (for the run record)")
    p.add_argument("--duration", type=float, default=20.0, help="seconds (default 20)")
    p.add_argument("--fps", type=int, default=None, help="frames per second (default 24)")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--target", default="seahorse",
                   choices=["seahorse", "elephant", "minibrot", "spiral"],
                   help="zoom target (default seahorse)")
    p.add_argument("--palette", default="fire",
                   choices=["fire", "ocean", "monochrome", "neon", "sunset"])
    p.add_argument("--julia", action="store_true", help="Julia-set mode (c wobbles with the beat)")
    p.add_argument("--seed", type=int, help="fixed seed (default: random)")
    p.add_argument("--audio", help="audio track: beat-reactive render + mux")
    p.add_argument("--upscale", choices=["rtx", "seedvr2"], default=None,
                   help="post-stage upscale of the final video")
    p.add_argument("--no-interview", action="store_true",
                   help="skip the persona intake interview")
    p.set_defaults(func=cmd_fractal)

    p = sub.add_parser("music", help="beat-synced music video from an audio track")
    p.add_argument("request", help="creative brief (natural language)")
    p.add_argument("--audio", required=True, help="audio file (mp3/wav/...)")
    p.add_argument("--visual", choices=["shots", "fractal"], default="shots",
                   help="shots = ComfyUI generation per shot; fractal = CPU beat-reactive zoom")
    p.add_argument("--variant", choices=["base", "eros", "directors", "lipsync", "wan22"],
                   help="force generation variant (shots mode)")
    p.add_argument("--quality", choices=["draft", "balanced", "quality"], help="quality profile")
    p.add_argument("--seed", type=int, help="fixed seed (default: random)")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--no-judge", action="store_true", help="skip the judge loop")
    p.add_argument("--max-judge-rounds", type=int, default=None, help="judge retry budget")
    p.add_argument("--llm-panel",
                   help="storyboard LLM panel: preset (default | duo) or comma list")
    p.add_argument("--panel-judge",
                   help="provider that picks the winning storyboard")
    p.add_argument("--upscale", choices=["rtx", "seedvr2"], default=None,
                   help="post-stage upscale of the final video")
    p.add_argument("--no-interview", action="store_true",
                   help="skip the persona intake interview")
    p.set_defaults(func=cmd_music)

    p = sub.add_parser("download-flux", help="download Flux fp8 weights (~17GB, one-time)")
    p.set_defaults(func=cmd_download_flux)

    p = sub.add_parser("character", help="CCC stage: create | list")
    p.add_argument("character_command", choices=["create", "list"])
    p.add_argument("description", nargs="?", default="", help="character description")
    p.add_argument("--name", help="override the LLM-chosen character name")
    p.add_argument("--shots", type=int, default=None, help="cap sheet shots (default: all)")
    p.add_argument("--no-vision", action="store_true", help="skip vision identity curation")
    p.add_argument("--train", action="store_true", help="chain into LoRA train+validate loop")
    p.set_defaults(func=cmd_character)

    p = sub.add_parser("lora", help="Flux LoRA stage: setup | train | validate")
    p.add_argument("lora_command", choices=["setup", "train", "validate"])
    p.add_argument("name", nargs="?", default="", help="character name")
    p.add_argument("--steps", type=int, default=0, help="override training steps")
    p.add_argument("--lr", type=float, default=0.0, help="override learning rate")
    p.add_argument("--rank", type=int, default=0, help="override lora rank")
    p.add_argument("--attempt", type=int, default=1, help="attempt number (retry ladder)")
    p.add_argument("--no-resume", action="store_true", help="start training from scratch")
    p.add_argument("--validate", action="store_true", help="validate after training")
    p.set_defaults(func=cmd_lora)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
