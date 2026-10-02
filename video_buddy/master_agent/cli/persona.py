"""Persona, soul, character, and LoRA commands."""

from __future__ import annotations

import argparse
import json
import sys

from master_agent.cli.runtime import comfy_client

def cmd_persona(args: argparse.Namespace) -> int:
    import master_agent.config as cfg
    from master_agent.persona.persona import list_personas, load_persona, set_active_persona

    if args.persona_command == "list":
        for p in list_personas():
            mark = "*" if p.slug == cfg.PERSONA else " "
            print(f"{mark} {p.slug:10s} {p.name:10s} {p.path}")
        print("\n* = active. Switch with: python -m master_agent persona set <slug>")
        print("  Add your own: state/personas/<name>.md")
        return 0
    if args.persona_command == "set":
        if not args.name:
            print("FAIL  pass a persona slug: persona set exec")
            return 2
        try:
            persona = set_active_persona(args.name, session="cli")
        except ValueError as e:
            print(f"FAIL  {e}")
            return 1
        print(f"OK    persona={persona.slug} ({persona.name})")
        return 0
    try:
        persona = load_persona(args.name or None)
    except ValueError as e:
        print(f"FAIL  {e}")
        return 1
    print(persona.system_prompt)
    return 0


def cmd_soul(args: argparse.Namespace) -> int:
    import master_agent.config as cfg
    from master_agent.persona.soul import list_souls, load_soul, set_active_soul

    if args.soul_command == "list":
        for s in list_souls():
            mark = "*" if s.slug == cfg.SOUL else " "
            print(f"{mark} {s.slug:10s} {s.name:10s} {s.path}")
        print("\n* = active. Switch with: python -m master_agent soul set <slug>")
        print("  Add your own: state/souls/<name>.md")
        return 0
    if args.soul_command == "set":
        if not args.name:
            print("FAIL  pass a soul slug: soul set play")
            return 2
        try:
            soul = set_active_soul(args.name, session="cli")
        except ValueError as e:
            print(f"FAIL  {e}")
            return 1
        print(f"OK    soul={soul.slug} ({soul.name})")
        return 0
    try:
        soul = load_soul(args.name or None)
    except ValueError as e:
        print(f"FAIL  {e}")
        return 1
    print(soul.system_prompt)
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

    client = comfy_client()
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

    client = comfy_client()
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

