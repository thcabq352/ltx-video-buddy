"""Generate → junk filter → judge → top-K for one Rainey1 recipe.

One seed at a time. ``--dry-run`` returns before any Comfy queue.
Keepers land under ``--out`` with a ``buddy.clip.provenance/v1`` sidecar.

This module does not write ``training/datasets/``. Phase 2 LoRA YAML is out
of scope; see ``docs/RAINEY1_BATCH.md``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from master_agent.config import PROJECT_ROOT
from master_agent.judge.judge import BRIEF_ADHERENCE_LOW
from master_agent.judge.probe import MIN_FRAMES, TINY_FILE_BYTES
from master_agent.provenance import (
    build_clip_provenance,
    missing_required,
    write_clip_provenance,
)
from master_agent.rainey1.recipes import (
    LOOK_FLOOR,
    RaineyRecipe,
    rainey1_rubric_file,
    resolve_recipe,
)

GenerateFn = Callable[..., tuple[Optional[Path], int]]
JudgeFn = Callable[..., "JudgeView"]
ProbeFn = Callable[[Optional[Path]], dict[str, Any]]


@dataclass
class JudgeView:
    look_score: float
    brief_adherence: float | None = None
    fail_reasons: list[str] = field(default_factory=list)
    human_veto: bool = False
    score: float | None = None


@dataclass
class SeedRow:
    seed: int
    recipe_id: str
    look_score: float | None = None
    brief_adherence: float | None = None
    fail_reasons: list[str] = field(default_factory=list)
    path: str = ""
    junk: bool = False
    junk_reason: str = ""
    human_veto: bool = False
    keep: bool = False
    rank: int | None = None
    decision: str = "drop"
    queued: int = 0


@dataclass
class BatchResult:
    recipe_id: str
    slug: str
    variant: str
    width: int
    height: int
    frames: int
    dry_run: bool
    comfy_jobs_queued: int
    rows: list[SeedRow]
    keepers: list[SeedRow]
    out_dir: Path
    exit_code: int
    preset_source: str
    rubric: str
    notes: list[str] = field(default_factory=list)


def parse_seeds(seeds: str | Sequence[int]) -> list[int]:
    if isinstance(seeds, str):
        parts = [part.strip() for part in seeds.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in seeds if str(part).strip()]
    if not parts:
        raise ValueError("at least one --seeds value is required")
    parsed: list[int] = []
    for part in parts:
        try:
            parsed.append(int(part))
        except ValueError as exc:
            raise ValueError(f"seed {part!r} is not an integer") from exc
    return parsed


def is_junk_output(path: Path | None, probe: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Reject missing, ``<~100KB``, or ``<3`` frame outputs before judge spend.

    When ffprobe cannot count frames, size alone decides (same as diagnose).
    """
    info = dict(probe or {})
    if path is None or not Path(path).is_file():
        return True, "missing output"
    size = int(info.get("size_bytes") or 0)
    if size <= 0:
        size = Path(path).stat().st_size
    if size < TINY_FILE_BYTES:
        return True, f"size {size} bytes < {TINY_FILE_BYTES}"
    frames = info.get("frames")
    if frames is not None and int(frames) < MIN_FRAMES:
        return True, f"frames {int(frames)} < {MIN_FRAMES}"
    return False, ""


def _veto(row: SeedRow, *, look_floor: float) -> bool:
    if row.human_veto:
        return True
    if row.look_score is None or row.brief_adherence is None:
        return False
    return row.look_score >= look_floor and float(row.brief_adherence) < BRIEF_ADHERENCE_LOW


def rank_top_k(
    rows: list[SeedRow],
    top_k: int,
    *,
    look_floor: float = LOOK_FLOOR,
) -> list[SeedRow]:
    """Mark top-K keepers. Sort by look_score, then brief_adherence.

    Junk, dry misses, look below ``look_floor``, and human-veto rows drop.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    eligible: list[SeedRow] = []
    for row in rows:
        row.keep = False
        row.rank = None
        if row.junk or row.look_score is None:
            row.decision = "drop"
            continue
        if float(row.look_score) < look_floor:
            row.decision = "drop"
            if f"look {row.look_score:.2f} < {look_floor:.2f}" not in row.fail_reasons:
                row.fail_reasons.append(f"look {row.look_score:.2f} < {look_floor:.2f}")
            continue
        if _veto(row, look_floor=look_floor):
            row.human_veto = True
            row.decision = "drop"
            reason = "human_veto: brief_adherence low"
            if reason not in row.fail_reasons:
                row.fail_reasons.append(reason)
            continue
        eligible.append(row)
    eligible.sort(
        key=lambda row: (
            -float(row.look_score or 0.0),
            -float(row.brief_adherence if row.brief_adherence is not None else -1.0),
            row.seed,
        )
    )
    keepers = eligible[:top_k]
    kept_ids = {id(row) for row in keepers}
    for index, row in enumerate(keepers, start=1):
        row.keep = True
        row.rank = index
        row.decision = "keep"
    for row in rows:
        if id(row) not in kept_ids:
            row.decision = "drop"
            row.keep = False
    return keepers


def _default_probe(path: Path | None) -> dict[str, Any]:
    from master_agent.judge.probe import probe_video

    info = probe_video(path)
    if path is not None and Path(path).is_file() and not info.get("size_bytes"):
        info["size_bytes"] = Path(path).stat().st_size
        info["exists"] = True
    return info


def _default_judge(recipe: RaineyRecipe, seed: int, video_path: Path, *, llm: bool) -> JudgeView:
    """Heuristic judge, plus the text rubric when ``llm`` and a file exist.

    TODO(rainey1-judge): prompt 02's ``rainey1.md`` is appended via
    ``context['rubric_file']`` when present. Otherwise ``judge.md`` is used.
    """
    from master_agent.judge.judge import judge_segment
    from master_agent.judge.probe import analyze

    heuristic, issues = analyze(video_path, expected_duration_s=recipe.duration_s)
    rubric = rainey1_rubric_file()
    context: dict[str, Any] = {
        "recipe_id": recipe.id,
        "seed": seed,
        "duration_s": recipe.duration_s,
    }
    if rubric is not None:
        context["rubric_file"] = str(rubric)
    result = judge_segment(
        user_request=recipe.brief,
        ltx_prompt=recipe.positive,
        video_path=str(video_path),
        heuristic_score=heuristic,
        heuristic_issues=issues,
        judge_enabled=bool(llm),
        threshold=LOOK_FLOOR,
        max_rounds=1,
        context=context,
    )
    reasons = [str(item) for item in result.issues]
    if result.human_veto:
        reasons.append("human_veto: brief_adherence low")
    return JudgeView(
        look_score=float(result.look_score),
        brief_adherence=result.brief_adherence,
        fail_reasons=reasons,
        human_veto=bool(result.human_veto),
        score=float(result.score),
    )


def default_generate(
    recipe: RaineyRecipe,
    seed: int,
    *,
    dry_run: bool,
) -> tuple[Path | None, int]:
    """Queue one Comfy job, or none when ``dry_run`` is set.

    Dry-run returns before importing the orchestrator run path's submit.
    """
    if dry_run:
        return None, 0
    return _generate_live(recipe, seed)


def _generate_live(recipe: RaineyRecipe, seed: int) -> tuple[Path | None, int]:
    """One orchestrator run. OOM retries stay serial inside that run."""
    from master_agent.comfy.client import ComfyClient
    from master_agent.orchestrator.machine import Orchestrator

    queued = {"n": 0}
    original = ComfyClient.queue_prompt

    def _counting(self: ComfyClient, workflow: dict[str, Any]) -> str:
        queued["n"] += 1
        return original(self, workflow)

    ComfyClient.queue_prompt = _counting  # type: ignore[method-assign]
    try:
        state = Orchestrator().run(
            recipe.brief,
            prompt=recipe.positive,
            negative_prompt=recipe.negative,
            variant=recipe.variant,
            seed=seed,
            steps=recipe.steps,
            cfg=recipe.cfg,
            width=recipe.width,
            height=recipe.height,
            frames=recipe.frames,
            duration_s=recipe.duration_s,
            judge_enabled=False,
            revise_enabled=False,
            dry_run=False,
            downscale_level=recipe.ladder_index,
            shot_index=1,
        )
    finally:
        ComfyClient.queue_prompt = original  # type: ignore[method-assign]
    raw = getattr(state, "video_path", None)
    path = Path(raw) if raw else None
    if path is not None and not path.is_file():
        path = None
    return path, int(queued["n"])


def _refuse_dataset_path(path: Path) -> None:
    text = str(path).replace("\\", "/")
    if "/training/datasets/" in text or text.endswith("/training/datasets"):
        raise ValueError(
            "rainey1-batch keepers stay under outputs/rainey1/; "
            "training/datasets is Phase 2"
        )


def _write_keeper_provenance(
    recipe: RaineyRecipe,
    row: SeedRow,
    dest: Path,
) -> None:
    from master_agent.orchestrator.state import RunState

    state = RunState(
        request=recipe.brief,
        prompt=recipe.positive,
        negative_prompt=recipe.negative,
        variant=recipe.variant,
        seed=row.seed,
        steps=recipe.steps,
        cfg=recipe.cfg,
        width=recipe.width,
        height=recipe.height,
        fps=recipe.fps,
        duration_s=recipe.duration_s,
        attempt=1,
        shot_index=1,
        shot_id=f"seed-{row.seed}",
        judge_score=float(row.look_score or 0.0),
        judge_issues=list(row.fail_reasons),
        judge_reason="; ".join(row.fail_reasons),
        video_path=str(dest),
        output_dir=str(dest.parent),
        planned_clip=str(dest),
    )
    payload = build_clip_provenance(state, path=dest, revise_notes="rainey1 top-cut keep")
    payload["prompts"]["additives"] = list(recipe.additives)
    payload["rainey1"] = {
        "recipe_id": recipe.id,
        "seed": row.seed,
        "look_score": row.look_score,
        "brief_adherence": row.brief_adherence,
        "fail_reasons": list(row.fail_reasons),
        "kept": True,
    }
    missing = missing_required(payload)
    if missing:
        raise ValueError(f"keeper provenance missing fields: {missing}")
    write_clip_provenance(dest, payload)


def _copy_keepers(recipe: RaineyRecipe, keepers: list[SeedRow], out_dir: Path) -> None:
    _refuse_dataset_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in keepers:
        src = Path(row.path)
        dest = out_dir / f"seed-{row.seed}.mp4"
        shutil.copy2(src, dest)
        row.path = str(dest)
        _write_keeper_provenance(recipe, row, dest)


def run_rainey1_batch(
    *,
    recipe: str,
    seeds: str | Sequence[int],
    top_k: int = 2,
    out: str | Path | None = None,
    dry_run: bool = False,
    allow_empty: bool = False,
    llm_judge: bool = True,
    generate_fn: GenerateFn | None = None,
    judge_fn: JudgeFn | None = None,
    probe_fn: ProbeFn | None = None,
) -> BatchResult:
    """Run the loop. Dry-run queues zero Comfy jobs and writes no keepers."""
    spec = resolve_recipe(recipe)
    seed_list = parse_seeds(seeds)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if out is None:
        out_dir = PROJECT_ROOT / "outputs" / "rainey1" / "topcut" / spec.slug
    else:
        out_dir = Path(out)
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
    _refuse_dataset_path(out_dir)

    rubric_path = rainey1_rubric_file()
    notes = [spec.notes] if spec.notes else []
    if spec.source != "preset":
        notes.append(
            "TODO(rainey1-preset): prompt 01 is not merged; using inline recipe constants."
        )
    if rubric_path is None:
        notes.append(
            "TODO(rainey1-judge): judge/prompts/rainey1.md is missing; using the default judge."
        )
    notes.append(
        "Phase 2 train YAML stays out of this command; keepers land under outputs/rainey1/."
    )

    produce = generate_fn or default_generate
    score = judge_fn or (lambda rec, seed, path: _default_judge(rec, seed, path, llm=llm_judge))
    probe = probe_fn or _default_probe

    rows: list[SeedRow] = []
    queued_total = 0
    # One seed per iteration. Live generate waits for that Comfy job before the next.
    for seed in seed_list:
        row = SeedRow(seed=seed, recipe_id=spec.id)
        try:
            path, queued = produce(spec, seed, dry_run=dry_run)
        except Exception as exc:
            path, queued = None, 0
            row.fail_reasons.append(f"generate failed: {exc}")
        queued_total += int(queued or 0)
        row.queued = int(queued or 0)
        if dry_run:
            row.decision = "drop"
            row.fail_reasons.append("dry-run: not queued")
            rows.append(row)
            continue
        clip = Path(path) if path else None
        info = probe(clip)
        junk, why = is_junk_output(clip, info)
        if junk:
            row.junk = True
            row.junk_reason = why
            row.fail_reasons.append(why)
            row.path = str(clip) if clip else ""
            rows.append(row)
            continue
        assert clip is not None
        row.path = str(clip)
        try:
            view = score(spec, seed, clip)
        except Exception as exc:
            row.junk = True
            row.junk_reason = f"judge failed: {exc}"
            row.fail_reasons.append(row.junk_reason)
            rows.append(row)
            continue
        row.look_score = float(view.look_score)
        row.brief_adherence = view.brief_adherence
        row.human_veto = bool(view.human_veto)
        for reason in view.fail_reasons:
            if reason not in row.fail_reasons:
                row.fail_reasons.append(reason)
        rows.append(row)

    if dry_run:
        keepers: list[SeedRow] = []
        exit_code = 0
    else:
        keepers = rank_top_k(rows, top_k, look_floor=LOOK_FLOOR)
        if keepers:
            _copy_keepers(spec, keepers, out_dir)
        exit_code = 0 if keepers or allow_empty else 1

    return BatchResult(
        recipe_id=spec.id,
        slug=spec.slug,
        variant=spec.variant,
        width=spec.width,
        height=spec.height,
        frames=spec.frames,
        dry_run=dry_run,
        comfy_jobs_queued=queued_total,
        rows=rows,
        keepers=keepers,
        out_dir=out_dir,
        exit_code=exit_code,
        preset_source=spec.source,
        rubric="rainey1" if rubric_path is not None else "default",
        notes=notes,
    )


def format_batch_report(result: BatchResult) -> str:
    lines = [
        (
            f"recipe={result.recipe_id} source={result.preset_source} "
            f"rubric={result.rubric} variant={result.variant} "
            f"size={result.width}x{result.height} frames={result.frames}"
        ),
        (
            f"dry_run={int(result.dry_run)} comfy_jobs_queued={result.comfy_jobs_queued} "
            f"keepers={len(result.keepers)} out={result.out_dir}"
        ),
    ]
    for note in result.notes:
        if note:
            lines.append(f"note: {note}")
    lines.append("seed\tlook\tbrief\tkeep/drop\tpath")
    for row in result.rows:
        look = "-" if row.look_score is None else f"{row.look_score:.2f}"
        brief = "-" if row.brief_adherence is None else f"{row.brief_adherence:.2f}"
        lines.append(
            f"{row.seed}\t{look}\t{brief}\t{row.decision}\t{row.path or '-'}"
        )
    if result.dry_run:
        lines.append("dry-run: queued zero Comfy jobs; no keepers written")
    elif result.exit_code:
        lines.append("error: zero keepers (pass --allow-empty to exit 0)")
    return "\n".join(lines)
