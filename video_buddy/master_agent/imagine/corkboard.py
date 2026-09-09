"""Corkboard: Imagine every storyboard panel in parallel, vision-gate, queue passers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from master_agent.kb.corkboard import save_panels
from master_agent.storyboard.storyboard import ShotCard

GenerateFn = Callable[..., dict[str, Any]]
VisionFn = Callable[..., Optional[dict[str, Any]]]


@dataclass
class PanelRecord:
    index: int
    title: str
    prompt: str
    prompt_version: str
    path: str
    score: float
    passed: bool
    failure_reason: str
    queued: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_generate(prompt: str, dest: Path, **kwargs) -> dict[str, Any]:
    from master_agent.imagine.client import generate_image

    return generate_image(prompt, dest, **kwargs)


def _default_vision(path: Path | str, **kwargs) -> Optional[dict[str, Any]]:
    from master_agent.judge.vision import vision_review

    return vision_review(path, **kwargs)


def run_corkboard(
    shots: list[ShotCard],
    dest_dir: Path,
    *,
    generate_fn: Optional[GenerateFn] = None,
    vision_fn: Optional[VisionFn] = None,
    store_dir: Optional[Path] = None,
    prompt_version: str = "image_prompt:1",
    max_workers: int = 8,
) -> list[PanelRecord]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    gen = generate_fn or _default_generate
    vis = vision_fn or _default_vision

    def _one(shot: ShotCard) -> PanelRecord:
        dest = dest_dir / f"panel_{shot.index:02d}.png"
        prompt = shot.ltx_prompt or shot.visuals or shot.title
        try:
            gen(prompt, dest)
        except Exception as exc:
            rec = PanelRecord(
                index=shot.index,
                title=shot.title,
                prompt=prompt,
                prompt_version=prompt_version,
                path=str(dest),
                score=0.0,
                passed=False,
                failure_reason=str(exc)[:240],
                queued=False,
            )
            return rec
        review = vis(
            dest,
            user_request=shot.title,
            ltx_prompt=prompt,
        ) or {}
        score = float(review.get("score") or 0.0)
        passed = bool(review.get("pass", score >= 0.75))
        reason = ""
        if not passed:
            reason = str(review.get("reason") or "; ".join(review.get("issues") or []) or "vision reject")
        return PanelRecord(
            index=shot.index,
            title=shot.title,
            prompt=prompt,
            prompt_version=prompt_version,
            path=str(dest),
            score=score,
            passed=passed,
            failure_reason=reason,
            queued=passed,
        )

    records: list[PanelRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(shots) or 1))) as pool:
        futures = {pool.submit(_one, shot): shot.index for shot in shots}
        for fut in as_completed(futures):
            records.append(fut.result())
    records.sort(key=lambda r: r.index)
    if store_dir is not None:
        save_panels(Path(store_dir), [r.to_dict() for r in records])
    return records
