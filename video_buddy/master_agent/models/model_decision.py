"""Route, then GGUF: one explained choice.

The route is Flux I2V, LTX I2V (the user already has a still), MiniMax H3,
or plain LTX text-to-video. The model step then scores on-disk GGUFs for
that route. Callers get the winner, a reason for every loser, and the
pairwise lines Scott asked for (Sulphur vs base, Eros vs 10Eros, Q4 vs Q3,
dev vs distilled, MiniMax H3 vs LTX).

``workflows/flux_ltx_i2v_api.json`` is used when that file is on disk
(it arrived with PR #92). This module only detects it. It does not copy
a graph from another branch.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

_T2V_LOCK = re.compile(
    r"text-to-video|text to video|\bt2v\b|no flux|without flux",
    re.IGNORECASE,
)
_H3_NAMED = re.compile(r"\b(?:minimax|hailuo|h3_r2v|h3_t2v|h3_i2v|h3)\b", re.IGNORECASE)
_TALKING = re.compile(
    r"lip-?sync|lipsync|talking|speaks|spoken|says the line|\bvoice\b|hailuo|minimax",
    re.IGNORECASE,
)
_HOSTED_ENV = ("MINIMAX_API_KEY", "HAILUO_API_KEY", "H3_API_KEY", "H3_API_BASE")
_PLAIN_T2V = frozenset(
    {"base", "ltx25_t2v_i2v", "ltx23_t2v_base", "ltx23_t2v_distilled", "h3_t2v"}
)
_FLUX_VARIANT = "flux_ltx_i2v"
_FLUX_FILENAME = "flux_ltx_i2v_api.json"

# Higher tier is a better quant. Q4_K_M holds more than Q4_K_S, which holds more than Q3.
_TIER = {"Q4_K_M": 4, "Q4_K_S": 3, "Q5": 3, "Q3_K_S": 2, "other": 1}


@dataclass(frozen=True)
class Candidate:
    filename: str
    family: str
    kind: str
    tier: int
    quant: str
    size_gb: float
    vram_gb: float
    tags: tuple[str, ...]
    note: str
    aliases: tuple[str, ...] = ()

    def names(self) -> tuple[str, ...]:
        return (self.filename, *self.aliases)


def _c(
    filename: str,
    *,
    family: str,
    kind: str,
    quant: str,
    size_gb: float,
    vram_gb: float,
    tags: tuple[str, ...],
    note: str,
    aliases: tuple[str, ...] = (),
) -> Candidate:
    return Candidate(
        filename=filename,
        family=family,
        kind=kind,
        tier=_TIER.get(quant, 1),
        quant=quant,
        size_gb=size_gb,
        vram_gb=vram_gb,
        tags=tags,
        note=note,
        aliases=aliases,
    )


# Sizes are approximate published weights. VRAM is the resident figure used
# by the 16GB policy rows, not a measured trace.
CATALOG: tuple[Candidate, ...] = (
    _c(
        "LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf",
        family="ltx23",
        kind="distilled",
        quant="Q4_K_S",
        size_gb=16.7,
        vram_gb=9.5,
        tags=("general", "fast", "distilled", "base", "q4"),
        note="QuantStack distilled 1.1 Q4_K_S. General/fast base preset.",
    ),
    _c(
        "ltx-2.3-22b-distilled-Q3_K_S.gguf",
        family="ltx23",
        kind="distilled",
        quant="Q3_K_S",
        size_gb=12.5,
        vram_gb=7.5,
        tags=("general", "fast", "distilled", "base", "q3"),
        note="Distilled Q3_K_S. Same intent as the QuantStack Q4, lower tier.",
        aliases=("LTX-2.3-distilled-Q3_K_S.gguf",),
    ),
    _c(
        "sulphur_dev-Q3_K_S.gguf",
        family="ltx23",
        kind="dev",
        quant="Q3_K_S",
        size_gb=12.0,
        vram_gb=8.0,
        tags=("dev", "sulphur", "q3"),
        note="Sulphur dev Q3_K_S. A dev finetune, not the base distilled unet.",
    ),
    _c(
        "LTX-2.3-dev-Q4_K_S.gguf",
        family="ltx23",
        kind="dev",
        quant="Q4_K_S",
        size_gb=16.0,
        vram_gb=13.5,
        tags=("dev", "directors", "q4"),
        note="Dev Q4_K_S. Directors quality preset.",
    ),
    _c(
        "10Eros_v1.5-Q4_K_M.gguf",
        family="ltx23",
        kind="style",
        quant="Q4_K_M",
        size_gb=13.3,
        vram_gb=11.0,
        tags=("eros", "style", "q4"),
        note="10Eros v1.5 Q4_K_M. The eros-style GGUF. Not the general distilled unet.",
    ),
    _c(
        "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf",
        family="ltx25",
        kind="distilled",
        quant="Q4_K_M",
        size_gb=14.6,
        vram_gb=12.5,
        tags=("general", "distilled", "ltx25", "q4"),
        note="LTX 2.5 distilled GGUF Q4_K_M.",
    ),
    _c(
        "minimax_h3_fl2va_pruned-Q4_K.gguf",
        family="h3",
        kind="distilled",
        quant="Q4_K_S",
        size_gb=11.0,
        vram_gb=13.0,
        tags=("h3", "talking", "local", "q4"),
        note="MiniMax H3 fl2va Q4_K. Local Comfy, text-to-AV and image-to-AV.",
    ),
    _c(
        "minimax_h3_ref2va_pruned-Q4_K.gguf",
        family="h3",
        kind="distilled",
        quant="Q4_K_S",
        size_gb=11.0,
        vram_gb=13.4,
        tags=("h3", "talking", "lipsync", "local", "q4"),
        note="MiniMax H3 ref2va Q4_K. Local talking / coarse lip-sync from a voice sample.",
    ),
    _c(
        "flux-2-klein-4b-fp8.safetensors",
        family="flux",
        kind="distilled",
        quant="other",
        size_gb=4.0,
        vram_gb=8.0,
        tags=("flux", "klein", "fast"),
        note="FLUX.2 Klein 4B fp8. The only Flux download target.",
    ),
    _c(
        "flux-2-klein-4b.safetensors",
        family="flux",
        kind="distilled",
        quant="other",
        size_gb=7.2,
        vram_gb=12.0,
        tags=("flux", "klein"),
        note="FLUX.2 Klein 4B bf16. Accepted when already on disk. Not a download target.",
    ),
)

_BY_NAME: dict[str, Candidate] = {}
for _row in CATALOG:
    for _name in _row.names():
        _BY_NAME[_name.lower()] = _row

_DEFAULT_INTENT = {
    "base": "general_fast_distilled",
    "eros": "eros_style",
    "directors": "dev_quality",
    "ltx25": "general_fast_distilled",
    "h3": "talking",
    "h3_ref": "talking",
}
_DEFAULT_PREFER = {
    "base": (
        "LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf",
        "ltx-2.3-22b-distilled-Q3_K_S.gguf",
        "LTX-2.3-distilled-Q3_K_S.gguf",
        "sulphur_dev-Q3_K_S.gguf",
    ),
    "eros": ("10Eros_v1.5-Q4_K_M.gguf",),
    "directors": (
        "LTX-2.3-dev-Q4_K_S.gguf",
        "sulphur_dev-Q3_K_S.gguf",
    ),
}


@dataclass
class PresetPolicy:
    variant: str
    intent: str
    prefer: tuple[str, ...]
    model: str | None = None


@dataclass
class Rejection:
    name: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "reason": self.reason}


@dataclass
class ModelDecision:
    route: str
    variant: str
    score_variant: str
    apply_route: bool
    chosen: str | None
    chosen_path: Path | None
    family: str
    why_route: str
    why_model: str
    rejected: list[Rejection] = field(default_factory=list)
    comparisons: list[str] = field(default_factory=list)
    warning: str | None = None
    error: str | None = None
    vram_gb: float = 16.0
    vram_source: str = "declared"
    input_image: bool = False
    flux_installed: bool = False
    flux_model: str | None = None
    flux_detail: str = ""
    together_gb: float | None = None
    h3_mode: str = "absent"
    text: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "variant": self.variant,
            "score_variant": self.score_variant,
            "apply_route": self.apply_route,
            "chosen": self.chosen,
            "chosen_path": str(self.chosen_path) if self.chosen_path else None,
            "family": self.family,
            "why_route": self.why_route,
            "why_model": self.why_model,
            "rejected": [row.to_dict() for row in self.rejected],
            "comparisons": list(self.comparisons),
            "warning": self.warning,
            "error": self.error,
            "vram_gb": self.vram_gb,
            "vram_source": self.vram_source,
            "input_image": self.input_image,
            "flux_installed": self.flux_installed,
            "flux_model": self.flux_model,
            "flux_detail": self.flux_detail,
            "together_gb": self.together_gb,
            "h3_mode": self.h3_mode,
            "text": self.text,
        }


class ModelSelectionError(RuntimeError):
    """No GGUF fits. The message lists what is missing and what would have fit."""


_VRAM_CACHE: tuple[float, str] | None = None


def clear_vram_cache() -> None:
    global _VRAM_CACHE
    _VRAM_CACHE = None


def vram_budget(explicit: float | None = None) -> tuple[float, str]:
    """Free-or-declared gigabytes and a short source label.

    ``explicit`` wins (tests). Then ``VRAM_FREE_GB``. Then ``nvidia-smi``
    memory.free. Then declared ``config.VRAM_GB`` (env, total memory, or 16).
    """
    if explicit is not None:
        return float(explicit), "argument"
    raw = (os.getenv("VRAM_FREE_GB") or "").strip()
    if raw:
        try:
            return float(raw), "VRAM_FREE_GB"
        except ValueError:
            pass
    global _VRAM_CACHE
    if _VRAM_CACHE is not None:
        return _VRAM_CACHE
    probed = _probe_free_vram_gb()
    if probed is not None:
        _VRAM_CACHE = (probed, "nvidia-smi memory.free")
        return _VRAM_CACHE
    from master_agent.config import VRAM_GB, VRAM_SOURCE

    _VRAM_CACHE = (float(VRAM_GB), f"declared {VRAM_SOURCE or 'default'}")
    return _VRAM_CACHE


def _probe_free_vram_gb() -> float | None:
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    try:
        mib = float(line[0].split(",")[0].strip())
    except ValueError:
        return None
    if mib <= 0:
        return None
    return round(mib / 1024.0, 1)


def h3_opt_in() -> bool:
    return (os.getenv("H3_OPT_IN") or "").strip().lower() in {"1", "true", "yes", "on"}


def hosted_h3_env() -> list[str]:
    return [name for name in _HOSTED_ENV if (os.getenv(name) or "").strip()]


def preset_policy(variant: str) -> PresetPolicy:
    key = _preset_key(variant)
    intent = _DEFAULT_INTENT.get(key, "general_fast_distilled")
    prefer = _DEFAULT_PREFER.get(key, ())
    if not prefer and "flux" in key:
        prefer = (
            "flux-2-klein-4b-fp8.safetensors",
            "flux-2-klein-4b.safetensors",
        )
    model: str | None = None
    try:
        import yaml

        from master_agent.config import WORKFLOWS_DIR

        path = WORKFLOWS_DIR / "ltx23_av_presets.yaml"
        if path.is_file() and key in {"base", "eros", "directors"}:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            row = data.get(key) if isinstance(data, dict) else None
            if isinstance(row, dict):
                raw_intent = row.get("intent")
                if isinstance(raw_intent, str) and raw_intent.strip():
                    intent = raw_intent.strip()
                raw_model = row.get("model")
                if isinstance(raw_model, str) and raw_model.strip():
                    model = raw_model.strip()
                raw_prefer = row.get("prefer")
                if isinstance(raw_prefer, list):
                    prefer = tuple(str(item) for item in raw_prefer if str(item).strip())
    except Exception:
        pass
    env_model = (os.getenv("LTX_GGUF_MODEL") or "").strip()
    if env_model:
        model = env_model
    return PresetPolicy(variant=key, intent=intent, prefer=prefer, model=model)


def _preset_key(variant: str) -> str:
    key = (variant or "base").strip().lower()
    if key in {"base", "eros", "directors"}:
        return key
    if key.startswith("ltx25") or "ltx-2.5" in key or "ltx2.5" in key:
        return "ltx25"
    if "ref2va" in key or key in {"h3_r2v", "ref2va", "h3_ref2va"}:
        return "h3_ref"
    if key.startswith("h3") or "minimax" in key or "fl2va" in key:
        return "h3"
    return key


def _family_for(variant: str) -> str:
    key = _preset_key(variant)
    if key in {"base", "eros", "directors"}:
        return "ltx23"
    if key == "ltx25":
        return "ltx25"
    if key in {"h3", "h3_ref"}:
        return "h3"
    if "flux" in key:
        return "flux"
    return "ltx23"


def _intent_score(candidate: Candidate, intent: str) -> int:
    tags = set(candidate.tags)
    if intent == "eros_style":
        if "eros" in tags:
            return 100
        if "distilled" in tags and "q4" in tags:
            return 45
        if "distilled" in tags:
            return 35
        if "dev" in tags and "sulphur" not in tags:
            return 25
        return 20
    if intent == "dev_quality":
        if "directors" in tags and "q4" in tags:
            return 100
        if "sulphur" in tags:
            return 70
        if "dev" in tags:
            return 80
        if "distilled" in tags and "q4" in tags:
            return 40
        if "distilled" in tags:
            return 35
        return 20
    if intent == "talking":
        if "lipsync" in tags:
            return 100
        if "h3" in tags or "talking" in tags:
            return 90
        if "distilled" in tags:
            return 30
        return 10
    # general_fast_distilled
    if "eros" in tags:
        return 15
    if "distilled" in tags and "q4" in tags and "dev" not in tags:
        return 100
    if "distilled" in tags and "q3" in tags:
        return 75
    if "sulphur" in tags:
        return 40
    if "dev" in tags and "q4" in tags:
        return 30
    if "distilled" in tags:
        return 50
    return 10


def _prefer_index(name: str, prefer: Sequence[str]) -> int:
    low = name.lower()
    for index, item in enumerate(prefer):
        if item.lower() == low:
            return index
    return 100


def combined_vram_gb(flux_gb: float, video_gb: float) -> float:
    """One graph holds both weights. Peak is the larger plus 40% of the smaller."""
    big, small = (flux_gb, video_gb) if flux_gb >= video_gb else (video_gb, flux_gb)
    return round(big + 0.4 * small, 1)


def _is_klein_4b_diffusion(name: str) -> bool:
    """Klein 4B diffusion safetensors only. Other Flux names are not selectable."""
    low = Path(name).name.lower()
    return "flux-2-klein-4b" in low and low.endswith(".safetensors") and "lora" not in low


def _locate_all(roots: Sequence[Path]) -> dict[str, Path]:
    from master_agent.models.weights import (
        _search_name,
        is_ltx25_quant_gguf,
        is_sulphur_or_ltx23_gguf,
    )

    found: dict[str, Path] = {}
    for candidate in CATALOG:
        for name in candidate.names():
            hit = _search_name(name, roots)
            if hit is not None:
                found[name.lower()] = hit
                found[hit.name.lower()] = hit
    folders = ("diffusion_models", "unet", "checkpoints", "gguf")
    for root in roots:
        if not root.is_dir():
            continue
        parents = [root, *[root / name for name in folders]]
        parents.append(root / "diffusion_models" / "gguf")
        for folder in parents:
            if not folder.is_dir():
                continue
            try:
                children = list(folder.glob("*.gguf"))
                children.extend(folder.glob("*.safetensors"))
            except OSError:
                continue
            for path in children:
                if not path.is_file() or path.stat().st_size <= 0:
                    continue
                low = path.name.lower()
                if low in found:
                    continue
                interesting = (
                    is_sulphur_or_ltx23_gguf(path.name)
                    or is_ltx25_quant_gguf(path.name)
                    or "minimax_h3" in low
                    or _is_klein_4b_diffusion(path.name)
                )
                if interesting:
                    found[low] = path
    return found


def _candidate_for_path(path: Path) -> Candidate:
    known = _BY_NAME.get(path.name.lower())
    if known is not None:
        return known
    low = path.name.lower()
    quant = "other"
    if "q4_k_m" in low:
        quant = "Q4_K_M"
    elif "q4" in low:
        quant = "Q4_K_S"
    elif "q3" in low:
        quant = "Q3_K_S"
    elif "q5" in low:
        quant = "Q5"
    family = "ltx23"
    tags: list[str] = []
    kind = "distilled"
    if "ltx-2.5" in low or "ltx2.5" in low or "ltx25" in low:
        family = "ltx25"
        tags.extend(["general", "distilled", "ltx25"])
    elif "minimax" in low or low.startswith("h3"):
        family = "h3"
        tags.extend(["h3", "talking", "local"])
    elif _is_klein_4b_diffusion(path.name):
        family = "flux"
        tags.extend(("flux", "klein"))
    elif "flux" in low or "klein" in low:
        family = "refused"
    else:
        if "sulphur" in low:
            tags.extend(["sulphur", "dev"])
            kind = "dev"
        if "10eros" in low or "eros" in low:
            tags.extend(["eros", "style"])
            kind = "style"
        if "dev" in low and "distill" not in low:
            tags.append("dev")
            kind = "dev"
        if "distill" in low:
            tags.extend(["distilled", "general", "fast"])
            kind = "distilled"
    if quant.startswith("Q4"):
        tags.append("q4")
    elif quant.startswith("Q3"):
        tags.append("q3")
    vram = 9.5
    if kind == "style":
        vram = 11.0
    elif kind == "dev" and quant.startswith("Q4"):
        vram = 13.5
    elif quant.startswith("Q3"):
        vram = 7.5 if kind == "distilled" else 8.0
    if family == "ltx25":
        vram = 12.5
    if family == "h3":
        vram = 13.4 if "ref" in low else 13.0
    return Candidate(
        filename=path.name,
        family=family,
        kind=kind,
        tier=_TIER.get(quant, 1),
        quant=quant,
        size_gb=0.0,
        vram_gb=vram,
        tags=tuple(dict.fromkeys(tags)),
        note=f"Found on disk ({path.name}). Not in the built-in catalog.",
    )


def _family_candidates(family: str, h3_ref: bool) -> list[Candidate]:
    rows = [row for row in CATALOG if row.family == family]
    if family == "h3":
        if h3_ref:
            return [row for row in rows if "lipsync" in row.tags] or rows
        return [row for row in rows if "lipsync" not in row.tags] or rows
    if family == "flux":
        return rows
    return rows


def select_model(
    variant: str = "base",
    roots: Iterable[Path] | None = None,
    *,
    vram_gb: float | None = None,
    override: str | None = None,
) -> ModelDecision:
    """Score on-disk GGUFs for ``variant``. Does not log and does not raise."""
    if roots is None:
        from master_agent.models.weights import model_search_roots

        search = list(model_search_roots())
    else:
        search = [Path(path) for path in roots]
    budget, source = vram_budget(vram_gb)
    policy = preset_policy(variant)
    if override:
        policy = PresetPolicy(
            variant=policy.variant,
            intent=policy.intent,
            prefer=policy.prefer,
            model=override,
        )
    family = _family_for(variant)
    h3_ref = _preset_key(variant) == "h3_ref"
    located = _locate_all(search)
    pool = _family_candidates(family, h3_ref)
    present_rows: list[tuple[Candidate, Path]] = []
    seen_files: set[str] = set()
    for candidate in pool:
        path = None
        for name in candidate.names():
            path = located.get(name.lower())
            if path is not None:
                break
        if path is None or path.name.lower() in seen_files:
            continue
        seen_files.add(path.name.lower())
        present_rows.append((candidate, path))
    for path in located.values():
        if path.name.lower() in seen_files:
            continue
        guessed = _candidate_for_path(path)
        if guessed.family != family:
            continue
        if family == "h3" and h3_ref and "lipsync" not in guessed.tags and "ref" not in path.name.lower():
            continue
        seen_files.add(path.name.lower())
        present_rows.append((guessed, path))

    rejected: list[Rejection] = []
    for candidate in pool:
        if any(candidate.filename == row.filename for row, _path in present_rows):
            continue
        if any(alias.lower() in located for alias in (n.lower() for n in candidate.names())):
            continue
        rejected.append(Rejection(candidate.filename, "not on disk"))

    warning: str | None = None
    error: str | None = None
    chosen: Candidate | None = None
    chosen_path: Path | None = None

    if policy.model:
        want = policy.model
        path = located.get(Path(want).name.lower())
        match = _BY_NAME.get(Path(want).name.lower())
        if path is None:
            error = (
                f"Model override {want} is not on disk. "
                + _missing_sentence(pool, budget)
            )
        else:
            chosen_path = path
            chosen = match or _candidate_for_path(path)
            if chosen.vram_gb > budget + 0.05:
                warning = (
                    f"Override {path.name} wins, but it needs about {chosen.vram_gb:.1f} GB "
                    f"and the budget is {budget:.1f} GB ({source})."
                )
            for row, row_path in present_rows:
                if row_path.name == path.name:
                    continue
                rejected.append(
                    Rejection(row_path.name, "override set; this candidate was not requested")
                )
    else:
        eligible: list[tuple[int, int, int, str, Candidate, Path]] = []
        for row, path in present_rows:
            if row.vram_gb > budget + 0.05:
                rejected.append(
                    Rejection(
                        path.name,
                        f"exceeds VRAM (needs about {row.vram_gb:.1f} GB, budget {budget:.1f} GB)",
                    )
                )
                continue
            score = _intent_score(row, policy.intent)
            eligible.append(
                (
                    score,
                    row.tier,
                    -_prefer_index(path.name, policy.prefer),
                    path.name.lower(),
                    row,
                    path,
                )
            )
        if not eligible:
            if family == "ltx23" and _preset_key(variant) in {"base", "eros", "directors"}:
                error = _nothing_fits(pool, present_rows, budget, source)
            elif not present_rows and family in {"ltx25", "h3"}:
                error = None
        else:
            eligible.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
            _score, _tier, _pref, _name, chosen, chosen_path = eligible[0]
            for row, path in present_rows:
                if chosen_path is not None and path.name == chosen_path.name:
                    continue
                if any(rej.name == path.name for rej in rejected):
                    continue
                score = _intent_score(row, policy.intent)
                winner_score = _intent_score(chosen, policy.intent) if chosen else 0
                if row.tier < (chosen.tier if chosen else 0) and score <= winner_score:
                    rejected.append(Rejection(path.name, "lower tier"))
                elif score < winner_score:
                    rejected.append(
                        Rejection(path.name, f"intent mismatch for {policy.intent}")
                    )
                else:
                    rejected.append(Rejection(path.name, "lower preference order"))

    why_model = _why_model(policy, chosen, chosen_path, budget, source, warning, error)
    comparisons = _comparisons(
        intent=policy.intent,
        chosen=chosen_path.name if chosen_path else None,
        present={path.name for _row, path in present_rows},
        rejected=rejected,
        h3_mode=_h3_mode(located),
    )
    decision = ModelDecision(
        route="pending",
        variant=variant,
        score_variant=variant,
        apply_route=False,
        chosen=chosen_path.name if chosen_path else None,
        chosen_path=chosen_path,
        family=family,
        why_route="",
        why_model=why_model,
        rejected=rejected,
        comparisons=comparisons,
        warning=warning,
        error=error,
        vram_gb=budget,
        vram_source=source,
        h3_mode=_h3_mode(located),
    )
    decision.text = render_decision(decision)
    return decision


def _h3_mode(located: dict[str, Path]) -> str:
    local = any("minimax_h3" in name and name.endswith(".gguf") for name in located)
    if local:
        return "local"
    if hosted_h3_env():
        return "hosted"
    return "absent"


def _missing_sentence(pool: Sequence[Candidate], budget: float) -> str:
    fitting = [row.filename for row in pool if row.vram_gb <= budget + 0.05]
    heavy = [row.filename for row in pool if row.vram_gb > budget + 0.05]
    parts = ["Nothing suitable is on disk."]
    if fitting:
        parts.append("These would fit the VRAM budget: " + ", ".join(fitting) + ".")
    if heavy:
        parts.append("These would exceed it: " + ", ".join(heavy) + ".")
    return " ".join(parts)


def _nothing_fits(
    pool: Sequence[Candidate],
    present: list[tuple[Candidate, Path]],
    budget: float,
    source: str,
) -> str:
    if not present:
        return (
            f"No LTX GGUF is on disk (budget {budget:.1f} GB via {source}). "
            + _missing_sentence(pool, budget)
        )
    names = ", ".join(
        f"{path.name} (needs about {row.vram_gb:.1f} GB)" for row, path in present
    )
    fitting = [row.filename for row in pool if row.vram_gb <= budget + 0.05]
    extra = ""
    if fitting:
        extra = " A file that would fit if it were on disk: " + ", ".join(fitting) + "."
    return (
        f"No LTX GGUF fits {budget:.1f} GB ({source}). On disk but rejected: {names}.{extra}"
    )


def _why_model(
    policy: PresetPolicy,
    chosen: Candidate | None,
    path: Path | None,
    budget: float,
    source: str,
    warning: str | None,
    error: str | None,
) -> str:
    if error:
        return error
    if chosen is None or path is None:
        return (
            f"No {policy.variant} GGUF was selected "
            f"(budget {budget:.1f} GB via {source}). The loader ladder may still use a non-GGUF pack."
        )
    base = (
        f"Chose {path.name}: {chosen.note} "
        f"Intent {policy.intent}. Quality tier {chosen.quant}. "
        f"About {chosen.size_gb:.1f} GB on disk, about {chosen.vram_gb:.1f} GB VRAM. "
        f"Budget {budget:.1f} GB via {source}."
    )
    if warning:
        return base + " " + warning
    return base


def _comparisons(
    *,
    intent: str,
    chosen: str | None,
    present: set[str],
    rejected: list[Rejection],
    h3_mode: str,
) -> list[str]:
    reasons = {row.name: row.reason for row in rejected}
    q4 = "LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf"
    q3 = "ltx-2.3-22b-distilled-Q3_K_S.gguf"
    sulphur = "sulphur_dev-Q3_K_S.gguf"
    dev = "LTX-2.3-dev-Q4_K_S.gguf"
    eros = "10Eros_v1.5-Q4_K_M.gguf"

    def _has(name: str) -> bool:
        return name in present or any(name.lower() == item.lower() for item in present)

    q4_reason = reasons.get(q4, "")
    if chosen == sulphur and (not _has(q4) or "not on disk" in q4_reason):
        sulphur_line = (
            "Sulphur vs base: Sulphur wins over the base distilled Q4 because "
            f"{q4} is not on disk. Sulphur Q3 is the on-disk stand-in."
        )
    elif chosen == sulphur and "exceeds VRAM" in q4_reason:
        sulphur_line = (
            "Sulphur vs base: Sulphur wins over the base distilled Q4 because "
            "that Q4 exceeds VRAM and Sulphur Q3 fits."
        )
    elif chosen == q4:
        sulphur_line = (
            "Sulphur vs base: the base distilled Q4 wins over Sulphur. "
            "Sulphur is a dev Q3 finetune; the base intent is general/fast distilled, "
            "and Q4 beats Q3 when the Q4 file fits."
        )
    else:
        sulphur_line = (
            "Sulphur vs base: the base preset wants distilled Q4 "
            f"({q4}). Sulphur ({sulphur}) is a dev Q3, so it loses when the distilled file is present and fits. "
            f"Chosen: {chosen or 'none'}."
        )

    if intent == "eros_style" and chosen == eros:
        eros_line = (
            "Eros vs 10Eros: the eros preset picks 10Eros v1.5 Q4_K_M. "
            "That file is the eros style. A generic distilled unet is not Eros, "
            "and an older Eros safetensors bake is not a diffusion GGUF."
        )
    elif chosen and chosen != eros and (_has(eros) or "not on disk" not in reasons.get(eros, "not on disk")):
        eros_line = (
            "Eros vs 10Eros: 10Eros v1.5 loses here. "
            "The eros style tag does not match this preset, so 10Eros is an intent mismatch. "
            "Eros the preset and 10Eros the file are the same choice only when the preset is eros."
        )
    else:
        eros_line = (
            "Eros vs 10Eros: 10Eros v1.5 Q4_K_M is the eros-style file. "
            "It wins for the eros preset and loses for base, where a general distilled Q4 "
            "is a better intent match than the 10Eros style finetune."
        )

    if chosen and "Q3" in chosen and any("exceeds VRAM" in reasons.get(name, "") and "Q4" in name for name in reasons):
        q_line = (
            "Q4 vs Q3: Q3 wins this run because the Q4 candidate exceeds VRAM. "
            "Q4 is the higher tier and wins whenever it is on disk and fits."
        )
    elif chosen and ("Q4" in chosen or "q4" in chosen.lower()):
        q_line = (
            "Q4 vs Q3: Q4 wins over Q3. Q4 keeps more of the weight. "
            "Q3 is only chosen when Q4 is not on disk or Q4 exceeds VRAM."
        )
    else:
        q_line = (
            "Q4 vs Q3: Q4 is the higher tier. Q3 is the VRAM downgrade "
            "or the stand-in when no Q4 file is on disk."
        )

    if intent == "dev_quality":
        dev_line = (
            "dev vs distilled: this preset is dev quality, so dev Q4 "
            f"({dev}) beats the base distilled Q4 when it is on disk and fits. "
            "Distilled is the general/fast model, not the directors model."
        )
    else:
        dev_line = (
            "dev vs distilled: this preset wants distilled, not dev. "
            f"Distilled Q4 beats dev Q4 ({dev}) on intent. "
            "Directors is the preset that flips that comparison."
        )

    if h3_mode == "hosted":
        h3_line = (
            "MiniMax H3 vs LTX: a paid hosted MiniMax H3 key is set "
            f"({', '.join(hosted_h3_env())}). Paid hosted MiniMax H3 is not auto-selected. "
            "Set H3_OPT_IN=1 or name MiniMax, Hailuo, or H3. A hosted call has a cost. "
            "This repo has no hosted client, so the key does not place a call. "
            "Talking and lip-sync stay on local LTX unless local H3 weights are on disk."
        )
    elif h3_mode == "local":
        h3_line = (
            "MiniMax H3 vs LTX: MiniMax H3 weights are on disk (local Comfy, no per-call charge). "
            "H3 wins a talking or lip-sync brief when it is named or H3_OPT_IN=1. "
            "It loses a generic text-to-video to LTX. Paid hosted MiniMax H3 is not auto-selected."
        )
    else:
        h3_line = (
            "MiniMax H3 vs LTX: MiniMax H3 is not available (no local GGUF, no hosted key). "
            "It is the talking and lip-sync candidate when local weights exist. "
            "Paid hosted MiniMax H3 is not auto-selected without H3_OPT_IN. "
            "A hosted call would have a cost, and this repo has no hosted client."
        )
    return [sulphur_line, eros_line, q_line, dev_line, h3_line]


def flux_i2v_workflow() -> Path | None:
    """The hybrid graph if it is actually on disk. Never copies PR #92."""
    from master_agent.config import WORKFLOWS_DIR

    direct = WORKFLOWS_DIR / _FLUX_FILENAME
    if direct.is_file():
        return direct
    manifest = WORKFLOWS_DIR / "manifests.yaml"
    if not manifest.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for meta in data.values():
        if not isinstance(meta, dict):
            continue
        file_name = str(meta.get("file") or "")
        low = file_name.lower()
        if "flux" in low and "i2v" in low:
            path = WORKFLOWS_DIR / file_name
            if path.is_file():
                return path
    return None


def _first_flux(located: dict[str, Path]) -> tuple[Candidate, Path] | None:
    for row in CATALOG:
        if row.family != "flux":
            continue
        for name in row.names():
            path = located.get(name.lower())
            if path is not None:
                return row, path
    return None


def _animator(located: dict[str, Path]) -> tuple[str, Candidate, Path] | None:
    """LTX file that would animate a Flux still. 2.5 GGUF, else 2.3 distilled."""
    order = (
        ("ltx25_t2v_i2v", "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf"),
        ("base", "LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf"),
        ("base", "ltx-2.3-22b-distilled-Q3_K_S.gguf"),
        ("base", "sulphur_dev-Q3_K_S.gguf"),
    )
    for variant, name in order:
        path = located.get(name.lower())
        row = _BY_NAME.get(name.lower())
        if path is not None and row is not None:
            return variant, row, path
    return None


def decide_generation(
    variant: str,
    *,
    request: str = "",
    has_image: bool = False,
    has_audio: bool = False,
    forced: bool = False,
    roots: Iterable[Path] | None = None,
    vram_gb: float | None = None,
    override: str | None = None,
    allowed: set[str] | None = None,
    flux_workflow: Path | None = None,
) -> ModelDecision:
    """Pick the route, then the model. Logs the text. Does not raise."""
    if roots is None:
        from master_agent.models.weights import model_search_roots

        search = list(model_search_roots())
    else:
        search = [Path(path) for path in roots]
    budget, source = vram_budget(vram_gb)
    located = _locate_all(search)
    if flux_workflow is not None:
        workflow = flux_workflow if Path(flux_workflow).is_file() else None
    else:
        workflow = flux_i2v_workflow()
    flux_hit = _first_flux(located)
    animator = _animator(located)
    installed = workflow is not None
    flux_model = flux_hit[1].name if flux_hit else None
    together = None
    fits = False
    if flux_hit and animator:
        together = combined_vram_gb(flux_hit[0].vram_gb, animator[1].vram_gb)
        fits = together <= budget + 0.05
    h3_mode = _h3_mode(located)
    named_h3 = bool(_H3_NAMED.search(request or ""))
    talking = bool(_TALKING.search(request or "")) or named_h3
    opt_in = h3_opt_in()
    t2v_lock = bool(_T2V_LOCK.search(request or ""))
    plain = variant in _PLAIN_T2V
    allowed_set = allowed

    route = "ltx_t2v"
    route_variant = variant
    score_variant = variant if _family_for(variant) != "flux" else "base"
    apply = False
    warning = None
    why: list[str] = []

    if has_image:
        route = "ltx_i2v"
        why.append(
            "Input image: yes. The route is LTX I2V with that image, not a new Flux still."
        )
        if named_h3 and h3_mode == "local":
            route = "h3"
            route_variant = "h3_r2v" if has_audio else "h3_i2v"
            score_variant = "h3_r2v" if has_audio else "h3_i2v"
            why.append(
                "The brief names MiniMax H3 and a local GGUF is on disk, so the image route is H3."
            )
            apply = route_variant != variant
    else:
        why.append("Input image: no.")

    if forced:
        why.append(
            f"User override wins: variant {variant} was forced. The route was not replaced."
        )
        apply = False
        route_variant = variant
    elif route == "ltx_i2v" and not (named_h3 and h3_mode == "local"):
        apply = False
    elif h3_mode == "hosted" and not (named_h3 or opt_in):
        why.append(
            "Paid hosted MiniMax H3 is not auto-selected. "
            "A hosted key is set and H3_OPT_IN is off, so H3 lost to LTX. "
            "A hosted call has a cost, and this repo has no hosted client."
        )
    elif h3_mode == "hosted" and (named_h3 or opt_in):
        why.append(
            "Paid hosted MiniMax H3 was requested "
            f"({'the brief names it' if named_h3 else 'H3_OPT_IN=1'}) "
            "but there is no local GGUF and this repo has no hosted client. "
            "H3 was not selected. A hosted call would have a cost."
        )
        warning = (
            "Hosted MiniMax H3 was not started. Set up local H3 weights or do not expect a call."
        )
    elif (
        h3_mode == "local"
        and talking
        and (named_h3 or opt_in)
        and not t2v_lock
        and not forced
    ):
        route = "h3"
        route_variant = "h3_r2v" if has_audio else "h3_t2v"
        score_variant = route_variant
        apply = route_variant != variant
        why.append(
            "MiniMax H3 wins over LTX and over Flux I2V: the brief needs talking or lip-sync, "
            "local H3 weights are on disk (no per-call charge), and "
            f"{'the user named H3' if named_h3 else 'H3_OPT_IN=1'}."
        )
    elif h3_mode == "local" and not talking:
        why.append(
            "MiniMax H3 is on disk but was not chosen: the brief is not a talking or lip-sync ask."
        )
    elif h3_mode == "absent":
        why.append("MiniMax H3 was skipped: no local GGUF and no hosted key.")

    flux_detail = ""
    if not installed:
        flux_detail = (
            f"{_FLUX_FILENAME} is not on this branch. "
            "The hybrid graph lives on draft PR #92 (branch cursor/gguf-vram-flux-ltx). "
            "This decision does not copy it."
        )
    elif flux_model is None:
        flux_detail = "The Flux I2V workflow file is present, but no Flux diffusion file is on disk."
    elif animator is None:
        flux_detail = (
            f"Flux model {flux_model} is on disk, but no LTX GGUF is on disk to animate the still."
        )
    elif together is not None and not fits:
        flux_detail = (
            f"Flux model {flux_model} plus the LTX GGUF need about {together:.1f} GB together, "
            f"which exceeds {budget:.1f} GB ({source})."
        )
    else:
        flux_detail = (
            f"Flux model {flux_model} is on disk. "
            f"Flux+LTX together is about {together:.1f} GB and fits {budget:.1f} GB ({source})."
        )

    if (
        not forced
        and not has_image
        and route != "h3"
        and not t2v_lock
        and plain
        and installed
        and flux_model
        and animator
        and fits
    ):
        target = _FLUX_VARIANT
        known = target in allowed_set if allowed_set is not None else _variant_allowed(target)
        if known:
            route = "flux_i2v"
            route_variant = target
            score_variant = animator[0]
            apply = True
            why.append(
                "T2V was not chosen. Flux I2V wins over plain LTX text-to-video: "
                "the workflow is installed, "
                f"{flux_model} is the Flux model, and Flux+LTX together fits VRAM. "
                f"The still is animated with {animator[2].name}."
            )
        else:
            why.append(
                "Flux I2V files are present but the variant is not registered, so T2V was chosen. "
                + flux_detail
            )
    elif t2v_lock and not forced:
        why.append("T2V was chosen because the brief asks for text-to-video (user override).")
    elif route == "ltx_t2v":
        why.append("T2V was chosen. " + flux_detail)
    elif route == "flux_i2v":
        pass
    else:
        why.append(flux_detail)

    if route != "flux_i2v" and "T2V was chosen" not in " ".join(why) and "T2V was not chosen" not in " ".join(why):
        if route == "h3":
            why.append("T2V was not chosen. MiniMax H3 took the route for talking or lip-sync.")
        elif route == "ltx_i2v":
            why.append("T2V was not chosen because an input image was supplied.")

    model = select_model(
        score_variant,
        search,
        vram_gb=budget,
        override=override,
    )
    model.route = route
    model.variant = route_variant
    model.score_variant = score_variant
    model.apply_route = apply and route_variant != variant
    model.why_route = " ".join(why)
    model.input_image = has_image
    model.flux_installed = installed
    model.flux_model = flux_model
    model.flux_detail = flux_detail
    model.together_gb = together
    model.h3_mode = h3_mode
    model.vram_gb = budget
    model.vram_source = source
    if warning and not model.warning:
        model.warning = warning
    elif warning and model.warning:
        model.warning = warning + " " + model.warning
    model.text = render_decision(model)
    log.info("%s", model.text)
    return model


def _variant_allowed(variant: str) -> bool:
    try:
        from master_agent.config import WORKFLOW_FILES

        return variant in WORKFLOW_FILES
    except Exception:
        return False


def render_decision(decision: ModelDecision) -> str:
    rejected = (
        "\n".join(f"- {row.name}: {row.reason}" for row in decision.rejected)
        or "- (none)"
    )
    comparisons = "\n".join(f"- {line}" for line in decision.comparisons) or "- (none)"
    together = (
        f"{decision.together_gb:.1f} GB"
        if decision.together_gb is not None
        else "n/a"
    )
    warning = f"\nWarning: {decision.warning}" if decision.warning else ""
    error = f"\nError: {decision.error}" if decision.error else ""
    return (
        f"Route: {decision.route}\n"
        f"Variant: {decision.variant}\n"
        f"Input image: {'yes' if decision.input_image else 'no'}\n"
        f"Flux I2V installed: {'yes' if decision.flux_installed else 'no'}. {decision.flux_detail}\n"
        f"Flux model: {decision.flux_model or 'none'}\n"
        f"VRAM: {decision.vram_gb:.1f} GB ({decision.vram_source}). "
        f"Flux+LTX together: {together}.\n"
        f"Why this route: {decision.why_route}\n"
        f"Model: {decision.chosen or 'none'}\n"
        f"Why this model: {decision.why_model}\n"
        f"Rejected:\n{rejected}\n"
        f"Comparisons:\n{comparisons}"
        f"{warning}{error}"
    )


def require_model(decision: ModelDecision) -> ModelDecision:
    """Raise when a base / eros / directors run has no usable GGUF."""
    if decision.error and _preset_key(decision.score_variant) in {"base", "eros", "directors"}:
        raise ModelSelectionError(decision.error)
    return decision
