"""Download Klein 4B / LTX 2.5 weights into models/ (idempotent, consent-gated).

FLUX.2 [klein] 4B (Apache 2.0, ungated) is the only Flux stills pack:
  - black-forest-labs/FLUX.2-klein-4b-fp8 / flux-2-klein-4b-fp8.safetensors
  - Comfy-Org/Flux-2-Klein-4B text encoder qwen_3_4b_fp4_flux2.safetensors
  - Comfy-Org/Flux-2-Klein-4B VAE flux2-vae.safetensors

Flux.1 dev/schnell/pro/fill/depth/canny/kontext, Klein 9B, and FLUX.3
are not offered.

LTX 2.5 official bf16 split pack lives in ``master_agent.models.weights``
(Hub paths under ``Lightricks/LTX-2.5``). Never auto-download those gated
files — callers must pass ``yes=True``.

Local-first: files already at the destination, in Comfy ``models/``,
``EXTRA_MODELS_DIRS``, ``extra_model_paths.yaml``, or the Hugging Face hub
cache are reused. A clear SKIP log is printed. Nothing is re-downloaded
when a usable local copy exists.

New bytes are written only under Buddy ``MODELS_DIR``. A destination inside
an attached or external Comfy tree is refused.

If configured model dirs are missing, this module refuses to start a Hub
fetch and names every path it checked.

Usage:
    python -m master_agent download-flux
    python -m master_agent download-models --ltx25 --yes
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

class DownloadDestinationError(RuntimeError):
    """Pack download destination is outside Buddy ``MODELS_DIR`` or inside an attached tree."""


class LocalModelNotFound(RuntimeError):
    """Configured model dirs are missing or unusable; do not start a Hub fetch."""

    def __init__(
        self,
        message: str,
        *,
        paths_checked: list[str],
        filename: str = "",
    ):
        super().__init__(message)
        self.paths_checked = paths_checked
        self.filename = filename


def _usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _skip_local(path: Path, dest: Path, progress: Callable[[str], None]) -> Path:
    progress(
        f"SKIP download of {dest.name} — local file found at {path} (not re-downloading)"
    )
    return path


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True


def assert_pack_download_destination(dest: Path) -> Path:
    """Require ``dest`` under Buddy ``MODELS_DIR`` and outside attached Comfy trees.

    Search order is unchanged (``MODELS_DIR``, then Comfy ``models/``, then
    YAML / HF). This guard only blocks the write.
    """
    from master_agent.comfy.model_paths import path_is_comfy_like, read_only_weight_roots
    from master_agent.config import MODELS_DIR as models_dir

    models = Path(models_dir).expanduser().resolve()
    target = Path(dest).expanduser().resolve()
    if not _is_under(target, models):
        raise DownloadDestinationError(
            f"Refusing to write {target} outside Buddy MODELS_DIR ({models}). "
            "download-models and pack pulls never write into an attached or "
            "external Comfy tree."
        )
    for tree in read_only_weight_roots():
        try:
            root = Path(tree).expanduser().resolve()
        except OSError:
            continue
        if root == models or _is_under(target, root):
            if _is_under(models, root) and root != models and not path_is_comfy_like(root):
                continue
            raise DownloadDestinationError(
                f"Refusing to write {target} into read-only model tree {root}. "
                f"Pack downloads go only to Buddy MODELS_DIR ({models})."
            )
    return target


def _configured_model_roots() -> list[Path]:
    """Read live config (tests and runtime env rebinds must be visible)."""
    from master_agent.config import COMFYUI_ROOT as comfy_root
    from master_agent.config import MODELS_DIR as models_dir
    from master_agent.config import extra_models_dirs as extra_dirs

    return [Path(models_dir), Path(comfy_root) / "models", *extra_dirs()]


def _find_existing_local(repo_filename: str, dest: Path) -> Path | None:
    """Reuse dest, Comfy/extra trees, accepted aliases, or the HF hub cache."""
    if _usable(dest):
        return dest
    needle = Path(str(repo_filename).replace("\\", "/")).name or dest.name
    try:
        from master_agent.models.weights import find_weight_file

        for name in (needle, dest.name, repo_filename):
            found = find_weight_file(name)
            if found is not None:
                return found
    except Exception:
        return None
    return None


_XET_ERROR_MARKERS = ("xet", "cas service", "cas client", "cas error", "cas::", "reconstruction")


def is_xet_error(exc: BaseException) -> bool:
    """True when ``exc`` (or its cause chain) looks like an hf_xet / CAS failure."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = f"{type(cur).__module__}.{type(cur).__name__} {cur}".lower()
        if any(marker in text for marker in _XET_ERROR_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def xet_disabled() -> bool:
    return os.environ.get("HF_HUB_DISABLE_XET", "").strip().lower() in {"1", "true", "yes", "on"}


def disable_xet() -> None:
    """Turn Xet off for this process; huggingface_hub reads the flag at import."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import constants

        constants.HF_HUB_DISABLE_XET = True
    except Exception:
        pass


def download_hub_file(
    *,
    repo_id: str,
    repo_filename: str,
    dest: Path,
    progress: Callable[[str], None] = print,
) -> Path:
    """Reuse a local copy when present. Only hits the Hub after a full scan."""
    found = _find_existing_local(repo_filename, dest)
    if found is not None:
        return _skip_local(found, dest, progress)

    # try_to_load_from_cache with the real repo (no network)
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id=repo_id, filename=repo_filename)
        if cached and _usable(Path(cached)):
            return _skip_local(Path(cached), dest, progress)
    except Exception:
        pass

    roots: list[Path] = []
    try:
        from master_agent.models.weights import model_search_roots

        roots = list(model_search_roots())
    except Exception:
        roots = []
    configured = _configured_model_roots()
    if not any(p.is_dir() for p in configured):
        checked: list[str] = []
        seen: set[str] = set()
        for path in (*configured, *roots, dest.parent):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            checked.append(key)
        raise LocalModelNotFound(
            f"Refusing to download {dest.name} from {repo_id}/{repo_filename} — "
            "no usable local model directories were found. "
            "Set MODELS_DIR, COMFYUI_ROOT, EXTRA_MODELS_DIRS, or "
            "Comfy extra_model_paths.yaml.\n"
            "Paths checked:\n" + "\n".join(f"  - {p}" for p in checked),
            paths_checked=checked,
            filename=dest.name,
        )

    dest = assert_pack_download_destination(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError

    progress(f"Downloading {repo_id}/{repo_filename} ...")
    try:
        try:
            cached = hf_hub_download(repo_id=repo_id, filename=repo_filename)
        except GatedRepoError:
            raise
        except Exception as exc:
            if not is_xet_error(exc) or xet_disabled():
                raise
            progress(
                f"WARN Xet/CAS download of {repo_filename} failed ({exc}); "
                "retrying with HF_HUB_DISABLE_XET=1 (plain HTTPS)"
            )
            disable_xet()
            cached = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    except GatedRepoError as exc:
        raise RuntimeError(
            f"Gated repo {repo_id} — accept the license on Hugging Face and set "
            f"HF_TOKEN (huggingface-cli login). {exc}"
        ) from exc
    shutil.copyfile(cached, dest)
    progress(f"OK   {dest}")
    return dest


def download_flux_weights(progress: Callable[[str], None] = print) -> list[Path]:
    """Download FLUX.2 [klein] 4B into MODELS_DIR. Flux.1 is not offered."""
    from master_agent.models.klein import LICENSE_LINE, download_klein_4b

    progress(LICENSE_LINE)
    return download_klein_4b(progress)
