"""Public GGUF sources for the local LLM (llama.cpp router).

Same idea as the LTX weight manifest: a pinned Hub repo, filename, and byte
size. Automatic install downloads the missing files into
``MODELS_DIR/<served-name>/``, which is the folder llama-server's
``--models-dir`` router exposes under that name. ``download_hub_file`` is the
resumable Hub fetch (cache resume, then a copy into ``MODELS_DIR``).

Before that download, a vision-capable pair already on disk is the brain.
``MODELS_DIR/gguf_vl_heretic/`` is checked first, then any other folder directly
under ``MODELS_DIR`` that holds a model ``*.gguf`` and an ``mmproj*.gguf``.
The preset points ``qwen3-vl-heretic`` at those files (``mmproj`` set) and the
4B file is not downloaded. An Ollama ``nomic-embed-text`` blob can be referenced
read-only instead of downloading the 84 MB embed file. Nothing already on disk
is moved, renamed, or deleted.

Verified against the Hugging Face file listing on 2026-10-22:

- ``DreamFast/Qwen3-VL-4b-Heretic-GGUF`` siblings include
  ``qwen3-vl-4b-heretic-Q4_K_M.gguf`` (2,497,281,696 bytes) and no file whose
  name contains ``mmproj``. The publisher marks these GGUFs as text-path.
- ``nomic-ai/nomic-embed-text-v1.5-GGUF`` includes
  ``nomic-embed-text-v1.5.Q4_K_M.gguf`` (84,106,624 bytes).

8B heretic GGUFs that do ship an mmproj are not downloaded. They are used when
they are already on disk. See ``OPTIONAL_BRAIN_ALTERNATIVES``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

# Hub listing 2026-10-22. Do not invent an mmproj for the 4B repo: it is not there.
BRAIN_REPO = "DreamFast/Qwen3-VL-4b-Heretic-GGUF"
BRAIN_FILENAME = "qwen3-vl-4b-heretic-Q4_K_M.gguf"
BRAIN_SIZE_BYTES = 2_497_281_696

EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
EMBED_FILENAME = "nomic-embed-text-v1.5.Q4_K_M.gguf"
EMBED_SIZE_BYTES = 84_106_624

LAUNCH_NOTES_NAME = "LAUNCH_NOTES.txt"
PRESET_NAME = "llamacpp-models.ini"

# Served name of the default brain (LLAMACPP_MODEL / VISION_MODEL).
DEFAULT_BRAIN_NAME = "qwen3-vl-heretic"
# Scott's tower keeps a complete vision pair here. Checked before any other folder.
PREFERRED_VISION_DIR = "gguf_vl_heretic"

_OLLAMA_MODEL_MEDIA = "application/vnd.ollama.image.model"
_SHA256_DIGEST = re.compile(r"^sha256:([0-9a-fA-F]+)$")
_GGUF_MAGIC = b"GGUF"

_PROVIDER_LINE = re.compile(
    r"^(?:export\s+)?LLM_PROVIDER\s*=\s*(.*?)\s*$",
    re.IGNORECASE,
)


def hub_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


@dataclass(frozen=True)
class LlmFile:
    """One GGUF automatic install may fetch."""

    filename: str
    repo_id: str
    repo_filename: str
    size_bytes: int
    role: str  # "model" or "mmproj"
    url: str

    @property
    def size_label(self) -> str:
        n = float(self.size_bytes)
        if n >= 1e9:
            return f"{n / 1e9:.1f} GB"
        if n >= 1e6:
            return f"{n / 1e6:.0f} MB"
        return f"{int(n)} B"


@dataclass(frozen=True)
class LlmAlternative:
    """Documented optional pack. Automatic install does not download these."""

    repo_id: str
    model_filename: str
    mmproj_filename: str
    note: str

    @property
    def model_url(self) -> str:
        return hub_url(self.repo_id, self.model_filename)

    @property
    def mmproj_url(self) -> str:
        return hub_url(self.repo_id, self.mmproj_filename)


# Not the default. Both were on the Hub on 2026-10-22, including an mmproj.
# llama.cpp's --models-dir router only auto-loads a projector whose filename
# starts with "mmproj". The mradermacher name does not, so a preset mmproj=
# line (or a rename) is required for that file.
OPTIONAL_BRAIN_ALTERNATIVES: tuple[LlmAlternative, ...] = (
    LlmAlternative(
        repo_id="SergiusFlavius/Qwen3-VL-8B-Instruct-heretic-gguf",
        model_filename="Qwen3-VL-8B-Instruct-heretic-Q4_K_M.gguf",
        mmproj_filename="mmproj-Qwen3-VL-8B-Instruct-heretic-F16.gguf",
        note="Optional 8B, not the default. The mmproj name starts with mmproj.",
    ),
    LlmAlternative(
        repo_id="mradermacher/Qwen-3-VL-8B-Instruct-heretic-GGUF",
        model_filename="Qwen-3-VL-8B-Instruct-heretic.Q4_K_M.gguf",
        mmproj_filename="Qwen-3-VL-8B-Instruct-heretic.mmproj-f16.gguf",
        note=(
            "Optional 8B, not the default. The mmproj name does not start with "
            "mmproj; point llama.cpp at it with a preset mmproj= line or rename it."
        ),
    ),
)


@dataclass(frozen=True)
class LlmSlot:
    """One name llama-server serves (``LLAMACPP_MODEL`` or ``KB_EMBED_MODEL``)."""

    name: str
    role: str
    files: tuple[LlmFile, ...]
    # True only when this slot's own repo published an mmproj we download.
    publishes_mmproj: bool = False
    vision: bool = False

    @property
    def has_source(self) -> bool:
        return bool(self.files)


def _file(
    filename: str,
    repo_id: str,
    size_bytes: int,
    *,
    role: str = "model",
) -> LlmFile:
    return LlmFile(
        filename=filename,
        repo_id=repo_id,
        repo_filename=filename,
        size_bytes=size_bytes,
        role=role,
        url=hub_url(repo_id, filename),
    )


# Keyed by the served name (the folder under MODELS_DIR, and the model id
# llama-server's router exposes). The 4B repo has no mmproj; a local pair
# can still fill this slot. See ``discover_local_vision_pair``.
LLM_SLOTS: dict[str, LlmSlot] = {
    DEFAULT_BRAIN_NAME: LlmSlot(
        name=DEFAULT_BRAIN_NAME,
        role="text + vision: brief, director, judge",
        files=(_file(BRAIN_FILENAME, BRAIN_REPO, BRAIN_SIZE_BYTES),),
        publishes_mmproj=False,
        vision=True,
    ),
    "nomic-embed-text": LlmSlot(
        name="nomic-embed-text",
        role="knowledge-base embeddings",
        files=(_file(EMBED_FILENAME, EMBED_REPO, EMBED_SIZE_BYTES),),
        publishes_mmproj=False,
        vision=False,
    ),
}


def models_root() -> Path:
    from master_agent import config

    return Path(config.MODELS_DIR)


def state_dir() -> Path:
    from master_agent import config

    return Path(config.STATE_DIR)


def launch_notes_path() -> Path:
    return state_dir() / LAUNCH_NOTES_NAME


def preset_path() -> Path:
    return state_dir() / PRESET_NAME


def env_file() -> Path:
    from master_agent.setup import ENV_FILE

    return Path(ENV_FILE)


def configured_slots() -> list[LlmSlot]:
    """Slots for ``LLAMACPP_MODEL`` and ``KB_EMBED_MODEL``. Unknown names have no files."""
    from master_agent import config

    out: list[LlmSlot] = []
    seen: set[str] = set()
    pairs = (
        (config.LLAMACPP_MODEL, "text + vision: brief, director, judge", True),
        (config.KB_EMBED_MODEL, "knowledge-base embeddings", False),
    )
    for name, role, vision in pairs:
        key = (name or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        known = LLM_SLOTS.get(key)
        if known is not None:
            out.append(known)
        else:
            out.append(LlmSlot(name=key, role=role, files=(), publishes_mmproj=False, vision=vision))
    return out


def slot_dir(slot: LlmSlot, root: Path | None = None) -> Path:
    return (root or models_root()) / slot.name


def size_ok(path: Path, size_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= size_bytes > 0
    except OSError:
        return False


def _matching_ggufs(name: str, root: Path) -> list[Path]:
    """GGUFs llama-server's ``--models-dir`` router would serve as ``name``.

    A file in ``MODELS_DIR`` is served by its stem. A file one folder down is
    served by that folder's name. Same rule as ``automatic_install.local_llm_gguf``.
    """
    want = name.strip().lower()
    if not want or not root.is_dir():
        return []
    found: list[Path] = []
    paths = sorted(root.glob("*.gguf")) + sorted(root.glob("*/*.gguf"))
    for path in paths:
        label = (path.parent.name if path.parent != root else path.stem).lower()
        if label == want or label.startswith(want + "-") or label.startswith(want + "."):
            found.append(path)
    return found


@dataclass(frozen=True)
class LocalVisionPair:
    """A model GGUF and an ``mmproj*.gguf`` that already share one folder."""

    model: Path
    mmproj: Path

    @property
    def directory(self) -> Path:
        return self.model.parent


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _pair_in_directory(folder: Path) -> LocalVisionPair | None:
    """Model ``*.gguf`` plus ``mmproj*.gguf`` as direct children. Read-only."""
    if not folder.is_dir():
        return None
    models: list[tuple[int, str, Path]] = []
    projectors: list[tuple[int, str, Path]] = []
    try:
        entries = list(folder.iterdir())
    except OSError:
        return None
    for path in entries:
        if not path.is_file() or path.suffix.lower() != ".gguf":
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= 0:
            continue
        key = path.name.lower()
        if key.startswith("mmproj"):
            projectors.append((size, key, path))
        elif "mmproj" not in key:
            models.append((size, key, path))
    if not models or not projectors:
        return None
    model = sorted(models, key=lambda item: (-item[0], item[1]))[0][2]
    mmproj = sorted(projectors, key=lambda item: (-item[0], item[1]))[0][2]
    return LocalVisionPair(model=model, mmproj=mmproj)


def _vision_dir_rank(name: str) -> int:
    label = name.lower()
    if "heretic" in label:
        return 0
    if "qwen" in label or "vl" in label:
        return 1
    return 2


def discover_local_vision_pair(root: Path | None = None) -> LocalVisionPair | None:
    """On-disk vision pair, or None.

    Order: ``MODELS_DIR/gguf_vl_heretic/``, then ``MODELS_DIR/qwen3-vl-heretic/``
    when that folder itself has a projector, then any other direct child folder
    that has both a model ``*.gguf`` and an ``mmproj*.gguf``. Names with
    ``heretic``, then ``qwen`` / ``vl``, win ties. Nested folders are not walked.
    Nothing is moved or deleted.
    """
    base = root or models_root()
    if not base.is_dir():
        return None
    preferred = _pair_in_directory(base / PREFERRED_VISION_DIR)
    if preferred is not None:
        return preferred
    own = _pair_in_directory(base / DEFAULT_BRAIN_NAME)
    if own is not None:
        return own
    ranked: list[tuple[int, str, LocalVisionPair]] = []
    try:
        children = list(base.iterdir())
    except OSError:
        return None
    skip = {PREFERRED_VISION_DIR, DEFAULT_BRAIN_NAME}
    for child in children:
        if not child.is_dir() or child.name in skip or child.name.startswith("."):
            continue
        hit = _pair_in_directory(child)
        if hit is None:
            continue
        ranked.append((_vision_dir_rank(child.name), child.name.lower(), hit))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def adopted_vision_pair(slot: LlmSlot, root: Path | None = None) -> LocalVisionPair | None:
    """Pair that should serve as the default brain, if this slot is that brain.

    A custom ``LLAMACPP_MODEL`` is left alone. Only ``qwen3-vl-heretic`` adopts
    a pair found in another folder, so the vision judge's model name stays put.
    """
    if not slot.vision or slot.name != DEFAULT_BRAIN_NAME:
        return None
    return discover_local_vision_pair(root or models_root())


def ollama_models_root() -> Path:
    """Ollama's models directory (``blobs`` and ``manifests``). Never created here."""
    raw = os.environ.get("OLLAMA_MODELS", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".ollama" / "models"


def is_ollama_reference(path: Path) -> bool:
    """True when ``path`` lives under the Ollama models directory."""
    return _is_under(path, ollama_models_root())


def _gguf_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == _GGUF_MAGIC
    except OSError:
        return False


def _ollama_blob(root: Path, digest: str) -> Path | None:
    match = _SHA256_DIGEST.match(digest.strip())
    if match is None:
        return None
    return root / "blobs" / f"sha256-{match.group(1)}"


def _blob_from_manifest(root: Path, manifest: Path) -> Path | None:
    try:
        if not manifest.is_file():
            return None
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    layers = data.get("layers") if isinstance(data, dict) else None
    if not isinstance(layers, list):
        return None
    ranked: list[tuple[int, int, Path]] = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        path = _ollama_blob(root, str(layer.get("digest") or ""))
        if path is None or not _gguf_magic(path):
            continue
        media = str(layer.get("mediaType") or "")
        try:
            size = int(layer.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        prefer = 0 if media == _OLLAMA_MODEL_MEDIA or media.endswith(".model") else 1
        ranked.append((prefer, -size, path))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return ranked[0][2]


def ollama_embed_blob(model_name: str = "nomic-embed-text") -> Path | None:
    """Read-only path to an Ollama GGUF blob for ``model_name``, or None.

    Looks up ``<name>:latest`` (or the tag in the name) under the Ollama
    manifests and returns the model-layer blob when it starts with the GGUF
    magic. Does not create, copy, move, or delete anything. A miss means the
    caller should download the Q4 file instead.
    """
    raw = (model_name or "").strip()
    if not raw:
        return None
    name, _, tag = raw.partition(":")
    name = name.strip().strip("/")
    tag = (tag or "latest").strip() or "latest"
    if not name or "/" in name or "/" in tag:
        return None
    root = ollama_models_root()
    manifests = (
        root / "manifests" / "registry.ollama.ai" / "library" / name / tag,
        root / "manifests" / "registry.ollama.ai" / name / tag,
    )
    for manifest in manifests:
        blob = _blob_from_manifest(root, manifest)
        if blob is not None:
            return blob
    return None


def adopted_embed_blob(slot: LlmSlot) -> Path | None:
    """Ollama embed blob to reference when this embed slot has no local GGUF.

    The blob lives outside ``MODELS_DIR``. Callers try it only after a local
    GGUF missed, and they never write to the path.
    """
    if slot.vision or "embed" not in slot.name:
        return None
    return ollama_embed_blob(slot.name)


def file_ready(slot: LlmSlot, spec: LlmFile, root: Path | None = None) -> Path | None:
    """Canonical file at full size, a stand-in GGUF, a local vision pair, or an Ollama embed blob.

    A short copy of the canonical filename is not ready: it is a partial download.
    A vision pair wins over the 4B file so the brain can see images.
    An Ollama embed blob is used only when no embed GGUF is already under ``MODELS_DIR``.
    """
    base = root or models_root()
    if spec.role == "model":
        pair = adopted_vision_pair(slot, base)
        if pair is not None:
            return pair.model
    dest = slot_dir(slot, base) / spec.filename
    if size_ok(dest, spec.size_bytes):
        return dest
    for path in _matching_ggufs(slot.name, base):
        if path.name == spec.filename:
            continue
        try:
            if path.is_file() and path.stat().st_size > 0 and "mmproj" not in path.name.lower():
                if spec.role == "model":
                    return path
        except OSError:
            continue
    if spec.role == "mmproj":
        for path in _matching_ggufs(slot.name, base):
            if path.name.lower().startswith("mmproj"):
                try:
                    if path.is_file() and path.stat().st_size > 0:
                        return path
                except OSError:
                    continue
    if spec.role == "model":
        # Only after every local candidate missed, so a file already in MODELS_DIR wins.
        return adopted_embed_blob(slot)
    return None


def mmproj_present(slot: LlmSlot, root: Path | None = None) -> Path | None:
    base = root or models_root()
    pair = adopted_vision_pair(slot, base)
    if pair is not None:
        return pair.mmproj
    for path in _matching_ggufs(slot.name, base):
        if "mmproj" in path.name.lower():
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return path
            except OSError:
                continue
    folder = slot_dir(slot, base)
    if folder.is_dir():
        for path in sorted(folder.glob("*.gguf")):
            if path.name.lower().startswith("mmproj") and path.stat().st_size > 0:
                return path
    return None


def model_path_for(slot: LlmSlot, root: Path | None = None) -> Path | None:
    """GGUF this slot will load, or None when the slot is not ready."""
    base = root or models_root()
    for spec in slot.files:
        if spec.role == "model":
            return file_ready(slot, spec, base)
    matches = [
        path
        for path in _matching_ggufs(slot.name, base)
        if "mmproj" not in path.name.lower() and path.is_file() and path.stat().st_size > 0
    ]
    return matches[0] if matches else None


def projector_how_to() -> str:
    """One plain sentence: how to give the vision judge a projector."""
    return (
        "To add a projector, put a model .gguf and a file whose name starts with mmproj "
        f"(mmproj*.gguf) in the same folder under MODELS_DIR. Buddy looks in "
        f"MODELS_DIR/{PREFERRED_VISION_DIR}/ first, then any other folder that has both. "
        "It does not download the projector and it does not move or delete files. "
        "Then run: python -m master_agent automatic-install --yes"
    )


def slot_ready(slot: LlmSlot, root: Path | None = None) -> bool:
    if not slot.files:
        base = root or models_root()
        return any(
            p.is_file() and p.stat().st_size > 0 and "mmproj" not in p.name.lower()
            for p in _matching_ggufs(slot.name, base)
        )
    return all(file_ready(slot, spec, root) is not None for spec in slot.files)


def bytes_still_needed(root: Path | None = None, slots: list[LlmSlot] | None = None) -> int:
    total = 0
    for slot in slots if slots is not None else configured_slots():
        for spec in slot.files:
            if file_ready(slot, spec, root) is None:
                total += spec.size_bytes
    return total


def usage_lines(root: Path | None = None) -> list[str]:
    """Which brain, projector, and embed files are in use, and whether vision is on."""
    base = models_root() if root is None else root
    lines: list[str] = []
    brain = next((slot for slot in configured_slots() if slot.vision), None)
    embed = next((slot for slot in configured_slots() if not slot.vision), None)
    if brain is not None:
        model = model_path_for(brain, base)
        proj = mmproj_present(brain, base)
        if model is not None:
            lines.append(f"brain: {model}")
        else:
            lines.append(f"brain: missing ({brain.name})")
        if proj is not None:
            lines.append(f"projector: {proj}")
            lines.append("vision: enabled")
        else:
            lines.append("projector: none")
            lines.append("vision: text-only")
    if embed is not None:
        path = model_path_for(embed, base)
        if path is None:
            lines.append(f"embed: missing ({embed.name})")
        elif is_ollama_reference(path):
            lines.append(f"embed: {path} (read-only Ollama file, not copied)")
        else:
            lines.append(f"embed: {path}")
    return lines


def vision_note(root: Path | None = None) -> str:
    """Plain-language vision status for the install notes.

    With a projector on disk, vision is on and the 4B download is not the brain.
    With only the 4B (or nothing), vision is text-only and the note says how to
    add a projector. ``root`` defaults to ``MODELS_DIR``.
    """
    base = models_root() if root is None else root
    brain = next((slot for slot in configured_slots() if slot.vision), None)
    model = model_path_for(brain, base) if brain is not None else None
    proj = mmproj_present(brain, base) if brain is not None else None
    if model is not None and proj is not None:
        canonical = slot_dir(brain, base) / BRAIN_FILENAME if brain is not None else None
        if canonical is not None and model.resolve() == canonical.resolve():
            which = "The brain is the 4B file plus this projector."
        else:
            which = "This existing pair is the brain. The 4B file is not downloaded."
        return "\n".join(
            [
                "Vision: enabled. The vision judge can see images.",
                f"Brain: {model}",
                f"Projector: {proj}",
                "llama-server reads the mmproj line in state/llamacpp-models.ini and loads that projector (--mmproj).",
                which,
                "The files stay where they are. Buddy does not move, rename, or delete them.",
            ]
        )
    lines = [
        "Vision: text-only. The vision judge cannot see images (no mmproj).",
        f"Checked {BRAIN_REPO}: {BRAIN_FILENAME} is text-path only.",
        "Text still works (brief, director, storyboard, text judge).",
        projector_how_to(),
        "Optional packs that include a projector (not downloaded):",
    ]
    for alt in OPTIONAL_BRAIN_ALTERNATIVES:
        lines.append(f"  - {alt.model_url}")
        lines.append(f"    mmproj: {alt.mmproj_url}")
        lines.append(f"    {alt.note}")
    return "\n".join(lines)


def manual_lines(slot: LlmSlot, *, reason: str) -> list[str]:
    """Exact steps and the direct link. Shown when a slot cannot be fetched."""
    folder = slot_dir(slot)
    lines = [
        f"WARNING: {slot.name} ({slot.role}) is not ready.",
        reason,
    ]
    if not slot.files:
        lines += [
            f"This install has no public download for {slot.name}.",
            f"1. Get a GGUF for {slot.name} yourself.",
            f"2. Create this folder: {folder}",
            "3. Put the .gguf file in that folder. llama.cpp serves the folder name as the model.",
            "4. Run again: python -m master_agent automatic-install --yes",
        ]
        return lines
    lines.append(f"Put the file in {folder} (llama.cpp serves that folder as {slot.name}).")
    step = 1
    for spec in slot.files:
        lines.append(f"{step}. Download {spec.filename} ({spec.size_label}).")
        lines.append(f"   Direct link: {spec.url}")
        lines.append(f"   Save it as: {folder / spec.filename}")
        step += 1
    lines.append(
        f"{step}. Run again: python -m master_agent automatic-install --yes"
    )
    lines.append("   A file that is already the full size is skipped. A short file is downloaded again.")
    return lines


def provider_sentence(value: str) -> str:
    """Plain language for someone who has not used either program."""
    name = (value or "auto").strip().lower()
    if name in {"", "auto"}:
        return (
            "LLM_PROVIDER=auto. Buddy tries llama.cpp first: the model files on this computer. "
            "If that program is not running, it tries Ollama, a separate app you install yourself. "
            "It does not use a cloud model."
        )
    if name in {"llamacpp", "llama.cpp", "llama-cpp", "llama_cpp"}:
        return (
            "LLM_PROVIDER=llamacpp. Buddy uses llama.cpp on this computer and does not switch to Ollama. "
            "You do not need to install Ollama."
        )
    if name == "ollama":
        return (
            "LLM_PROVIDER=ollama. Buddy uses the Ollama app, which you install yourself from "
            "https://ollama.com/download. The GGUF files are for llama.cpp. "
            "To use them, set LLM_PROVIDER=llamacpp in .env."
        )
    if name == "grok":
        return (
            "LLM_PROVIDER=grok. Buddy uses a cloud model. The local GGUF files are not used "
            "until you set LLM_PROVIDER=llamacpp in .env."
        )
    return f"LLM_PROVIDER={value}."


def read_provider(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PROVIDER_LINE.match(stripped)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    return None


def ensure_provider(apply: bool) -> tuple[str, str]:
    """Return ``(value, action)`` where action is unchanged, planned, set, or unset.

    An explicit line in ``.env`` is left alone (Ollama, llama.cpp, auto, or grok).
    When the line is missing and ``apply`` is true, write ``LLM_PROVIDER=llamacpp``.
    """
    from master_agent import config

    path = env_file()
    current = None
    text = ""
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        current = read_provider(text)
    if current:
        return current, "unchanged"
    if not apply:
        return "llamacpp", "planned"
    if not path.is_file():
        return (config.LLM_PROVIDER or "auto"), "unset"
    suffix = "" if text.endswith("\n") or text == "" else "\n"
    path.write_text(text + suffix + "LLM_PROVIDER=llamacpp\n", encoding="utf-8")
    config.LLM_PROVIDER = "llamacpp"
    os.environ["LLM_PROVIDER"] = "llamacpp"
    return "llamacpp", "set"


def download_into(spec: LlmFile, dest: Path, progress=print) -> Path:
    """Fetch ``spec`` to ``dest``. Skip is the caller's job. Short dest is replaced.

    The Hub download itself resumes in the Hugging Face cache (same helper the
    LTX weights use). The copy into ``MODELS_DIR`` happens after that file is complete.
    An Ollama blob is never the destination: those files are referenced, not changed.
    """
    dest = Path(dest)
    if is_ollama_reference(dest):
        raise RuntimeError(
            f"Refusing to change {dest}. An Ollama file is only referenced, never modified."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not size_ok(dest, spec.size_bytes):
        dest.unlink()
    from master_agent.models.download import download_hub_file

    got = download_hub_file(
        repo_id=spec.repo_id,
        repo_filename=spec.repo_filename,
        dest=dest,
        progress=progress,
    )
    if Path(got).resolve() != dest.resolve():
        if not size_ok(Path(got), spec.size_bytes):
            raise RuntimeError(
                f"Found {got} but it is smaller than {spec.size_bytes} bytes. "
                f"Download {spec.url} and save it as {dest}."
            )
        shutil.copyfile(got, dest)
    if not size_ok(dest, spec.size_bytes):
        have = dest.stat().st_size if dest.is_file() else 0
        raise RuntimeError(
            f"{dest.name} is {have} bytes; expected at least {spec.size_bytes}. "
            f"Download {spec.url} and save it as {dest}."
        )
    return dest


@dataclass
class SlotOutcome:
    slot: LlmSlot
    status: str  # present | downloaded | planned | failed
    detail: str
    lines: list[str]


def _found_lines(slot: LlmSlot, root: Path, found: list[str]) -> list[str]:
    head = f"{slot.name} ({slot.role}): found {found[0]}" if found else f"{slot.name}: found"
    lines = [head]
    if slot.vision:
        proj = mmproj_present(slot, root)
        if proj is not None:
            lines.append(f"  projector: {proj}")
            lines.append("  vision: enabled")
        else:
            lines.append("  vision: text-only (no mmproj). The vision judge cannot see images.")
            lines.append(f"  {projector_how_to()}")
    elif found and is_ollama_reference(Path(found[0])):
        lines.append("  read-only Ollama file; not copied or modified")
    return lines


def install_slot(slot: LlmSlot, *, apply: bool, fetch: bool, progress=print) -> SlotOutcome:
    root = models_root()
    if slot_ready(slot, root):
        found = []
        for spec in slot.files:
            path = file_ready(slot, spec, root)
            if path is not None:
                found.append(str(path))
        if not found:
            found = [str(p) for p in _matching_ggufs(slot.name, root)]
        detail = f"found {found[0]}" if found else "found"
        return SlotOutcome(slot, "present", detail, _found_lines(slot, root, found))

    if not slot.has_source:
        lines = manual_lines(slot, reason="No public GGUF source is defined for this name.")
        return SlotOutcome(slot, "failed", "no public GGUF source", lines)

    if not apply or not fetch:
        lines = []
        for spec in slot.files:
            dest = slot_dir(slot, root) / spec.filename
            if file_ready(slot, spec, root) is not None:
                continue
            lines.append(
                f"{spec.filename} -> {dest} ({spec.size_label})  [{spec.repo_id}]"
            )
            lines.append(f"   {spec.url}")
        if not fetch and apply:
            lines = manual_lines(
                slot,
                reason="This light install does not download the multi-GB GGUF. Automatic install does.",
            )
            return SlotOutcome(slot, "failed", "not downloaded by the light install", lines)
        return SlotOutcome(
            slot,
            "planned",
            f"download {len(slot.files)} file(s)",
            lines,
        )

    lines: list[str] = []
    try:
        root.mkdir(parents=True, exist_ok=True)
        for spec in slot.files:
            if file_ready(slot, spec, root) is not None:
                lines.append(f"{slot.name}: {spec.filename} already present")
                continue
            dest = slot_dir(slot, root) / spec.filename
            progress(f"Downloading {spec.repo_id}/{spec.filename} -> {dest}")
            download_into(spec, dest, progress=progress)
            if not size_ok(dest, spec.size_bytes):
                raise RuntimeError(f"{dest} failed the size check ({spec.size_bytes} bytes)")
            lines.append(f"{slot.name}: downloaded {dest}")
    except Exception as exc:
        fail = manual_lines(slot, reason=f"Download failed: {exc}")
        return SlotOutcome(slot, "failed", str(exc)[:400], [*lines, *fail])
    if not slot_ready(slot, root):
        fail = manual_lines(slot, reason="Download finished but the file is still missing or short.")
        return SlotOutcome(slot, "failed", "verify failed", [*lines, *fail])
    return SlotOutcome(slot, "downloaded", "downloaded", lines)


def write_preset(root: Path | None = None) -> Path | None:
    """INI llama-server reads with ``--models-preset``. Only slots that are on disk."""
    base = root or models_root()
    sections: list[str] = [
        "version = 1",
        "",
        "; Written by automatic install. Names match LLAMACPP_MODEL and KB_EMBED_MODEL.",
        "; llama-server --models-dir serves the folder name; this preset pins the file.",
        "; A mmproj line is the projector llama-server loads for that model (--mmproj).",
        "",
    ]
    wrote = False
    for slot in configured_slots():
        model_path: Path | None = None
        mmproj: Path | None = None
        pair = adopted_vision_pair(slot, base)
        if pair is not None:
            model_path = pair.model
            mmproj = pair.mmproj
        for spec in slot.files:
            if model_path is not None and spec.role == "model":
                continue
            if mmproj is not None and spec.role == "mmproj":
                continue
            if spec.role == "model":
                ready = file_ready(slot, spec, base)
                if ready is not None and ready.name == spec.filename and size_ok(ready, spec.size_bytes):
                    model_path = ready
                elif ready is not None and spec.filename != ready.name:
                    model_path = ready
            elif spec.role == "mmproj":
                ready = file_ready(slot, spec, base)
                if ready is not None:
                    mmproj = ready
        if model_path is None and slot_ready(slot, base):
            matches = [
                p
                for p in _matching_ggufs(slot.name, base)
                if "mmproj" not in p.name.lower() and p.stat().st_size > 0
            ]
            model_path = matches[0] if matches else None
        if mmproj is None and slot.vision:
            mmproj = mmproj_present(slot, base)
        if model_path is None:
            continue
        wrote = True
        sections.append(f"[{slot.name}]")
        sections.append(f"model = {model_path.resolve()}")
        if mmproj is not None:
            sections.append(f"mmproj = {mmproj.resolve()}")
        if slot.name == "nomic-embed-text" or "embed" in slot.name:
            sections.append("embedding = true")
        sections.append("")
    if not wrote:
        return None
    path = preset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return path


def render_launch_notes(*, outcomes: list[SlotOutcome], provider: str, provider_action: str) -> str:
    failed = [o for o in outcomes if o.status == "failed"]
    lines = [
        "LOCAL LLM",
        "",
        provider_sentence(provider),
        f"(this install: {provider_action})",
        "",
    ]
    if failed:
        lines.append("WARNING: one or more local LLM files are missing.")
        lines.append("Brief, director, storyboard, text judge, and vision judge stay off until you fix this.")
        lines.append("Video generation can still run.")
        lines.append("")
        for outcome in failed:
            lines.extend(outcome.lines)
            lines.append("")
    else:
        lines.append("The local LLM files are in place.")
        for outcome in outcomes:
            lines.append(f"  {outcome.slot.name}: {outcome.status}")
        lines.append("")
    lines.append(vision_note())
    lines.append("")
    lines.extend(usage_lines())
    lines.append("")
    lines.append("llama.cpp is the program on this computer. Ollama is a different app.")
    lines.append("Buddy does not install Ollama and does not run ollama pull.")
    return "\n".join(lines).rstrip() + "\n"


def write_launch_notes(text: str) -> Path:
    path = launch_notes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def warning_box(notes_path: Path, failed: list[SlotOutcome]) -> str:
    bar = "!" * 72
    body = [
        bar,
        "!  WARNING: LOCAL LLM IS NOT READY",
        "!  Brief, director, storyboard, text judge, and vision judge will not run.",
        "!  Video generation can still run. This is not a silent skip.",
        "!",
    ]
    for outcome in failed:
        for line in outcome.lines:
            body.append("!  " + line)
        body.append("!")
    body.append(f"!  Full steps: {notes_path}")
    body.append(bar)
    return "\n".join(body)


def doctor_row() -> dict[str, str | bool]:
    """Scan only. Missing files are not OK. The caller prints this as WARN, not a quiet INFO."""
    from master_agent import config

    provider = config.LLM_PROVIDER or "auto"
    missing: list[str] = []
    present: list[str] = []
    fixes: list[str] = []
    for slot in configured_slots():
        if slot_ready(slot):
            present.append(slot.name)
            continue
        missing.append(slot.name)
        if slot.files:
            fixes.append(slot.files[0].url + " -> " + str(slot_dir(slot) / slot.files[0].filename))
        else:
            fixes.append(f"no public source for {slot.name}; put a GGUF in {slot_dir(slot)}")
    parts = [f"LLM_PROVIDER={provider}"]
    if present:
        parts.append("present: " + ", ".join(present))
    if missing:
        parts.append("MISSING: " + ", ".join(missing))
    parts.extend(usage_lines())
    brain = next((slot for slot in configured_slots() if slot.vision), None)
    proj = mmproj_present(brain) if brain is not None else None
    if brain is not None and brain.name == DEFAULT_BRAIN_NAME and proj is None:
        parts.append("vision mmproj: not in the default repo (text works; vision judge cannot see images)")
        if model_path_for(brain) is not None:
            parts.append(projector_how_to())
    elif proj is not None:
        parts.append(f"vision mmproj: {proj.name}")
    fix = ""
    if missing:
        fix = " | ".join(fixes) + "  then python -m master_agent automatic-install --yes"
        fix += f"  See {launch_notes_path()}"
    return {
        "name": "llm-models",
        "ok": not missing,
        "detail": "; ".join(parts),
        "fix": fix,
    }
