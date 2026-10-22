"""Public GGUF sources for the local LLM (llama.cpp router).

Same idea as the LTX weight manifest: a pinned Hub repo, filename, and byte
size. Automatic install downloads the missing files into
``MODELS_DIR/<served-name>/``, which is the folder llama-server's
``--models-dir`` router exposes under that name. ``download_hub_file`` is the
resumable Hub fetch (cache resume, then a copy into ``MODELS_DIR``).

Verified against the Hugging Face file listing on 2026-10-22:

- ``DreamFast/Qwen3-VL-4b-Heretic-GGUF`` siblings include
  ``qwen3-vl-4b-heretic-Q4_K_M.gguf`` (2,497,281,696 bytes) and no file whose
  name contains ``mmproj``. The publisher marks these GGUFs as text-path.
- ``nomic-ai/nomic-embed-text-v1.5-GGUF`` includes
  ``nomic-embed-text-v1.5.Q4_K_M.gguf`` (84,106,624 bytes).

8B heretic GGUFs that do ship an mmproj are optional, not the default. See
``OPTIONAL_BRAIN_ALTERNATIVES``.
"""

from __future__ import annotations

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
# llama-server's router exposes). mmproj is absent on purpose.
LLM_SLOTS: dict[str, LlmSlot] = {
    "qwen3-vl-heretic": LlmSlot(
        name="qwen3-vl-heretic",
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


def file_ready(slot: LlmSlot, spec: LlmFile, root: Path | None = None) -> Path | None:
    """Canonical file at full size, or a different non-empty GGUF the user already put here.

    A short copy of the canonical filename is not ready: it is a partial download.
    """
    base = root or models_root()
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
    return None


def mmproj_present(slot: LlmSlot, root: Path | None = None) -> Path | None:
    base = root or models_root()
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


def vision_note() -> str:
    lines = [
        "Vision: the default brain repo has no mmproj file, so the vision judge cannot see images.",
        f"Checked {BRAIN_REPO}: {BRAIN_FILENAME} is text-path only.",
        "Text still works (brief, director, storyboard, text judge).",
        "Optional 8B packs that include an mmproj (not downloaded by default):",
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
    """
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
        return SlotOutcome(slot, "present", f"found {found[0]}" if found else "found", [f"{slot.name} ({slot.role}): found {found[0]}" if found else f"{slot.name}: found"])

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
        "",
    ]
    wrote = False
    for slot in configured_slots():
        model_path: Path | None = None
        mmproj: Path | None = None
        for spec in slot.files:
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
    vision = mmproj_present(LLM_SLOTS["qwen3-vl-heretic"]) if "qwen3-vl-heretic" in {s.name for s in configured_slots()} else None
    brain = next((s for s in configured_slots() if s.vision), None)
    if brain is not None and brain.name == "qwen3-vl-heretic" and vision is None:
        parts.append("vision mmproj: not in the default repo (text works; vision judge cannot see images)")
    elif vision is not None:
        parts.append(f"vision mmproj: {vision.name}")
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
