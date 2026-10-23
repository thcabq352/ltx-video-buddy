"""Local LLM GGUF manifest and install. No GPU, no network, no multi-GB files.

Run: python -m pytest tests/test_llm_gguf_install.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from master_agent import config
from master_agent import llamacpp_server as srv
from master_agent.models import llm_gguf

BRAIN_URL = (
    "https://huggingface.co/DreamFast/Qwen3-VL-4b-Heretic-GGUF/resolve/main/"
    "qwen3-vl-4b-heretic-Q4_K_M.gguf"
)
EMBED_URL = (
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/"
    "nomic-embed-text-v1.5.Q4_K_M.gguf"
)
MMPROJ_URL = (
    "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/"
    "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
)


def test_manifest_pins_the_4b_brain_the_qwen_projector_and_the_embed():
    brain = llm_gguf.LLM_SLOTS["qwen3-vl-heretic"]
    embed = llm_gguf.LLM_SLOTS["nomic-embed-text"]
    assert brain.files[0].filename == "qwen3-vl-4b-heretic-Q4_K_M.gguf"
    assert brain.files[0].repo_id == "DreamFast/Qwen3-VL-4b-Heretic-GGUF"
    assert brain.files[0].size_bytes == 2_497_281_696
    assert brain.files[0].sha256 == llm_gguf.BRAIN_SHA256
    assert brain.files[0].url == BRAIN_URL
    assert "mmproj" not in brain.files[0].filename.lower()
    proj = brain.files[1]
    assert proj.role == "mmproj"
    assert proj.filename == llm_gguf.MMPROJ_FILENAME
    assert proj.repo_id == "Qwen/Qwen3-VL-4B-Instruct-GGUF"
    assert proj.size_bytes == 836_180_256
    assert proj.sha256 == llm_gguf.MMPROJ_SHA256
    assert proj.url == MMPROJ_URL
    assert brain.publishes_mmproj is True
    assert embed.files[0].filename == "nomic-embed-text-v1.5.Q4_K_M.gguf"
    assert embed.files[0].repo_id == "nomic-ai/nomic-embed-text-v1.5-GGUF"
    assert embed.files[0].size_bytes == 84_106_624
    assert embed.files[0].sha256 == llm_gguf.EMBED_SHA256
    assert embed.files[0].url == EMBED_URL
    assert llm_gguf.bytes_still_needed(root=Path("/no/such/models"), slots=[brain, embed]) == (
        2_497_281_696 + 836_180_256 + 84_106_624
    )
    repos = {alt.repo_id for alt in llm_gguf.OPTIONAL_BRAIN_ALTERNATIVES}
    assert repos == {
        "SergiusFlavius/Qwen3-VL-8B-Instruct-heretic-gguf",
        "mradermacher/Qwen-3-VL-8B-Instruct-heretic-GGUF",
    }
    note = llm_gguf.vision_note(root=Path("/no/such/llm-models"))
    assert "text-only" in note and "no mmproj" in note
    assert llm_gguf.MMPROJ_FILENAME in note and "Qwen/Qwen3-VL-4B-Instruct-GGUF" in note
    assert "gguf_vl_heretic" in note
    assert "mmproj-Qwen3-VL-8B-Instruct-heretic-F16.gguf" in note
    assert "Qwen-3-VL-8B-Instruct-heretic.mmproj-f16.gguf" in note


def _tiny(name: str = "tiny-brain", size: int = 8) -> llm_gguf.LlmSlot:
    spec = llm_gguf.LlmFile(
        filename="tiny.gguf",
        repo_id="org/repo",
        repo_filename="tiny.gguf",
        size_bytes=size,
        role="model",
        url="https://example.invalid/tiny.gguf",
    )
    return llm_gguf.LlmSlot(name=name, role="role", files=(spec,))


def test_short_canonical_is_not_ready_and_a_stand_in_is(tmp_path):
    slot = llm_gguf.LLM_SLOTS["qwen3-vl-heretic"]
    folder = tmp_path / slot.name
    folder.mkdir()
    short = folder / llm_gguf.BRAIN_FILENAME
    short.write_bytes(b"x")
    assert llm_gguf.file_ready(slot, slot.files[0], tmp_path) is None
    stand_in = tmp_path / "qwen3-vl-heretic-custom.gguf"
    stand_in.write_bytes(b"x")
    assert llm_gguf.file_ready(slot, slot.files[0], tmp_path) == stand_in


def test_install_slot_skips_a_full_file(tmp_path, monkeypatch):
    slot = _tiny()
    dest = tmp_path / slot.name / "tiny.gguf"
    dest.parent.mkdir()
    dest.write_bytes(b"12345678")
    monkeypatch.setattr(llm_gguf, "models_root", lambda: tmp_path)
    monkeypatch.setattr(llm_gguf, "download_into", lambda *a, **k: pytest.fail("downloaded"))
    out = llm_gguf.install_slot(slot, apply=True, fetch=True)
    assert out.status == "present"


def test_install_slot_refetches_a_short_canonical(tmp_path, monkeypatch):
    slot = _tiny()
    dest = tmp_path / slot.name / "tiny.gguf"
    dest.parent.mkdir()
    dest.write_bytes(b"x")
    monkeypatch.setattr(llm_gguf, "models_root", lambda: tmp_path)

    def fake(spec, target, progress=print):
        target.write_bytes(b"12345678")
        return target

    monkeypatch.setattr(llm_gguf, "download_into", fake)
    out = llm_gguf.install_slot(slot, apply=True, fetch=True)
    assert out.status == "downloaded"
    assert dest.read_bytes() == b"12345678"


class _Body:
    """In-memory HTTPS response. ``limit`` stops the body early, like a dropped connection."""

    def __init__(self, payload: bytes, *, status: int = 200, headers: dict | None = None, url: str = "", limit: int | None = None):
        self._payload = payload
        self._off = 0
        self._limit = len(payload) if limit is None else limit
        self.status = status
        self.headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        self._url = url

    def read(self, n: int = -1) -> bytes:
        if self._off >= self._limit:
            return b""
        if n is None or n < 0:
            n = self._limit - self._off
        n = min(n, self._limit - self._off)
        chunk = self._payload[self._off : self._off + n]
        self._off += len(chunk)
        return chunk

    def getheader(self, name: str, default=None):
        return self.headers.get(name.lower(), default)

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def test_download_resolve_resumes_checks_size_and_sha_and_is_idempotent(tmp_path, monkeypatch):
    """Direct resolve URL, Range resume from .partial, exact size, SHA256, then no second GET."""
    import hashlib

    body = b"abcdefghij"
    digest = hashlib.sha256(body).hexdigest()
    url = "https://example.invalid/tiny.gguf"
    dest = tmp_path / "models" / "slot" / "tiny.gguf"
    calls: list[tuple[str, str | None]] = []

    def opener(req, timeout=None):
        have = req.get_header("Range")
        calls.append((req.full_url, have))
        assert req.full_url == url
        assert str(req.get_header("User-agent") or req.get_header("User-Agent") or "").startswith("video-buddy/")
        if not have:
            return _Body(
                body,
                status=200,
                headers={"Content-Length": str(len(body))},
                url=url,
                limit=4,
            )
        assert have == "bytes=4-"
        return _Body(
            body[4:],
            status=206,
            headers={"Content-Range": f"bytes 4-{len(body) - 1}/{len(body)}", "Content-Length": str(len(body) - 4)},
            url=url,
        )

    with pytest.raises(RuntimeError, match="4 bytes"):
        llm_gguf.download_resolve(url, dest, len(body), sha256=digest, opener=opener, progress=lambda _m: None)
    partial = dest.with_name(dest.name + ".partial")
    assert partial.read_bytes() == b"abcd"
    assert not dest.exists()

    got = llm_gguf.download_resolve(url, dest, len(body), sha256=digest, opener=opener, progress=lambda _m: None)
    assert got == dest
    assert dest.read_bytes() == body
    assert not partial.exists()
    assert calls == [(url, None), (url, "bytes=4-")]

    def explode(req, timeout=None):
        raise AssertionError("second install must not fetch")

    again = llm_gguf.download_resolve(url, dest, len(body), sha256=digest, opener=explode, progress=lambda _m: None)
    assert again == dest


def test_download_resolve_refuses_a_different_length_and_a_bad_sha(tmp_path):
    url = "https://example.invalid/tiny.gguf"
    dest = tmp_path / "tiny.gguf"
    body = b"abcdefghij"

    def wrong_length(req, timeout=None):
        return _Body(body, status=200, headers={"Content-Length": "99"}, url=url)

    with pytest.raises(RuntimeError, match="declared 99 bytes"):
        llm_gguf.download_resolve(url, dest, len(body), opener=wrong_length, progress=lambda _m: None)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".partial").exists()

    def wrong_hash(req, timeout=None):
        return _Body(body, status=200, headers={"Content-Length": str(len(body))}, url=url)

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        llm_gguf.download_resolve(
            url,
            dest,
            len(body),
            sha256="0" * 64,
            opener=wrong_hash,
            progress=lambda _m: None,
        )
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".partial").exists()


def test_download_resolve_restarts_after_range_not_satisfiable(tmp_path):
    url = "https://example.invalid/tiny.gguf"
    dest = tmp_path / "tiny.gguf"
    partial = dest.with_name(dest.name + ".partial")
    partial.write_bytes(b"abcd")
    body = b"abcdefghij"
    calls: list[str | None] = []

    def opener(req, timeout=None):
        have = req.get_header("Range")
        calls.append(have)
        if have:
            return _Body(b"", status=416, headers={}, url=url)
        return _Body(body, status=200, headers={"Content-Length": str(len(body))}, url=url)

    got = llm_gguf.download_resolve(url, dest, len(body), opener=opener, progress=lambda _m: None)
    assert got.read_bytes() == body
    assert calls == ["bytes=4-", None]
    assert not partial.exists()


def test_download_resolve_refuses_a_non_https_redirect(tmp_path):
    url = "https://example.invalid/tiny.gguf"
    dest = tmp_path / "tiny.gguf"

    def opener(req, timeout=None):
        return _Body(b"abcdefghij", status=200, headers={"Content-Length": "10"}, url="http://evil.example/file")

    with pytest.raises(RuntimeError, match="non-HTTPS"):
        llm_gguf.download_resolve(url, dest, 10, opener=opener, progress=lambda _m: None)
    assert not dest.exists()


def test_download_failure_keeps_the_link_and_does_not_pin_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# LLM_PROVIDER=ollama\nFOO=1\n", encoding="utf-8")
    monkeypatch.setattr(llm_gguf, "env_file", lambda: env_file)

    def boom(spec, dest, progress=print):
        raise RuntimeError("network down")

    monkeypatch.setattr(llm_gguf, "download_into", boom)
    from master_agent import automatic_install as ai

    res = ai.step_llm_models(True, skip=False, fetch=True)
    assert res.status == ai.FAILED and res.required
    text = "\n".join(res.lines)
    assert BRAIN_URL in text and EMBED_URL in text
    assert "WARNING: LOCAL LLM IS NOT READY" in text
    assert "not changed (download failed)" in text
    assert "LLM_PROVIDER=llamacpp" not in env_file.read_text(encoding="utf-8")
    notes = (tmp_path / "state" / "LAUNCH_NOTES.txt").read_text(encoding="utf-8")
    assert BRAIN_URL in notes and "network down" in notes


def test_light_install_warns_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(llm_gguf, "download_into", lambda *a, **k: pytest.fail("downloaded"))
    from master_agent import automatic_install as ai

    res = ai.step_llm_models(True, skip=False, fetch=False)
    assert res.status == ai.FAILED and not res.required
    text = "\n".join(res.lines)
    assert "light install does not download" in text
    assert BRAIN_URL in text
    assert not (tmp_path / "models").exists()


def test_ensure_provider_sets_llamacpp_and_leaves_ollama(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    missing = tmp_path / "fresh.env"
    missing.write_text("FOO=1\n", encoding="utf-8")
    monkeypatch.setattr(llm_gguf, "env_file", lambda: missing)
    value, action = llm_gguf.ensure_provider(True)
    assert (value, action) == ("llamacpp", "set")
    assert missing.read_text(encoding="utf-8").endswith("LLM_PROVIDER=llamacpp\n")
    assert config.LLM_PROVIDER == "llamacpp"

    pinned = tmp_path / "pinned.env"
    pinned.write_text('LLM_PROVIDER=ollama\n', encoding="utf-8")
    monkeypatch.setattr(llm_gguf, "env_file", lambda: pinned)
    assert llm_gguf.ensure_provider(True) == ("ollama", "unchanged")
    assert pinned.read_text(encoding="utf-8") == "LLM_PROVIDER=ollama\n"

    brand_new = tmp_path / "brand-new.env"
    monkeypatch.setattr(llm_gguf, "env_file", lambda: brand_new)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_gguf.ensure_provider(False) == ("llamacpp", "planned")
    assert not brand_new.exists()
    assert llm_gguf.ensure_provider(True) == ("llamacpp", "set")
    assert brand_new.read_text(encoding="utf-8") == "LLM_PROVIDER=llamacpp\n"

    commented = tmp_path / "commented.env"
    commented.write_text("# LLM_PROVIDER=ollama\n", encoding="utf-8")
    monkeypatch.setattr(llm_gguf, "env_file", lambda: commented)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_gguf.ensure_provider(True) == ("llamacpp", "set")
    text = commented.read_text(encoding="utf-8")
    assert text.startswith("# LLM_PROVIDER=ollama\n")
    assert text.endswith("LLM_PROVIDER=llamacpp\n")

    explicit_env = tmp_path / "from-env.env"
    explicit_env.write_text("FOO=1\n", encoding="utf-8")
    monkeypatch.setattr(llm_gguf, "env_file", lambda: explicit_env)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert llm_gguf.ensure_provider(True) == ("ollama", "unchanged")
    assert explicit_env.read_text(encoding="utf-8") == "FOO=1\n"
    assert os_environ_untouched("ollama")


def os_environ_untouched(value: str) -> bool:
    import os

    return os.environ.get("LLM_PROVIDER") == value


def test_preset_wires_model_mmproj_and_embedding(tmp_path, monkeypatch):
    models = tmp_path / "models"
    state = tmp_path / "state"
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "STATE_DIR", state)
    monkeypatch.setattr(config, "LLAMACPP_MODEL", "qwen3-vl-heretic")
    monkeypatch.setattr(config, "KB_EMBED_MODEL", "nomic-embed-text")
    brain = models / "qwen3-vl-heretic" / "my-brain.gguf"
    mmproj = models / "qwen3-vl-heretic" / "mmproj-custom.gguf"
    embed = models / "nomic-embed-text" / "my-embed.gguf"
    brain.parent.mkdir(parents=True)
    embed.parent.mkdir(parents=True)
    brain.write_bytes(b"brain")
    mmproj.write_bytes(b"proj")
    embed.write_bytes(b"embed")
    path = llm_gguf.write_preset()
    text = path.read_text(encoding="utf-8")
    assert f"model = {brain.resolve()}" in text
    assert f"mmproj = {mmproj.resolve()}" in text
    assert "[nomic-embed-text]" in text and "embedding = true" in text
    argv = srv.server_argv("llama-server", host="127.0.0.1", port=8080, models_dir=models)
    assert argv[argv.index("--models-preset") + 1] == str(path)

    row = llm_gguf.doctor_row()
    assert row["ok"] is True
    assert "vision mmproj: mmproj-custom.gguf" in row["detail"]
    assert row["fix"] == ""


def _touch(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _vision_pair(folder: Path, model_name: str = "model.Q4_K_M.gguf") -> tuple[Path, Path]:
    model = _touch(folder / model_name, b"brain-bytes")
    proj = _touch(folder / "mmproj-Q8_0.gguf", b"proj-bytes")
    return model, proj


def _ollama_embed(root: Path, payload: bytes = b"GGUF" + b"\0" * 16) -> Path:
    import json

    blob = _touch(root / "blobs" / "sha256-abc123", payload)
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / "nomic-embed-text" / "latest"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:abc123",
                        "size": len(payload),
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.license",
                        "digest": "sha256:not-a-model",
                        "size": 12,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return blob


def _use_models(tmp_path, monkeypatch):
    models = tmp_path / "models"
    state = tmp_path / "state"
    models.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "STATE_DIR", state)
    monkeypatch.setattr(config, "LLAMACPP_MODEL", "qwen3-vl-heretic")
    monkeypatch.setattr(config, "KB_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=1\n", encoding="utf-8")
    monkeypatch.setattr(llm_gguf, "env_file", lambda: env_file)
    return models, state


def test_prefers_gguf_vl_heretic_pair_and_skips_4b_download(tmp_path, monkeypatch):
    models, state = _use_models(tmp_path, monkeypatch)
    model, proj = _vision_pair(models / "gguf_vl_heretic")
    other_model, _other_proj = _vision_pair(models / "other_vl", model_name="other.gguf")
    embed = _touch(models / "nomic-embed-text" / "kept-embed.gguf", b"embed-bytes")
    before = {path: path.read_bytes() for path in (model, proj, other_model, embed)}
    monkeypatch.setattr(llm_gguf, "download_into", lambda *a, **k: pytest.fail("downloaded"))

    brain = llm_gguf.install_slot(llm_gguf.LLM_SLOTS["qwen3-vl-heretic"], apply=True, fetch=True)
    assert brain.status == "present"
    assert str(model) in "\n".join(brain.lines)
    assert "vision: enabled" in "\n".join(brain.lines)
    assert llm_gguf.bytes_still_needed(models, slots=[llm_gguf.LLM_SLOTS["qwen3-vl-heretic"]]) == 0

    from master_agent import automatic_install as ai

    res = ai.step_llm_models(True, skip=False, fetch=True)
    text = "\n".join(res.lines)
    assert "vision: enabled" in text
    assert str(model) in text and str(proj) in text
    assert "4B file is not downloaded" in text
    preset = (state / "llamacpp-models.ini").read_text(encoding="utf-8")
    assert "[qwen3-vl-heretic]" in preset
    assert f"model = {model.resolve()}" in preset
    assert f"mmproj = {proj.resolve()}" in preset
    assert "other_vl" not in preset
    assert not (models / "qwen3-vl-heretic").exists()
    assert {path: path.read_bytes() for path in before} == before

    row = llm_gguf.doctor_row()
    assert row["ok"] is True
    assert "vision: enabled" in row["detail"]
    assert f"brain: {model}" in row["detail"]
    assert f"projector: {proj}" in row["detail"]
    assert f"embed: {embed}" in row["detail"]
    assert "vision mmproj: mmproj-Q8_0.gguf" in row["detail"]
    argv = srv.server_argv("llama-server", host="127.0.0.1", port=8080, models_dir=models)
    assert argv[argv.index("--models-preset") + 1] == str(state / "llamacpp-models.ini")


def test_other_folder_pair_used_when_preferred_dir_is_absent(tmp_path, monkeypatch):
    models, _state = _use_models(tmp_path, monkeypatch)
    _touch(models / "diffusion_models" / "ltx.gguf", b"not-a-brain")
    model, proj = _vision_pair(models / "custom_vl", model_name="weights.gguf")
    found = llm_gguf.discover_local_vision_pair(models)
    assert found is not None
    assert found.model == model and found.mmproj == proj


def test_nested_pair_and_unprefixed_projector_are_not_adopted(tmp_path):
    nested = tmp_path / "gguf_vl_heretic" / "nested"
    _vision_pair(nested)
    assert llm_gguf.discover_local_vision_pair(tmp_path) is None
    odd = tmp_path / "odd_vl"
    _touch(odd / "weights.gguf", b"m")
    _touch(odd / "weights.mmproj-f16.gguf", b"p")
    assert llm_gguf.discover_local_vision_pair(tmp_path) is None


def test_doctor_warns_when_brain_gguf_present_but_projector_missing(tmp_path, monkeypatch, capsys):
    """A brain file with no mmproj is a visible WARN, not a quiet text-only OK."""
    models, _state = _use_models(tmp_path, monkeypatch)
    brain = _touch(models / "qwen3-vl-heretic" / "custom-brain.gguf", b"4b")
    embed = _touch(models / "nomic-embed-text" / "nomic-embed-text-v1.5.f16.gguf", b"e")
    row = llm_gguf.doctor_row()
    assert row["ok"] is False
    assert "projector MISSING" in row["detail"]
    assert f"brain GGUF is present ({brain.name})" in row["detail"]
    assert f"brain: {brain}" in row["detail"]
    assert f"embed: {embed}" in row["detail"]
    assert "vision: text-only" in row["detail"]
    assert MMPROJ_URL in row["fix"]
    assert BRAIN_URL not in row["fix"]
    assert EMBED_URL not in row["fix"]
    assert "projector: none yet" not in row["detail"]

    from master_agent.setup import classify_row, print_report

    assert classify_row(row) == "WARN"
    assert print_report([row]) == 0
    out = capsys.readouterr().out
    assert "WARN  llm-models" in out
    assert "projector MISSING" in out
    assert "the brain GGUF is on disk but the vision projector is not" in out
    assert "not safe to ignore" in out


def _sparse_file(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


def test_present_brain_fetches_only_the_projector_into_the_preset(tmp_path, monkeypatch):
    """Brain and embed already at full size: only the Qwen mmproj is fetched, then pinned."""
    models, state = _use_models(tmp_path, monkeypatch)
    brain = _sparse_file(models / "qwen3-vl-heretic" / llm_gguf.BRAIN_FILENAME, llm_gguf.BRAIN_SIZE_BYTES)
    embed = _sparse_file(models / "nomic-embed-text" / llm_gguf.EMBED_FILENAME, llm_gguf.EMBED_SIZE_BYTES)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "no-ollama"))
    calls: list[str] = []
    monkeypatch.setattr(llm_gguf, "sha256_file", _pinned_digest)
    monkeypatch.setattr(
        llm_gguf.urllib.request,
        "urlopen",
        _sparse_resolve(calls, models, allow={MMPROJ_URL}),
    )
    from master_agent import automatic_install as ai

    res = ai.step_llm_models(True, skip=False, fetch=True)
    assert calls == [MMPROJ_URL]
    assert "downloaded: qwen3-vl-heretic" in res.detail
    assert "present: nomic-embed-text" in res.detail
    proj = models / "qwen3-vl-heretic" / llm_gguf.MMPROJ_FILENAME
    assert proj.is_file() and proj.stat().st_size == llm_gguf.MMPROJ_SIZE_BYTES
    preset = (state / "llamacpp-models.ini").read_text(encoding="utf-8")
    assert f"model = {brain.resolve()}" in preset
    assert f"mmproj = {proj.resolve()}" in preset
    assert f"model = {embed.resolve()}" in preset
    row = llm_gguf.doctor_row()
    assert row["ok"] is True
    assert f"vision mmproj: {proj.name}" in row["detail"]
    assert "projector MISSING" not in row["detail"]


def test_ollama_embed_blob_is_referenced_read_only(tmp_path, monkeypatch):
    models, state = _use_models(tmp_path, monkeypatch)
    _vision_pair(models / "gguf_vl_heretic")
    ollama = tmp_path / "ollama"
    blob = _ollama_embed(ollama)
    monkeypatch.setenv("OLLAMA_MODELS", str(ollama))
    snapshot = blob.read_bytes()
    inode = blob.stat().st_ino
    monkeypatch.setattr(llm_gguf, "download_into", lambda *a, **k: pytest.fail("downloaded"))

    embed = llm_gguf.install_slot(llm_gguf.LLM_SLOTS["nomic-embed-text"], apply=True, fetch=True)
    assert embed.status == "present"
    assert str(blob) in "\n".join(embed.lines)
    assert "not copied" in "\n".join(embed.lines)
    assert llm_gguf.bytes_still_needed(models) == 0
    assert not (models / "nomic-embed-text").exists()
    assert blob.read_bytes() == snapshot and blob.stat().st_ino == inode

    path = llm_gguf.write_preset(models)
    text = path.read_text(encoding="utf-8")
    assert f"model = {blob.resolve()}" in text
    assert "embedding = true" in text
    row = llm_gguf.doctor_row()
    assert row["ok"] is True
    assert "vision: enabled" in row["detail"]
    assert f"embed: {blob}" in row["detail"]
    assert "read-only" in row["detail"]
    assert (state / "llamacpp-models.ini").is_file()


def test_local_embed_wins_over_ollama_blob(tmp_path, monkeypatch):
    models, _state = _use_models(tmp_path, monkeypatch)
    local = _touch(models / "nomic-embed-text" / "nomic-embed-text-v1.5.f16.gguf", b"local-embed")
    blob = _ollama_embed(tmp_path / "ollama")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama"))
    ready = llm_gguf.file_ready(
        llm_gguf.LLM_SLOTS["nomic-embed-text"],
        llm_gguf.LLM_SLOTS["nomic-embed-text"].files[0],
        models,
    )
    assert ready == local
    assert blob.read_bytes().startswith(b"GGUF")


def test_non_gguf_ollama_blob_is_ignored(tmp_path, monkeypatch):
    models, _state = _use_models(tmp_path, monkeypatch)
    blob = _ollama_embed(tmp_path / "ollama", payload=b"NOT-A-GGUF-FILE!!")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama"))
    assert llm_gguf.ollama_embed_blob("nomic-embed-text") is None
    slot = llm_gguf.LLM_SLOTS["nomic-embed-text"]
    assert llm_gguf.file_ready(slot, slot.files[0], models) is None
    assert llm_gguf.bytes_still_needed(models, slots=[slot]) == llm_gguf.EMBED_SIZE_BYTES
    assert blob.is_file()


def test_download_into_refuses_to_touch_an_ollama_blob(tmp_path, monkeypatch):
    blob = _ollama_embed(tmp_path / "ollama")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama"))
    before = blob.read_bytes()
    with pytest.raises(RuntimeError, match="never modified"):
        llm_gguf.download_into(llm_gguf.LLM_SLOTS["nomic-embed-text"].files[0], blob)
    assert blob.read_bytes() == before


def test_doctor_row_warns_when_the_files_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "LLAMACPP_MODEL", "qwen3-vl-heretic")
    monkeypatch.setattr(config, "KB_EMBED_MODEL", "nomic-embed-text")
    row = llm_gguf.doctor_row()
    assert row["ok"] is False
    assert "MISSING: qwen3-vl-heretic, nomic-embed-text" in row["detail"]
    assert "projector: none yet" in row["detail"]
    assert "projector MISSING" not in row["detail"]
    assert "brain GGUF is present" not in row["detail"]
    assert BRAIN_URL in row["fix"] and MMPROJ_URL in row["fix"] and EMBED_URL in row["fix"]
    assert "ollama" not in (str(row["detail"]) + str(row["fix"])).lower()


def _pinned_digest(path: Path) -> str:
    size = path.stat().st_size
    if llm_gguf.BRAIN_FILENAME in path.name:
        assert size == llm_gguf.BRAIN_SIZE_BYTES
        return llm_gguf.BRAIN_SHA256
    if llm_gguf.EMBED_FILENAME in path.name:
        assert size == llm_gguf.EMBED_SIZE_BYTES
        return llm_gguf.EMBED_SHA256
    if llm_gguf.MMPROJ_FILENAME in path.name:
        assert size == llm_gguf.MMPROJ_SIZE_BYTES
        return llm_gguf.MMPROJ_SHA256
    raise AssertionError(path.name)


def _sparse_resolve(calls: list, models: Path, *, allow: set[str]):
    """Mock urlopen: record the resolve URL and sparse-fill the .partial to the pinned size."""

    def opener(req, timeout=None):
        url = req.full_url
        calls.append(url)
        assert url in allow, url
        assert req.get_header("Range") is None
        if url == BRAIN_URL:
            partial = models / "qwen3-vl-heretic" / (llm_gguf.BRAIN_FILENAME + ".partial")
            size = llm_gguf.BRAIN_SIZE_BYTES
        elif url == MMPROJ_URL:
            partial = models / "qwen3-vl-heretic" / (llm_gguf.MMPROJ_FILENAME + ".partial")
            size = llm_gguf.MMPROJ_SIZE_BYTES
        else:
            partial = models / "nomic-embed-text" / (llm_gguf.EMBED_FILENAME + ".partial")
            size = llm_gguf.EMBED_SIZE_BYTES

        class _Sparse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def geturl(self):
                return url

            def getheader(self, name, default=None):
                if str(name).lower() == "content-length":
                    return str(size)
                return default

            def read(self, n=-1):
                with open(partial, "r+b") as extra:
                    extra.truncate(size)
                return b""

        return _Sparse()

    return opener


def test_nothing_local_fetches_brain_projector_and_embed(tmp_path, monkeypatch):
    """Empty MODELS_DIR: brain, Qwen mmproj, and embed are fetched into MODELS_DIR."""
    models, state = _use_models(tmp_path, monkeypatch)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "no-ollama"))
    env_file = tmp_path / ".env"
    env_file.unlink()
    calls: list[str] = []
    monkeypatch.setattr(llm_gguf, "sha256_file", _pinned_digest)
    monkeypatch.setattr(
        llm_gguf.urllib.request,
        "urlopen",
        _sparse_resolve(calls, models, allow={BRAIN_URL, MMPROJ_URL, EMBED_URL}),
    )
    from master_agent import automatic_install as ai

    res = ai.step_llm_models(True, skip=False, fetch=True)
    assert res.status == ai.OK
    assert res.detail == "downloaded: qwen3-vl-heretic, nomic-embed-text"
    assert calls == [BRAIN_URL, MMPROJ_URL, EMBED_URL]
    brain = models / "qwen3-vl-heretic" / llm_gguf.BRAIN_FILENAME
    proj = models / "qwen3-vl-heretic" / llm_gguf.MMPROJ_FILENAME
    embed = models / "nomic-embed-text" / llm_gguf.EMBED_FILENAME
    assert brain.is_file() and brain.stat().st_size == llm_gguf.BRAIN_SIZE_BYTES
    assert proj.is_file() and proj.stat().st_size == llm_gguf.MMPROJ_SIZE_BYTES
    assert embed.is_file() and embed.stat().st_size == llm_gguf.EMBED_SIZE_BYTES
    assert brain.stat().st_blocks == 0 and proj.stat().st_blocks == 0 and embed.stat().st_blocks == 0
    assert not brain.with_name(brain.name + ".partial").exists()
    assert not proj.with_name(proj.name + ".partial").exists()
    assert not embed.with_name(embed.name + ".partial").exists()
    assert env_file.read_text(encoding="utf-8") == "LLM_PROVIDER=llamacpp\n"
    preset = (state / "llamacpp-models.ini").read_text(encoding="utf-8")
    assert f"model = {brain.resolve()}" in preset
    assert f"mmproj = {proj.resolve()}" in preset
    assert f"model = {embed.resolve()}" in preset
    assert "embedding = true" in preset
    row = llm_gguf.doctor_row()
    assert row["ok"] is True
    assert f"vision mmproj: {llm_gguf.MMPROJ_FILENAME}" in row["detail"]

    def explode(*_a, **_k):
        raise AssertionError("already on disk; must not fetch again")

    monkeypatch.setattr(llm_gguf.urllib.request, "urlopen", explode)
    again = ai.step_llm_models(True, skip=False, fetch=True)
    assert again.status == ai.SKIPPED
    assert "present: qwen3-vl-heretic, nomic-embed-text" in again.detail


def test_local_vision_pair_and_ollama_blob_skip_both_fetches(tmp_path, monkeypatch):
    models, _state = _use_models(tmp_path, monkeypatch)
    model, proj = _vision_pair(models / "gguf_vl_heretic")
    blob = _ollama_embed(tmp_path / "ollama")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama"))
    before = (model.read_bytes(), proj.read_bytes(), blob.read_bytes(), blob.stat().st_ino)

    def explode(*_a, **_k):
        raise AssertionError("local pair and Ollama blob must satisfy both slots")

    monkeypatch.setattr(llm_gguf.urllib.request, "urlopen", explode)
    from master_agent import automatic_install as ai

    res = ai.step_llm_models(True, skip=False, fetch=True)
    assert res.status == ai.SKIPPED
    assert res.detail == "present: qwen3-vl-heretic, nomic-embed-text"
    text = "\n".join(res.lines)
    assert "4B file is not downloaded" in text
    assert str(blob) in text
    assert not (models / "qwen3-vl-heretic").exists()
    assert not (models / "nomic-embed-text").exists()
    assert (model.read_bytes(), proj.read_bytes(), blob.read_bytes(), blob.stat().st_ino) == before


def test_only_the_missing_slot_is_fetched(tmp_path, monkeypatch):
    """A local vision pair skips the 4B URL. An Ollama embed blob skips the 84 MB URL."""
    models, _state = _use_models(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_gguf, "sha256_file", _pinned_digest)
    from master_agent import automatic_install as ai

    _vision_pair(models / "gguf_vl_heretic")
    calls: list[str] = []
    monkeypatch.setattr(
        llm_gguf.urllib.request,
        "urlopen",
        _sparse_resolve(calls, models, allow={EMBED_URL}),
    )
    res = ai.step_llm_models(True, skip=False, fetch=True)
    assert calls == [EMBED_URL]
    assert "downloaded: nomic-embed-text" in res.detail
    assert "present: qwen3-vl-heretic" in res.detail
    assert not (models / "qwen3-vl-heretic" / llm_gguf.BRAIN_FILENAME).exists()

    # Fresh tree: brain must be fetched, embed blob must not.
    models2 = tmp_path / "models2"
    state2 = tmp_path / "state2"
    monkeypatch.setattr(config, "MODELS_DIR", models2)
    monkeypatch.setattr(config, "STATE_DIR", state2)
    blob = _ollama_embed(tmp_path / "ollama2")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama2"))
    calls.clear()
    monkeypatch.setattr(
        llm_gguf.urllib.request,
        "urlopen",
        _sparse_resolve(calls, models2, allow={BRAIN_URL, MMPROJ_URL}),
    )
    res = ai.step_llm_models(True, skip=False, fetch=True)
    assert calls == [BRAIN_URL, MMPROJ_URL]
    assert "downloaded: qwen3-vl-heretic" in res.detail
    assert "present: nomic-embed-text" in res.detail
    assert not (models2 / "nomic-embed-text").exists()
    assert blob.read_bytes().startswith(b"GGUF")


def test_explicit_provider_survives_a_successful_fetch(tmp_path, monkeypatch):
    models, _state = _use_models(tmp_path, monkeypatch)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=ollama\n", encoding="utf-8")
    monkeypatch.setattr(llm_gguf, "sha256_file", _pinned_digest)
    monkeypatch.setattr(
        llm_gguf.urllib.request,
        "urlopen",
        _sparse_resolve([], models, allow={BRAIN_URL, MMPROJ_URL, EMBED_URL}),
    )
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "no-ollama"))
    from master_agent import automatic_install as ai

    res = ai.step_llm_models(True, skip=False, fetch=True)
    assert res.status == ai.OK
    assert env_file.read_text(encoding="utf-8") == "LLM_PROVIDER=ollama\n"
    assert "unchanged" in "\n".join(res.lines)
