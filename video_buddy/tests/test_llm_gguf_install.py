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


def test_manifest_pins_verified_filenames_and_no_default_mmproj():
    brain = llm_gguf.LLM_SLOTS["qwen3-vl-heretic"]
    embed = llm_gguf.LLM_SLOTS["nomic-embed-text"]
    assert brain.files[0].filename == "qwen3-vl-4b-heretic-Q4_K_M.gguf"
    assert brain.files[0].repo_id == "DreamFast/Qwen3-VL-4b-Heretic-GGUF"
    assert brain.files[0].size_bytes == 2_497_281_696
    assert brain.files[0].url == BRAIN_URL
    assert brain.publishes_mmproj is False
    assert all("mmproj" not in spec.filename.lower() for spec in brain.files)
    assert embed.files[0].filename == "nomic-embed-text-v1.5.Q4_K_M.gguf"
    assert embed.files[0].repo_id == "nomic-ai/nomic-embed-text-v1.5-GGUF"
    assert embed.files[0].size_bytes == 84_106_624
    assert embed.files[0].url == EMBED_URL
    assert llm_gguf.bytes_still_needed(root=Path("/no/such/models"), slots=[brain, embed]) == (
        2_497_281_696 + 84_106_624
    )
    repos = {alt.repo_id for alt in llm_gguf.OPTIONAL_BRAIN_ALTERNATIVES}
    assert repos == {
        "SergiusFlavius/Qwen3-VL-8B-Instruct-heretic-gguf",
        "mradermacher/Qwen-3-VL-8B-Instruct-heretic-GGUF",
    }
    note = llm_gguf.vision_note()
    assert "no mmproj" in note
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


def test_download_into_unlinks_short_dest_and_copies_the_full_file(tmp_path, monkeypatch):
    spec = _tiny().files[0]
    dest = tmp_path / "slot" / spec.filename
    dest.parent.mkdir()
    dest.write_bytes(b"x")
    cache = tmp_path / "cache.gguf"
    cache.write_bytes(b"12345678")

    def fake(*, repo_id, repo_filename, dest, progress):
        assert not Path(dest).exists()
        assert repo_id == "org/repo"
        return cache

    monkeypatch.setattr("master_agent.models.download.download_hub_file", fake)
    got = llm_gguf.download_into(spec, dest, progress=lambda _m: None)
    assert got == dest
    assert dest.read_bytes() == b"12345678"


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


def test_doctor_row_warns_when_the_files_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "LLAMACPP_MODEL", "qwen3-vl-heretic")
    monkeypatch.setattr(config, "KB_EMBED_MODEL", "nomic-embed-text")
    row = llm_gguf.doctor_row()
    assert row["ok"] is False
    assert "MISSING: qwen3-vl-heretic, nomic-embed-text" in row["detail"]
    assert "vision mmproj: not in the default repo" in row["detail"]
    assert BRAIN_URL in row["fix"] and EMBED_URL in row["fix"]
    assert "ollama" not in (str(row["detail"]) + str(row["fix"])).lower()
