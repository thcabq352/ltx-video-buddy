"""HeartMuLa pack. No GPU, no network, no weight downloads.

Run: PYTHONPATH=video_buddy pytest -q video_buddy/tests/test_heartmula.py
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.__main__ import cmd_download_models, cmd_heartmula
from master_agent.comfy.attach import AttachError, load_attach_recipe
from master_agent.comfy.capabilities import CAPABILITY_CATALOG
from master_agent.comfy.graph_ops import OPTIONAL_NODE_CLASS_TYPES, is_optional_node
from master_agent.heartmula.config import (
    HeartMuLaConfigError,
    all_slots,
    mula_dtype_name,
)
from master_agent.heartmula.doctor import (
    cmd_download_heartmula,
    download_missing_slots,
    format_slot_report,
    heartmula_doctor_rows,
)
from master_agent.heartmula.generate import (
    HeartMuLaUnavailable,
    generate_track,
    load_heartlib,
    plan_generate,
)
from master_agent.heartmula.transcribe import chunks_to_words, transcribe_audio, write_words
from master_agent.heartmula.wire import materialize_track
from master_agent.models.vram_policy import family_for_slug, format_vram_table
from master_agent.orchestrator.lipdub import load_words
from master_agent.orchestrator.state import RunState
from master_agent.provenance import (
    CLIP_PROVENANCE_SCHEMA,
    build_clip_provenance,
    missing_required,
)
from master_agent.setup import print_report, snapshot


def _clear_repo_env(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("HEARTMULA_MODELS_DIR", str(root))
    for name in (
        "HEARTMULA_MULA_REPO",
        "HEARTMULA_CODEC_REPO",
        "HEARTMULA_TRANSCRIPTOR_REPO",
        "HEARTMULA_GEN_REPO",
        "HEARTMULA_VERSION",
        "HEARTMULA_DTYPE",
        "HEARTMULA_CODEC_DTYPE",
        "HEARTMULA_LYRICS",
        "HEARTMULA_TAGS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_import_does_not_load_heartlib():
    sys.modules.pop("heartlib", None)
    import master_agent
    import master_agent.heartmula

    assert master_agent is not None
    assert "heartlib" not in sys.modules


def test_missing_heartlib_is_a_clear_error():
    sys.modules.pop("heartlib", None)
    with pytest.raises(HeartMuLaUnavailable, match="heartlib is not installed"):
        load_heartlib()


def test_doctor_rows_are_informational(capsys):
    names = {row["name"] for row in heartmula_doctor_rows()}
    assert names == {"heartlib", "heartmula-weights", "heartmula-comfy"}
    snap_names = {row["name"] for row in snapshot()}
    assert names <= snap_names
    code = print_report(heartmula_doctor_rows())
    assert code == 0
    text = capsys.readouterr().out
    assert "heartmula-weights" in text
    assert "All checked dependencies are ready." in text


def test_generate_dry_run_writes_nothing(tmp_path, capsys):
    sys.modules.pop("heartlib", None)
    out = tmp_path / "track.wav"
    code = cmd_heartmula(
        Namespace(
            heartmula_command="generate",
            lyrics="[Verse]\nlocal night",
            tags="piano,happy",
            duration=30,
            seed=None,
            out=str(out),
            topk=50,
            temperature=1.0,
            cfg_scale=1.5,
            dry_run=True,
        )
    )
    assert code == 0
    assert not out.exists()
    assert "heartlib" not in sys.modules
    assert "dry-run" in capsys.readouterr().out


def test_transcribe_dry_run_writes_nothing(tmp_path, capsys):
    sys.modules.pop("heartlib", None)
    out = tmp_path / "words.json"
    code = cmd_heartmula(
        Namespace(
            heartmula_command="transcribe",
            audio=str(tmp_path / "vocals.wav"),
            out=str(out),
            dry_run=True,
        )
    )
    assert code == 0
    assert not out.exists()
    assert "heartlib" not in sys.modules
    assert "dry-run" in capsys.readouterr().out


def test_download_lists_without_fetch(tmp_path, monkeypatch, capsys):
    _clear_repo_env(monkeypatch, tmp_path)

    def boom(*_a, **_k):
        raise AssertionError("hf_hub_download must not run")

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        boom,
        raising=False,
    )
    code = cmd_download_heartmula(
        Namespace(scan_only=False, use_existing=False, yes=False, download=False)
    )
    assert code == 2
    text = capsys.readouterr().out
    assert "tokenizer.json" in text
    assert "model-00001-of-00004.safetensors" in text
    assert "model-00001-of-00002.safetensors" in text
    assert "HeartCodec-oss-20260123" in text
    assert "Nothing downloaded" in text
    assert "28GB" in text
    assert "401" in text
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_download_models_flag_returns_before_other_packs(tmp_path, monkeypatch, capsys):
    _clear_repo_env(monkeypatch, tmp_path)
    code = cmd_download_models(
        Namespace(
            heartmula=True,
            ltx25=True,
            h3=False,
            scan_only=False,
            use_existing=False,
            yes=False,
            download=False,
        )
    )
    assert code == 2
    text = capsys.readouterr().out
    assert "HeartCodec-oss-20260123" in text
    assert "Nothing downloaded" in text


def test_yes_copies_with_mocked_hub(tmp_path, monkeypatch):
    root = tmp_path / "models"
    _clear_repo_env(monkeypatch, root)
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"not-a-real-weight")
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return str(blob)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    written = download_missing_slots(yes=True, progress=lambda _m: None, root=root)
    assert written
    assert (root / "tokenizer.json").read_bytes() == b"not-a-real-weight"
    assert (root / "HeartCodec-oss" / "model-00001-of-00002.safetensors").is_file()
    repos = {repo for repo, _name in calls}
    assert "HeartMuLa/HeartCodec-oss-20260123" in repos
    assert "HeartMuLa/HeartCodec-oss" not in repos
    with pytest.raises(RuntimeError, match="without --yes"):
        download_missing_slots(yes=False, root=root)


def test_words_round_trip_matches_lipdub(tmp_path):
    words = chunks_to_words(
        {"chunks": [{"text": " hello", "timestamp": (0.02, 0.40)}, {"text": "night", "timestamp": (0.40, 0.90)}]}
    )
    path = write_words(tmp_path / "words.json", words)
    loaded = load_words(str(path))
    assert [(w.text, w.start, w.end) for w in loaded] == [
        ("hello", 0.02, 0.40),
        ("night", 0.40, 0.90),
    ]
    listed = tmp_path / "list.json"
    listed.write_text(json.dumps([{"w": "hi", "s": 1, "e": 1.2}]), encoding="utf-8")
    assert load_words(str(listed))[0].text == "hi"
    assert chunks_to_words({"chunks": [{"text": "nope"}]}) == []


def test_mocked_generate_and_transcribe(tmp_path, monkeypatch):
    root = tmp_path / "models"
    _clear_repo_env(monkeypatch, root)
    seen: dict = {}

    class _Pipe:
        def __call__(self, payload, **kwargs):
            seen["payload"] = payload
            seen["kwargs"] = kwargs
            Path(kwargs["save_path"]).write_bytes(b"RIFF")

    class _Gen:
        @staticmethod
        def from_pretrained(*_a, **kw):
            seen["from"] = kw
            return _Pipe()

    result = generate_track(
        lyrics="[Verse]\nhello",
        tags="piano,happy",
        out=tmp_path / "track.wav",
        duration_s=8,
        seed=None,
        pipeline_cls=_Gen,
        require_weights=False,
    )
    assert result.wav.is_file()
    assert seen["payload"]["lyrics"].startswith("[Verse]")
    assert seen["kwargs"]["max_audio_length_ms"] == 8000
    assert seen["from"]["dtype"]["codec"] == "fp32"
    assert "heartlib" not in sys.modules

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")

    class _Tr:
        def __call__(self, path, **kwargs):
            seen["audio"] = path
            seen["whisper"] = kwargs
            return {"chunks": [{"text": "hello", "timestamp": (0.0, 0.5)}]}

    class _Trans:
        @staticmethod
        def from_pretrained(*_a, **_kw):
            return _Tr()

    transcribed = transcribe_audio(
        audio=audio,
        out=tmp_path / "out.json",
        pipeline_cls=_Trans,
        require_weights=False,
    )
    assert transcribed.words == [{"w": "hello", "s": 0.0, "e": 0.5}]
    assert load_words(str(transcribed.path))[0].text == "hello"
    assert seen["whisper"]["return_timestamps"] == "word"


def test_unknown_repo_and_nf4_are_refused(monkeypatch, tmp_path):
    _clear_repo_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HEARTMULA_CODEC_REPO", "HeartMuLa/HeartCodec-oss")
    with pytest.raises(HeartMuLaConfigError, match="HeartCodec-oss-20260123"):
        all_slots()
    monkeypatch.setenv("HEARTMULA_CODEC_REPO", "HeartMuLa/HeartCodec-oss-20260123")
    monkeypatch.setenv("HEARTMULA_DTYPE", "nf4")
    with pytest.raises(HeartMuLaConfigError, match="nf4"):
        mula_dtype_name()
    monkeypatch.delenv("HEARTMULA_DTYPE", raising=False)
    plan = plan_generate(
        lyrics="[Verse]\nline",
        tags="piano, happy",
        out=tmp_path / "x.wav",
        dry_run=True,
    )
    assert "without spaces" in (plan.get("note") or "")


def test_audio_wins_and_dry_run_silent_wav(tmp_path, monkeypatch):
    _clear_repo_env(monkeypatch, tmp_path)
    sys.modules.pop("heartlib", None)
    given = tmp_path / "given.wav"
    given.write_bytes(b"RIFF")
    won = materialize_track(
        audio=given,
        lyrics="[Verse]\nskip",
        tags="piano,happy",
        out_dir=tmp_path / "work",
        dry_run=False,
    )
    assert won.audio == given
    assert won.heartmula is None
    assert won.note and "not used" in won.note
    assert not (tmp_path / "work" / "heartmula.wav").exists()

    silent = materialize_track(
        audio=None,
        lyrics="[Verse]\nlocal",
        tags="piano,happy",
        out_dir=tmp_path / "dry",
        dry_run=True,
        duration_s=8,
    )
    assert silent.audio.is_file()
    assert silent.audio.stat().st_size > 0
    assert silent.heartmula["placeholder"] == "silent-wav"
    assert silent.heartmula["dry_run"] is True
    assert "heartlib" not in sys.modules


def test_provenance_keeps_schema_id():
    st = RunState(
        request="mv",
        prompt="mv",
        variant="ltx25_t2v_i2v",
        seed=1,
        heartmula={
            "lyrics": "[Chorus]\nhold",
            "tags": "piano,happy",
            "mula_repo": "HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
            "codec_repo": "HeartMuLa/HeartCodec-oss-20260123",
            "wav": "out/heartmula.wav",
            "transcribe_source": None,
            "words_path": None,
            "seed": 1,
        },
    )
    payload = build_clip_provenance(st)
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA == "buddy.clip.provenance/v1"
    assert missing_required(payload) == []
    block = payload["params"]["heartmula"]
    assert block["lyrics"].startswith("[Chorus]")
    assert block["codec_repo"].endswith("20260123")
    assert block["wav"] == "out/heartmula.wav"
    empty = build_clip_provenance(RunState(request="plain", prompt="plain", variant="base"))
    assert "heartmula" not in empty["params"]


def test_vram_and_capability_and_attach():
    assert family_for_slug("heartmula") == "heartmula"
    assert family_for_slug("heartcodec") == "heartmula"
    table = format_vram_table()
    assert "HeartMuLa" in table
    assert "sequential" in table
    cap = next(row for row in CAPABILITY_CATALOG if row.id == "heartmula")
    assert cap.class_types == ("HeartMuLa_Generate", "HeartMuLa_Transcribe")
    assert cap.surfaces == ()
    assert "HeartMuLa Music Generator" in cap.notes
    assert "HeartMuLa Lyrics Transcriber" in cap.notes
    assert "fdb53c4" in cap.notes
    assert "HeartMuLa_Generate" not in OPTIONAL_NODE_CLASS_TYPES
    assert "HeartMuLa_Transcribe" not in OPTIONAL_NODE_CLASS_TYPES
    assert not is_optional_node("HeartMuLa_Generate")
    assert not is_optional_node("HeartMuLa_Transcribe")

    with pytest.raises(AttachError, match="HeartMuLa_Generate"):
        load_attach_recipe(
            {
                "schema": "buddy.comfy.attach/v1",
                "required_nodes": ["HeartMuLa_NotRegistered"],
                "patches": [],
            }
        )
    ok = load_attach_recipe(
        {
            "schema": "buddy.comfy.attach/v1",
            "required_nodes": ["HeartMuLa_Generate", "HeartMuLa_Transcribe"],
            "patches": [],
        }
    )
    assert ok.required_nodes == ["HeartMuLa_Generate", "HeartMuLa_Transcribe"]
    with pytest.raises(AttachError, match="HeartMuLa_Transcribe"):
        load_attach_recipe(
            {
                "schema": "buddy.comfy.attach/v1",
                "patches": [{"class_type": "HeartMuLa_Extra"}],
            }
        )


def test_slot_report_names_codec_preference(tmp_path, monkeypatch):
    _clear_repo_env(monkeypatch, tmp_path)
    text = format_slot_report()
    assert "HeartMuLa/HeartCodec-oss-20260123" in text
    assert "HeartMuLa-RL-oss-3B-20260123" in text
    assert "Nothing downloaded" in text
