"""Mocked tests for the Flux LoRA stage (no subprocess/GPU/network).

Run: .venv/Scripts/python.exe -m pytest tests/test_lora.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from master_agent.config import FLUX_UNET, LORA_DEFAULTS
from master_agent.lora import config_gen as cg
from master_agent.lora import trainer as tr
from master_agent.lora.validate import retry_ladder


CHARACTER = {
    "name": "mira",
    "trigger_word": "zxc_mira",
    "appearance": "short red hair, green eyes",
    "shots": ["shot 0"],
    "scores": {},
    "dataset_count": 12,
    "created": "2026-07-30T00:00:00+00:00",
}


# --- config_gen -----------------------------------------------------------------


class TestConfigGen:
    def _render(self, tmp_path, monkeypatch, overrides=None):
        monkeypatch.setattr(cg, "CHARACTERS_DIR", tmp_path)
        return yaml.safe_load(cg.render_train_config(CHARACTER, overrides))

    def test_yaml_parses_with_expected_flags(self, tmp_path, monkeypatch):
        cfg = self._render(tmp_path, monkeypatch)
        assert cfg["job"] == "extension"
        proc = cfg["config"]["process"][0]
        assert proc["type"] == "sd_trainer"
        assert proc["trigger_word"] == "zxc_mira"
        assert proc["model"]["quantize"] is True
        assert proc["model"]["low_vram"] is True
        assert proc["model"]["is_flux"] is True
        assert proc["train"]["gradient_checkpointing"] is True
        assert proc["train"]["noise_scheduler"] == "flowmatch"
        assert proc["train"]["optimizer"] == "adamw8bit"
        assert proc["network"]["linear"] == LORA_DEFAULTS["rank"]
        assert proc["network"]["linear_alpha"] == LORA_DEFAULTS["alpha"]
        assert proc["train"]["steps"] == LORA_DEFAULTS["steps"]
        assert proc["save"]["save_every"] == LORA_DEFAULTS["save_every"]
        assert proc["sample"]["sampler"] == "flowmatch"
        assert cfg["meta"]["name"] == "mira"

    def test_dataset_path_absolute_forward_slashes(self, tmp_path, monkeypatch):
        cfg = self._render(tmp_path, monkeypatch)
        folder = cfg["config"]["process"][0]["datasets"][0]["folder_path"]
        assert "\\" not in folder
        assert folder.endswith("/mira/dataset")
        assert Path(folder).is_absolute()
        assert cfg["config"]["process"][0]["datasets"][0]["caption_ext"] == "txt"

    def test_resolution_override_applied(self, tmp_path, monkeypatch):
        cfg = self._render(tmp_path, monkeypatch, overrides={"resolution": [512, 768, 1024]})
        assert cfg["config"]["process"][0]["datasets"][0]["resolution"] == [512, 768, 1024]
        # untouched keys keep defaults
        assert cfg["config"]["process"][0]["train"]["steps"] == LORA_DEFAULTS["steps"]

    def test_rank_steps_lr_overrides(self, tmp_path, monkeypatch):
        cfg = self._render(
            tmp_path, monkeypatch, overrides={"rank": 32, "steps": 2000, "lr": 2e-4}
        )
        proc = cfg["config"]["process"][0]
        assert proc["network"]["linear"] == 32
        assert proc["train"]["steps"] == 2000
        assert proc["train"]["lr"] == 2e-4

    def test_name_or_path_points_at_flux_unet(self, tmp_path, monkeypatch):
        cfg = self._render(tmp_path, monkeypatch)
        path = cfg["config"]["process"][0]["model"]["name_or_path"]
        assert "\\" not in path
        assert path.endswith(f"diffusion_models/{FLUX_UNET}")

    def test_sample_prompts_use_trigger_literal(self, tmp_path, monkeypatch):
        cfg = self._render(tmp_path, monkeypatch)
        prompts = cfg["config"]["process"][0]["sample"]["prompts"]
        assert 3 <= len(prompts) <= 4
        assert all(p.startswith("[trigger], ") for p in prompts)


# --- retry_ladder -----------------------------------------------------------------


class TestRetryLadder:
    def test_progression(self):
        assert retry_ladder(1) == {"steps": LORA_DEFAULTS["steps"] + 500}
        assert retry_ladder(2) == {"lr": 2e-4}
        assert retry_ladder(3) == {"rank": 32}
        assert retry_ladder(4) == {"resolution": [512, 768, 1024]}

    def test_give_up_at_5(self):
        assert retry_ladder(5) == {}
        assert retry_ladder(9) == {}


# --- trainer --------------------------------------------------------------------


@pytest.fixture
def train_env(tmp_path, monkeypatch):
    """tmp CHARACTERS_DIR / AI_TOOLKIT_DIR / MODELS_DIR + character on disk."""
    chars = tmp_path / "characters"
    toolkit = tmp_path / "ai-toolkit"
    models = tmp_path / "models"
    monkeypatch.setattr(tr, "CHARACTERS_DIR", chars)
    monkeypatch.setattr(tr, "AI_TOOLKIT_DIR", toolkit)
    monkeypatch.setattr(tr, "MODELS_DIR", models)
    monkeypatch.setattr("master_agent.kb.store.kb_available", lambda: False)

    char_dir = chars / "mira"
    (char_dir / "dataset").mkdir(parents=True)
    (char_dir / "character.json").write_text(json.dumps(CHARACTER), encoding="utf-8")
    (char_dir / "dataset" / "00.png").write_bytes(b"fakepng")
    (char_dir / "dataset" / "00.txt").write_text("[trigger], x", encoding="utf-8")
    return chars, toolkit, models


def _make_toolkit(toolkit: Path, with_output: bool = True):
    toolkit.mkdir(parents=True, exist_ok=True)
    (toolkit / "run.py").write_text("# fake", encoding="utf-8")
    (toolkit / "venv" / "Scripts").mkdir(parents=True)
    (toolkit / "venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    if with_output:
        out = toolkit / "output" / "mira"
        out.mkdir(parents=True)
        (out / "mira.safetensors").write_bytes(b"weights")


class TestTrainer:
    def test_success_copies_lora(self, train_env):
        chars, toolkit, models = train_env
        _make_toolkit(toolkit)
        ran = {}

        def fake_runner(cmd, log_path):
            ran["cmd"] = cmd
            Path(log_path).write_text("training done\n", encoding="utf-8")
            return 0

        result = tr.train_lora("mira", runner=fake_runner)
        assert result["ok"] is True
        dest = models / "loras" / f"mira_r{LORA_DEFAULTS['rank']}.safetensors"
        assert result["lora_path"] == str(dest)
        assert dest.read_bytes() == b"weights"
        assert result["steps"] == LORA_DEFAULTS["steps"]
        # cmd targets the toolkit venv python + run.py + config, cwd-independent
        assert cmd_ok(ran["cmd"], toolkit)
        # config written into the toolkit
        cfg = yaml.safe_load((toolkit / "config" / "mira.yml").read_text(encoding="utf-8"))
        assert cfg["config"]["name"] == "mira"

    def test_rank_override_names_output(self, train_env):
        chars, toolkit, models = train_env
        _make_toolkit(toolkit)
        result = tr.train_lora("mira", overrides={"rank": 32}, runner=lambda c, l: 0)
        assert result["ok"] is True
        assert result["lora_path"].endswith("mira_r32.safetensors")

    def test_failed_run_returns_log_tail(self, train_env):
        chars, toolkit, models = train_env
        _make_toolkit(toolkit)

        def bad_runner(cmd, log_path):
            Path(log_path).write_text("step 1\nCUDA out of memory boom\n", encoding="utf-8")
            return 1

        result = tr.train_lora("mira", runner=bad_runner)
        assert result["ok"] is False
        assert "CUDA out of memory boom" in result["error"]
        assert not (models / "loras").exists() or not list((models / "loras").glob("*.safetensors"))

    def test_missing_toolkit_setup_error(self, train_env):
        chars, toolkit, models = train_env
        # no run.py — toolkit not installed
        result = tr.train_lora("mira", runner=lambda c, l: 0)
        assert result["ok"] is False
        assert "ai-toolkit" in result["error"]
        assert "git clone" in result["error"]

    def test_missing_character_json(self, train_env):
        chars, toolkit, models = train_env
        (chars / "mira" / "character.json").unlink()
        result = tr.train_lora("mira", runner=lambda c, l: 0)
        assert result["ok"] is False
        assert "character.json" in result["error"]


def cmd_ok(cmd, toolkit: Path) -> bool:
    toolkit_fwd = str(toolkit).replace("\\", "/")
    return (
        "python" in Path(cmd[0]).name.lower()
        and cmd[1] == "run.py"
        and cmd[2] == "config/mira.yml"
        and toolkit_fwd in cmd[0].replace("\\", "/")
    )
