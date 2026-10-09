"""Health and auto routing must stay local and must not touch the Hermes auth store."""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import patch

import pytest

from master_agent import llm


def _jwt(exp: float) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"h.{payload}.s"


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = {
        "providers": {
            "xai-oauth": {
                "tokens": {"access_token": _jwt(time.time() + 60), "refresh_token": "r"},
            }
        }
    }
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(store), encoding="utf-8")
    return path


def test_health_reads_grok_without_refresh_or_write(hermes_home):
    before = hermes_home.read_bytes()
    with patch("master_agent.xai_oauth._refresh_tokens") as refresh, patch(
        "master_agent.xai_oauth._token_endpoint"
    ) as endpoint, patch("master_agent.llm.endpoint_up", return_value=False), patch(
        "master_agent.llm.has_valid_api_key", return_value=False
    ):
        detail = llm.local_llm_health()
        out = llm.attach_llm_health({})
    refresh.assert_not_called()
    endpoint.assert_not_called()
    assert detail["grok"]["up"] is True
    assert out["grok"] is True
    assert hermes_home.read_bytes() == before


def test_grok_availability_is_read_only(hermes_home):
    with patch("master_agent.xai_oauth.resolve_access_token") as resolve, patch(
        "master_agent.llm.has_valid_api_key", return_value=False
    ):
        assert llm.provider_available("grok") is True
    resolve.assert_not_called()


def test_save_store_is_atomic(hermes_home):
    from master_agent import xai_oauth

    xai_oauth._save_store({"providers": {}})
    assert json.loads(hermes_home.read_text(encoding="utf-8")) == {"providers": {}}
    leftovers = [p.name for p in hermes_home.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
