"""Unit tests for llama.cpp as a first-class local LLM backend (no GPU).

Stub TCP + HTTP. Run: python -m pytest tests/test_llm_local.py -q
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from master_agent.llm import (
    AUTO_CHAIN,
    attach_llm_health,
    chat_client_config,
    embeddings_endpoint,
    format_local_llm_health,
    get_llm,
    local_llm_health,
    normalize_provider_name,
    openai_compat_base,
    parse_provider_spec,
    preferred_local_provider,
    reset_endpoint_cache,
    vision_endpoint,
    _llm_for,
)


@pytest.fixture(autouse=True)
def _clear_tcp_cache():
    reset_endpoint_cache()
    yield
    reset_endpoint_cache()


def _up_only(*names: str):
    live = set(names)

    def endpoint_up(url: str, default_port: int) -> bool:
        if default_port == 11434 or "11434" in url:
            return "ollama" in live
        if default_port == 8080 or "8080" in url:
            return "llamacpp" in live
        return False

    return endpoint_up


class TestProviderAliases:
    def test_llamacpp_aliases(self):
        for raw in ("llamacpp", "llama.cpp", "llama-cpp", "llama_cpp", "LLAMA.CPP"):
            assert normalize_provider_name(raw) == "llamacpp"
        assert parse_provider_spec("llama.cpp:hermes-q4") == ("llamacpp", "hermes-q4")
        assert normalize_provider_name("ollama") == "ollama"

    def test_auto_chain_order(self):
        assert AUTO_CHAIN == ("ollama", "llamacpp", "grok")


class TestChatBaseUrl:
    def test_ollama_openai_compat(self):
        cfg = chat_client_config("ollama")
        assert cfg["provider"] == "ollama"
        assert cfg["base_url"].endswith("/v1")
        assert "11434" in cfg["base_url"] or "ollama" in cfg["base_url"]
        cfg2 = chat_client_config("ollama:gemma4:latest")
        assert cfg2["model"] == "gemma4:latest"

    def test_llamacpp_openai_compat(self):
        cfg = chat_client_config("llamacpp")
        assert cfg["provider"] == "llamacpp"
        assert cfg["base_url"] == openai_compat_base("http://127.0.0.1:8080") or cfg[
            "base_url"
        ].endswith("/v1")
        assert "/v1" in cfg["base_url"]
        assert "8080" in cfg["base_url"]
        named = chat_client_config("llama.cpp:my-gguf")
        assert named["provider"] == "llamacpp"
        assert named["model"] == "my-gguf"

    def test_openai_compat_base_idempotent(self):
        assert openai_compat_base("http://127.0.0.1:8080") == "http://127.0.0.1:8080/v1"
        assert openai_compat_base("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/v1"

    def test_get_llm_llamacpp_builds_client(self):
        llm = get_llm(provider="llamacpp")
        base = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", "")
        assert "/v1" in str(base)
        assert "8080" in str(base)

    def test_get_llm_ollama_unchanged(self):
        llm = _llm_for("ollama", 0.2)
        base = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", "")
        assert "/v1" in str(base)
        assert "11434" in str(base)


class TestAutoDiscovery:
    def test_auto_uses_llamacpp_when_ollama_down(self):
        with patch("master_agent.llm.endpoint_up", side_effect=_up_only("llamacpp")), patch(
            "master_agent.llm.has_valid_api_key", return_value=False
        ), patch(
            "master_agent.llm._llamacpp_llm", return_value="LLAMACPP_CLIENT"
        ) as build:
            got = get_llm(provider="auto")
        assert got == "LLAMACPP_CLIENT"
        build.assert_called_once()

    def test_auto_prefers_ollama(self):
        with patch("master_agent.llm.endpoint_up", side_effect=_up_only("ollama", "llamacpp")), patch(
            "master_agent.llm._ollama_llm", return_value="OLLAMA_CLIENT"
        ) as build:
            got = get_llm(provider="auto")
        assert got == "OLLAMA_CLIENT"
        build.assert_called_once()

    def test_preferred_local_ollama_first(self):
        with patch("master_agent.llm.endpoint_up", side_effect=_up_only("ollama", "llamacpp")):
            assert preferred_local_provider() == "ollama"
        with patch("master_agent.llm.endpoint_up", side_effect=_up_only("llamacpp")):
            assert preferred_local_provider() == "llamacpp"
        with patch("master_agent.llm.endpoint_up", side_effect=_up_only()):
            assert preferred_local_provider() is None


class TestHealth:
    def test_only_llamacpp_does_not_claim_ollama(self):
        with patch("master_agent.llm.endpoint_up", side_effect=_up_only("llamacpp")):
            detail = local_llm_health()
        assert detail["ollama"]["up"] is False
        assert detail["llamacpp"]["up"] is True
        assert detail["preferred"] == "llamacpp"
        assert "8080" in detail["llamacpp"]["url"]
        with patch("master_agent.llm.endpoint_up", side_effect=_up_only("llamacpp")):
            reset_endpoint_cache()
            out = {}
            attach_llm_health(out)
        assert out["ollama"] is False
        assert out["llamacpp"] is True
        assert out["local_llm"]["llamacpp"]["up"] is True
        text = format_local_llm_health(detail)
        assert "llamacpp" in text
        assert "up" in text

    def test_only_ollama_does_not_claim_llamacpp(self):
        with patch("master_agent.llm.endpoint_up", side_effect=_up_only("ollama")):
            detail = local_llm_health()
        assert detail["ollama"]["up"] is True
        assert detail["llamacpp"]["up"] is False
        assert detail["preferred"] == "ollama"


class TestEmbeddingsPath:
    def test_endpoints(self):
        assert embeddings_endpoint("ollama").endswith("/api/embed")
        assert embeddings_endpoint("llamacpp").endswith("/v1/embeddings")
        assert embeddings_endpoint("llama.cpp").endswith("/v1/embeddings")

    def test_openai_compat_embed(self):
        from master_agent.kb.store import embed_texts

        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                ]
            }
            return resp

        with patch("master_agent.kb.store.active_local_backend", return_value="llamacpp"), patch(
            "httpx.post", side_effect=fake_post
        ):
            vecs = embed_texts(["a", "b"])
        assert captured["url"].endswith("/v1/embeddings")
        assert captured["json"]["input"] == ["a", "b"]
        assert vecs == [[0.1, 0.2], [0.3, 0.4]]

    def test_ollama_native_embed(self):
        from master_agent.kb.store import embed_texts

        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {"embeddings": [[1.0], [2.0]]}
            return resp

        with patch("httpx.post", side_effect=fake_post):
            vecs = embed_texts(["a", "b"], backend="ollama")
        assert captured["url"].endswith("/api/embed")
        assert vecs == [[1.0], [2.0]]

    def test_missing_embeddings_raise(self):
        from master_agent.kb.store import embed_texts

        def fake_post(url, json=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {"error": "no embeddings route"}
            return resp

        with patch("httpx.post", side_effect=fake_post):
            with pytest.raises(RuntimeError, match="embeddings"):
                embed_texts(["hello"], backend="llamacpp")


class TestVisionPath:
    def test_endpoints(self):
        assert vision_endpoint("ollama").endswith("/api/chat")
        assert vision_endpoint("llamacpp").endswith("/v1/chat/completions")

    def test_llamacpp_openai_multimodal(self):
        from master_agent.judge.vision import _vision_chat

        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "choices": [{"message": {"content": '{"score": 0.8}'}}]
            }
            return resp

        with patch("httpx.post", side_effect=fake_post):
            text = _vision_chat(
                "llamacpp",
                "sys",
                "review",
                [("image/jpeg", "abc123")],
            )
        assert captured["url"].endswith("/v1/chat/completions")
        content = captured["json"]["messages"][1]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert "base64,abc123" in content[1]["image_url"]["url"]
        assert "score" in text

    def test_ollama_native_images(self):
        from master_agent.judge.vision import _vision_chat

        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {"message": {"content": '{"score": 0.9}'}}
            return resp

        with patch("httpx.post", side_effect=fake_post):
            _vision_chat("ollama", "sys", "review", [("image/jpeg", "abc123")])
        assert captured["url"].endswith("/api/chat")
        assert captured["json"]["messages"][1]["images"] == ["abc123"]

    def test_vision_review_degrades_without_backend(self):
        from master_agent.judge.vision import vision_review

        with patch("master_agent.judge.vision.active_local_backend", return_value=None):
            assert vision_review("x.jpg", user_request="brief") is None

    def test_vision_available_with_only_llamacpp(self):
        from master_agent.judge import vision as vis

        with patch.object(vis, "VISION_ENABLED", True), patch(
            "shutil.which", return_value="/usr/bin/ffmpeg"
        ), patch("master_agent.judge.vision.active_local_backend", return_value="llamacpp"):
            assert vis.vision_available() is True
        with patch.object(vis, "VISION_ENABLED", True), patch(
            "shutil.which", return_value="/usr/bin/ffmpeg"
        ), patch("master_agent.judge.vision.active_local_backend", return_value=None):
            assert vis.vision_available() is False


class TestUnknownProvider:
    def test_unknown_raises(self):
        with pytest.raises(RuntimeError, match="unknown LLM provider"):
            chat_client_config("kimi")
        with pytest.raises(RuntimeError, match="unknown LLM provider"):
            _llm_for("kimi", 0.1)
