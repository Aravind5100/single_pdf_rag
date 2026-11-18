import pytest
import requests

import ollama_check
from ollama_check import OllamaNotReady, check_ollama


class FakeResponse:
    def __init__(self, models=None, status_ok=True):
        self._models = models or []
        self._ok = status_ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.HTTPError("500 Server Error")

    def json(self):
        return {"models": [{"name": n} for n in self._models]}


def patch_get(monkeypatch, response=None, error=None):
    def fake_get(url, timeout=None):
        assert url.endswith("/api/tags")
        if error:
            raise error
        return response
    monkeypatch.setattr(ollama_check.requests, "get", fake_get)


def test_passes_when_both_models_are_pulled(monkeypatch):
    patch_get(monkeypatch, FakeResponse(["nomic-embed-text:latest", "llama3.1:8b", "gemma3:4b"]))
    check_ollama()  # no exception


def test_unpinned_model_name_matches_latest_tag(monkeypatch):
    patch_get(monkeypatch, FakeResponse(["nomic-embed-text:latest"]))
    check_ollama(required_models=["nomic-embed-text"])


def test_pinned_tag_must_match_exactly(monkeypatch):
    patch_get(monkeypatch, FakeResponse(["llama3.1:70b"]))
    with pytest.raises(OllamaNotReady, match="llama3.1:8b"):
        check_ollama(required_models=["llama3.1:8b"])


def test_missing_model_message_lists_the_pull_command(monkeypatch):
    patch_get(monkeypatch, FakeResponse(["nomic-embed-text:latest"]))
    with pytest.raises(OllamaNotReady) as err:
        check_ollama()
    assert "ollama pull llama3.1:8b" in str(err.value)
    assert "ollama pull nomic-embed-text" not in str(err.value)


def test_unreachable_server_gives_a_friendly_error(monkeypatch):
    patch_get(monkeypatch, error=requests.ConnectionError("refused"))
    with pytest.raises(OllamaNotReady, match="ollama serve"):
        check_ollama()


def test_http_error_from_server_is_reported_as_not_ready(monkeypatch):
    patch_get(monkeypatch, FakeResponse(status_ok=False))
    with pytest.raises(OllamaNotReady, match="Cannot reach Ollama"):
        check_ollama()
