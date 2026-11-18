import pytest
import requests


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Fail loudly if any test tries a real HTTP call (for example to Ollama).

    Tests that need HTTP behavior patch requests.get / requests.post themselves.
    """
    def blocked(*args, **kwargs):
        raise RuntimeError("Test attempted a real network call; mock it instead.")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
