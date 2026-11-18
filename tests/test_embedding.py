import requests

import embedding_strategy


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"embedding": [0.1, 0.2]}


def test_embed_text_sends_a_timeout(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen.update(url=url, json=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    assert embedding_strategy.embed_text("hello") == [0.1, 0.2]
    assert seen["timeout"] == 120                      # a stuck server must fail, not hang
    assert seen["json"] == {"model": embedding_strategy.EMBED_MODEL, "prompt": "hello"}
