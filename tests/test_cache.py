import pytest

import artifacts
import persistence
from persistence import build_cache_key


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF fake content A")
    return path


def key(pdf_path, chunk_size=900, overlap=350, embed_model="nomic-embed-text"):
    return build_cache_key(str(pdf_path), chunk_size=chunk_size, overlap=overlap, embed_model=embed_model)


# ---- cache key ----

def test_key_is_stable_for_identical_inputs(pdf):
    assert key(pdf) == key(pdf)


def test_key_changes_when_pdf_content_changes(pdf):
    before = key(pdf)
    pdf.write_bytes(b"%PDF fake content B")
    assert key(pdf) != before


def test_key_ignores_pdf_location_and_depends_only_on_content(pdf, tmp_path):
    copy = tmp_path / "renamed.pdf"
    copy.write_bytes(pdf.read_bytes())
    assert key(copy) == key(pdf)


@pytest.mark.parametrize("change", [
    {"chunk_size": 800},
    {"overlap": 300},
    {"embed_model": "some-other-model"},
])
def test_key_changes_when_chunk_params_or_model_change(pdf, change):
    assert key(pdf, **change) != key(pdf)


# ---- load_or_build_artifacts uses the cache (no Ollama, no real PDF) ----

@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Point the cache at a temp dir and replace the slow, Ollama-backed steps."""
    monkeypatch.setattr(persistence, "CACHE_DIR", tmp_path / "cache")
    calls = {"embed": 0}

    monkeypatch.setattr(artifacts, "extract_text_from_pdf",
                        lambda p: [{"page_number": 1, "text": "alpha beta gamma"}])

    def fake_embed(chunks):
        calls["embed"] += 1
        return [{**c, "embedding": [1.0, 0.0, 0.0, float(i)]} for i, c in enumerate(chunks)]

    monkeypatch.setattr(artifacts, "embed_chunks", fake_embed)
    return calls


def test_second_call_loads_from_cache_without_re_embedding(pdf, fake_pipeline):
    chunks1, index1 = artifacts.load_or_build_artifacts(str(pdf))
    chunks2, index2 = artifacts.load_or_build_artifacts(str(pdf))

    assert fake_pipeline["embed"] == 1
    assert chunks2 == chunks1
    assert index2.ntotal == index1.ntotal == len(chunks1)


def test_changed_chunk_params_force_a_rebuild(pdf, fake_pipeline, monkeypatch):
    artifacts.load_or_build_artifacts(str(pdf))
    monkeypatch.setattr(artifacts, "CHUNK_SIZE", 800)
    artifacts.load_or_build_artifacts(str(pdf))

    assert fake_pipeline["embed"] == 2


def test_changed_pdf_content_forces_a_rebuild(pdf, fake_pipeline):
    artifacts.load_or_build_artifacts(str(pdf))
    pdf.write_bytes(b"%PDF fake content B")
    artifacts.load_or_build_artifacts(str(pdf))

    assert fake_pipeline["embed"] == 2


def test_log_callback_reports_cache_hit_and_miss(pdf, fake_pipeline):
    messages = []
    artifacts.load_or_build_artifacts(str(pdf), log=messages.append)
    artifacts.load_or_build_artifacts(str(pdf), log=messages.append)

    assert messages[0].startswith("First run")
    assert any(m.startswith("Cache saved") for m in messages)
    assert messages[-1].startswith("Loading embeddings")
