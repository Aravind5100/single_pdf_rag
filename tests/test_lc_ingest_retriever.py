"""Offline tests for lc_rag ingestion and hybrid retrieval (fake embeddings, generated PDF)."""
from typing import ClassVar

import fitz
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import persistence
from lc_rag import ingest
from lc_rag.retriever import HybridRetriever, PositiveBM25Retriever, tokenize


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Deterministic fake embeddings that count how many texts were embedded."""
    calls: ClassVar[int] = 0

    def embed_documents(self, texts):
        type(self).calls += len(texts)
        return super().embed_documents(texts)


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Page one talks about pets and aquariums.")
    doc.new_page()  # blank page
    doc.new_page().insert_text((72, 72), "Page three covers parking permits.")
    doc.save(str(path))
    return path


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "CACHE_DIR", tmp_path / "cache")
    CountingEmbeddings.calls = 0


# ---- loading and splitting ----

def test_load_pages_uses_one_based_page_numbers_and_skips_blank_pages(pdf):
    pages = ingest.load_pages(str(pdf))
    assert [p.metadata["page_number"] for p in pages] == [1, 3]
    assert "pets" in pages[0].page_content


def test_split_pages_keeps_chunks_on_their_page_with_per_page_ids():
    long_a = Document(page_content="alpha " * 400, metadata={"page_number": 4})
    long_b = Document(page_content="bravo " * 400, metadata={"page_number": 5})
    chunks = ingest.split_pages([long_a, long_b])

    assert all(len(c.page_content) <= ingest.CHUNK_SIZE for c in chunks)
    assert all(("alpha" in c.page_content) == (c.metadata["page_number"] == 4) for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids[:2] == ["4_chunk_1", "4_chunk_2"]
    assert "5_chunk_1" in ids and len(ids) == len(set(ids))


# ---- cache ----

def test_second_load_uses_cache_and_keeps_the_normalize_flag(pdf):
    emb = CountingEmbeddings(size=8)
    chunks1, store1 = ingest.load_or_build(str(pdf), embeddings=emb)
    embedded_once = CountingEmbeddings.calls
    chunks2, store2 = ingest.load_or_build(str(pdf), embeddings=emb)

    assert embedded_once == len(chunks1) > 0
    assert CountingEmbeddings.calls == embedded_once          # nothing re-embedded
    assert [c.page_content for c in chunks2] == [c.page_content for c in chunks1]
    assert store1._normalize_L2 and store2._normalize_L2


def test_cache_entry_lives_under_lc_and_changes_with_chunk_params(pdf, monkeypatch):
    before = ingest.cache_dir(str(pdf))
    assert before.parent.name == "lc"
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 800)
    assert ingest.cache_dir(str(pdf)) != before


# ---- tokenizer and BM25 ----

def test_tokenize_matches_baseline_term_rules():
    assert tokenize("What is the Parking-Fine policy?") == ["parking", "fine", "policy"]
    assert tokenize("of to in") == []


def make_docs(texts):
    return [Document(page_content=t, metadata={"page_number": i + 1, "chunk_id": f"{i + 1}_chunk_1"})
            for i, t in enumerate(texts)]


def test_positive_bm25_drops_zero_score_documents_and_empty_queries():
    docs = make_docs(["banana bread recipe", "apple pie recipe", "cherry tart"])
    bm25 = PositiveBM25Retriever.from_documents(docs, preprocess_func=tokenize, k=3)

    assert [d.metadata["chunk_id"] for d in bm25.invoke("banana")] == ["1_chunk_1"]
    assert bm25.invoke("zzzqqq") == []
    assert bm25.invoke("what is the") == []     # all stopwords


# ---- hybrid retriever ----

def build_hybrid(texts):
    from langchain_community.vectorstores import FAISS
    docs = make_docs(texts)
    store = FAISS.from_documents(docs, DeterministicFakeEmbedding(size=16), normalize_L2=True)
    return HybridRetriever(docs, store)


def test_result_is_capped_at_k():
    hybrid = build_hybrid([f"banana text number {i}" for i in range(12)])
    for k in (2, 5, 8):
        assert len(hybrid.retrieve("banana", k)) <= k


def test_chunk_found_by_both_retrievers_ranks_first():
    # The query equals chunk 3's text, so its (deterministic) embedding is an exact
    # match for the dense retriever, and BM25 also scores it.
    query = "banana banana banana"
    hybrid = build_hybrid(["banana once", "other words here", "nothing", query, "banana twice banana", "filler"])

    assert hybrid.retrieve(query, 4)[0].metadata["chunk_id"] == "4_chunk_1"


def test_identical_text_on_two_pages_stays_two_results():
    # id_key="chunk_id" fuses by id; the default (page_content) would merge these.
    hybrid = build_hybrid(["same boilerplate banana", "same boilerplate banana", "unrelated filler"])
    ids = [d.metadata["chunk_id"] for d in hybrid.retrieve("same boilerplate banana", 3)]

    assert "1_chunk_1" in ids and "2_chunk_1" in ids
