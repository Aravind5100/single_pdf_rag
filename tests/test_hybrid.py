"""Tests for keyword search and the keyword + vector merge.

A real (tiny) FAISS index is used with fake 2-D embeddings. Chunk i sits at
angle i * 10 degrees, and the query vector points at angle 0, so the vector
ranking is always 0, 1, 2, ... Only the query embedding call is stubbed, so
nothing needs Ollama.
"""
import math

import numpy as np
import pytest

import index_strategy
from index_strategy import build_faiss_index, hybrid_retrieve_indices, keyword_search_indices


def make_chunks(texts):
    chunks = []
    for i, text in enumerate(texts):
        angle = math.radians(i * 10)
        chunks.append({
            "page_number": i + 1,
            "chunk_id": f"{i + 1}_chunk_1",
            "text": text,
            "embedding": [math.cos(angle), math.sin(angle)],
        })
    return chunks


@pytest.fixture
def stub_query(monkeypatch):
    monkeypatch.setattr(index_strategy, "embed_query",
                        lambda q: np.array([[1.0, 0.0]], dtype="float32"))


# ---- keyword_search_indices ----

def test_keyword_search_ranks_by_term_count_and_drops_zero_scores():
    chunks = make_chunks(["parking parking parking", "parking", "nothing relevant"])
    assert keyword_search_indices(chunks, "parking rules", top_k=5) == [0, 1]


def test_keyword_search_ignores_stopwords_and_short_terms():
    chunks = make_chunks(["the of in to", "an is on"])
    # Every query term is a stopword or shorter than 3 characters.
    assert keyword_search_indices(chunks, "what is the of", top_k=5) == []


def test_keyword_search_respects_top_k():
    chunks = make_chunks(["pets pets", "pets", "pets pets pets"])
    assert keyword_search_indices(chunks, "pets", top_k=2) == [2, 0]


# ---- hybrid_retrieve_indices ----

def test_keyword_hits_come_before_vector_hits(stub_query):
    # Chunks 4 and 5 are the only keyword matches; the vector order is 0, 1, 2, ...
    chunks = make_chunks(["x"] * 4 + ["banana banana", "banana"] + ["x"] * 2)
    index, _ = build_faiss_index(chunks)

    result = hybrid_retrieve_indices(index, chunks, "banana", top_k=4)

    assert result[:2] == [4, 5]          # keyword-first, best keyword score first
    assert result[2:] == [0, 1]          # then vector neighbours


def test_duplicates_between_keyword_and_vector_are_removed(stub_query):
    # Chunk 0 is both the nearest vector hit and a keyword hit.
    chunks = make_chunks(["banana", "x", "x", "banana banana"] + ["x"] * 4)
    index, _ = build_faiss_index(chunks)

    result = hybrid_retrieve_indices(index, chunks, "banana", top_k=5)

    assert len(result) == len(set(result))
    assert result[:2] == [3, 0]          # keyword order, chunk 0 only once
    assert result.count(0) == 1


def test_result_is_capped_at_top_k(stub_query):
    chunks = make_chunks(["banana"] * 3 + ["x"] * 9)
    index, _ = build_faiss_index(chunks)

    for top_k in (4, 6, 10):
        assert len(hybrid_retrieve_indices(index, chunks, "banana", top_k=top_k)) <= top_k


def test_query_with_no_keyword_terms_falls_back_to_vector_order(stub_query):
    chunks = make_chunks(["x"] * 8)
    index, _ = build_faiss_index(chunks)

    # "what is the" is all stopwords, so there are no keyword hits.
    assert hybrid_retrieve_indices(index, chunks, "what is the", top_k=4) == [0, 1, 2, 3]


def test_keyword_quota_is_half_of_top_k_with_a_floor_of_two(stub_query):
    chunks = make_chunks(["banana"] * 6 + ["x"] * 6)
    index, _ = build_faiss_index(chunks)

    # top_k=6 -> 3 keyword slots; top_k=2 -> floor of 2 keyword slots.
    six = hybrid_retrieve_indices(index, chunks, "banana", top_k=6)
    assert six[:3] == keyword_search_indices(chunks, "banana", top_k=3)
    two = hybrid_retrieve_indices(index, chunks, "banana", top_k=2)
    assert two[:2] == keyword_search_indices(chunks, "banana", top_k=2)


def legacy_hybrid(index, chunks, query, top_k):
    """The merge exactly as it was before the cap was added (kept here to pin the 'no-op' claim)."""
    k_keyword = max(2, top_k // 2)
    k_idx = keyword_search_indices(chunks, query, top_k=k_keyword)
    _, v_idx = index_strategy.search_index(index, index_strategy.embed_query(query), top_k=top_k)
    merged, seen = [], set()
    for idx in k_idx:
        if idx != -1 and idx not in seen:
            merged.append(idx)
            seen.add(idx)
    for idx in list(v_idx):
        if idx != -1 and idx not in seen:
            merged.append(idx)
            seen.add(idx)
        if len(merged) >= top_k:
            break
    return merged


FIXTURES = [
    (["banana"] * 6 + ["x"] * 10, "banana"),                                # keyword hits overlap the vector hits
    (["x"] * 6 + ["banana"] * 3 + ["x"] * 7, "banana"),                     # keyword hits sit far from the vector hits
    (["banana apple", "apple", "x", "banana"] + ["x"] * 12, "banana apple"),
    (["x"] * 16, "what is the"),                                            # no keyword terms at all
]


@pytest.mark.parametrize("top_k", range(3, 11))
@pytest.mark.parametrize("texts,query", FIXTURES)
def test_cap_is_a_no_op_for_top_k_of_3_or_more(stub_query, texts, query, top_k):
    chunks = make_chunks(texts)
    index, _ = build_faiss_index(chunks)

    assert hybrid_retrieve_indices(index, chunks, query, top_k=top_k) == legacy_hybrid(index, chunks, query, top_k)


def test_top_k_of_two_used_to_overshoot_to_three_and_is_now_capped(stub_query):
    # Keyword hits are chunks 6, 7 (far from the vector neighbours 0, 1, ...), so with top_k=2 the keyword
    # loop already fills the result and the old vector loop appended one more before checking the cap.
    chunks = make_chunks(["x"] * 6 + ["banana banana", "banana"] + ["x"] * 6)
    index, _ = build_faiss_index(chunks)

    assert len(legacy_hybrid(index, chunks, "banana", 2)) == 3
    assert hybrid_retrieve_indices(index, chunks, "banana", top_k=2) == [6, 7]


def test_top_k_of_one_returns_exactly_one_chunk_the_best_keyword_hit(stub_query):
    # Regression test for a fixed quirk: the keyword quota has a floor of 2, so
    # top_k=1 (and top_k=2) used to return more than the cap. The result is now capped.
    chunks = make_chunks(["banana"] * 3 + ["x"] * 5)
    index, _ = build_faiss_index(chunks)

    result = hybrid_retrieve_indices(index, chunks, "banana", top_k=1)

    assert result == keyword_search_indices(chunks, "banana", top_k=1)
    assert len(result) == 1


@pytest.mark.parametrize("top_k", range(1, 11))
def test_result_never_exceeds_top_k_for_any_top_k(stub_query, top_k):
    chunks = make_chunks(["banana"] * 6 + ["x"] * 10)
    index, _ = build_faiss_index(chunks)

    assert len(hybrid_retrieve_indices(index, chunks, "banana", top_k=top_k)) <= top_k
