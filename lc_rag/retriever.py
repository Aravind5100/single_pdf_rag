"""Hybrid retrieval with LangChain retrievers: BM25 + FAISS fused by an EnsembleRetriever.

Where this differs from the hand-written hybrid_retrieve_indices (also see
eval/COMPARISON.md):
  * Fusion: EnsembleRetriever uses weighted reciprocal rank fusion, so a chunk
    that both retrievers return ranks first. The baseline puts keyword hits
    first, then vector hits, with no score fusion. LangChain has no built-in
    "keyword-first" merge.
  * Keyword scoring: BM25 (term frequency, IDF, length normalisation) instead
    of the baseline's raw substring counts.
  * No fixed keyword quota: the baseline reserves max(2, top_k // 2) slots for
    keyword hits; here each retriever returns top-k and fusion decides.
"""
import re
from typing import List

import numpy as np
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from index_strategy import STOPWORDS

_TOKEN = re.compile(r"[a-zA-Z0-9']+")


def tokenize(text: str) -> List[str]:
    """Same term rules as the baseline keyword search: lowercase, [a-z0-9'] runs,
    at least 3 characters, no stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if len(t) >= 3 and t not in STOPWORDS]


class PositiveBM25Retriever(BM25Retriever):
    """BM25Retriever that only returns documents with a positive score.

    The stock class returns the top-k documents even when no query term matches
    (all scores are zero, so it hands back arbitrary documents). The baseline
    drops zero-score chunks, so this subclass does the same.
    """

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        terms = self.preprocess_func(query)
        if not terms:
            return []
        scores = self.vectorizer.get_scores(terms)
        top = np.argsort(scores)[::-1][: self.k]
        return [self.docs[i] for i in top if scores[i] > 0]


class HybridRetriever:
    """BM25 + FAISS ensemble with a per-call k and a hard cap on the result size."""

    def __init__(self, chunks: List[Document], store: FAISS, weights=(0.5, 0.5)):
        self.bm25 = PositiveBM25Retriever.from_documents(chunks, preprocess_func=tokenize, k=5)
        self.dense = store.as_retriever(search_kwargs={"k": 5})
        # id_key="chunk_id": fuse by chunk id, not by text, so identical text on
        # two pages stays two results.
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.dense], weights=list(weights), id_key="chunk_id"
        )

    def retrieve(self, query: str, k: int) -> List[Document]:
        self.bm25.k = k
        self.dense.search_kwargs = {"k": k}
        # Fusion can return up to 2k distinct chunks; keep the top k.
        return self.ensemble.invoke(query)[:k]
