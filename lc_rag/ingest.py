"""PDF -> page-aware chunks -> FAISS vector store, using LangChain components.

Cached on disk under cache/lc/<key>/ so repeat runs do not re-embed. The key
reuses the baseline's build_cache_key (PDF content hash + chunk params + model)
with a pipeline tag added to the model field, so it never collides with a
baseline cache entry.
"""
import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import persistence
from chunk_strategy import CHUNK_SIZE, OVERLAP, normalize_for_search
from embedding_strategy import EMBED_MODEL

# Bump when ingestion logic changes so stale cache entries are not reused.
PIPELINE_TAG = "lc-recursive-normalized-v1"


def load_pages(pdf_path: str) -> List[Document]:
    """One Document per page, with a 1-based page_number in the metadata.

    PyMuPDFLoader stores a 0-based `page`; the baseline (and the eval ground
    truth) use 1-based page numbers, so the shift happens here once.
    The page text gets the baseline's normalize_for_search (hyphenated line
    breaks joined, whitespace collapsed) so both versions index the same text.
    """
    pages = []
    for doc in PyMuPDFLoader(pdf_path).load():
        text = normalize_for_search(doc.page_content)
        if not text:
            continue
        pages.append(Document(
            page_content=text,
            metadata={"page_number": int(doc.metadata["page"]) + 1},
        ))
    return pages


def split_pages(pages: List[Document]) -> List[Document]:
    """Split pages into CHUNK_SIZE / OVERLAP chunks and give each a baseline-style chunk_id."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    chunks = splitter.split_documents(pages)

    counters: dict = {}
    for chunk in chunks:
        page = chunk.metadata["page_number"]
        counters[page] = counters.get(page, 0) + 1
        chunk.metadata["chunk_id"] = f"{page}_chunk_{counters[page]}"
    return chunks


def make_embeddings() -> Embeddings:
    return OllamaEmbeddings(model=EMBED_MODEL)


def cache_dir(pdf_path: str) -> Path:
    key = persistence.build_cache_key(
        pdf_path,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        embed_model=f"{EMBED_MODEL}|{PIPELINE_TAG}",
    )
    return persistence.CACHE_DIR / "lc" / key


def _save_chunks(path: Path, chunks: List[Document]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"page_content": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")


def _load_chunks(path: Path) -> List[Document]:
    with path.open("r", encoding="utf-8") as f:
        return [Document(page_content=r["page_content"], metadata=r["metadata"])
                for r in (json.loads(line) for line in f if line.strip())]


def load_or_build(
    pdf_path: str,
    embeddings: Optional[Embeddings] = None,
    log: Callable[[str], None] = lambda msg: None,
) -> Tuple[List[Document], FAISS]:
    """Return (chunks, vector store), loading from cache when possible.

    normalize_L2=True makes FAISS L2-normalize vectors before indexing and
    before every query, which is what the baseline does by hand, so IndexFlatL2
    distances rank the same way as cosine similarity. It is passed on load too,
    because the flag is not stored in the saved index.
    """
    embeddings = embeddings or make_embeddings()
    folder = cache_dir(pdf_path)
    chunks_file = folder / "chunks.jsonl"

    if (folder / "index.faiss").exists() and (folder / "index.pkl").exists() and chunks_file.exists():
        log("Loading LangChain chunks + FAISS store from cache...")
        store = FAISS.load_local(
            str(folder), embeddings,
            allow_dangerous_deserialization=True,  # our own cache file, written by this module
            normalize_L2=True,
        )
        return _load_chunks(chunks_file), store

    log("First run: loading PDF, splitting, embedding, building FAISS store...")
    chunks = split_pages(load_pages(pdf_path))
    store = FAISS.from_documents(chunks, embeddings, normalize_L2=True)

    folder.mkdir(parents=True, exist_ok=True)
    store.save_local(str(folder))
    _save_chunks(chunks_file, chunks)
    log("Cache saved.")
    return chunks, store
