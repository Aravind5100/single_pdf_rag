"""Load the embedded chunks and FAISS index for a PDF from cache, or build them.

Shared by the Streamlit app (main.py) and the eval (eval/eval_runner.py) so
both use the same cache key and repeat runs do not re-embed the document.
"""
from typing import Callable, Tuple

from chunk_strategy import chunk_pages, CHUNK_SIZE, OVERLAP
from embedding_strategy import embed_chunks, EMBED_MODEL
from extract_pdf_text import extract_text_from_pdf
from index_strategy import build_faiss_index
from persistence import (
    build_cache_key,
    cache_exists,
    cache_paths,
    load_embedded_chunks,
    load_faiss_index,
    save_embedded_chunks,
    save_faiss_index,
    save_meta,
)


def load_or_build_artifacts(pdf_path: str, log: Callable[[str], None] = lambda msg: None) -> Tuple[list, object]:
    """Return (embedded_chunks, faiss_index) for pdf_path.

    The cache key covers the PDF content, chunk size, overlap and embedding
    model, so changing any of them builds a fresh entry. `log` receives short
    progress messages (main.py passes st.write, the eval passes print).
    """
    cache_key = build_cache_key(
        pdf_path,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        embed_model=EMBED_MODEL,
    )
    paths = cache_paths(cache_key)

    if cache_exists(paths):
        log("Loading embeddings + FAISS index from cache...")
        embedded_chunks = load_embedded_chunks(paths["chunks"])
        index = load_faiss_index(paths["index"])
        return embedded_chunks, index

    log("First run: extracting PDF, chunking, embedding, building FAISS index...")
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    embedded_chunks = embed_chunks(chunks)
    index, _vectors = build_faiss_index(embedded_chunks)

    save_embedded_chunks(paths["chunks"], embedded_chunks)
    save_faiss_index(paths["index"], index)
    save_meta(paths["meta"], {
        "pdf_path": pdf_path,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "embed_model": EMBED_MODEL,
    })
    log("Cache saved. Next run will be fast.")
    return embedded_chunks, index
