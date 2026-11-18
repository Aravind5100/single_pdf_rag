import os

from persistence import (
    build_cache_key,
    cache_paths,
    cache_exists,
    save_embedded_chunks,
    load_embedded_chunks,
    save_faiss_index,
    load_faiss_index,
    save_meta,
)
from extract_pdf_text import extract_text_from_pdf
from chunk_strategy import chunk_pages, CHUNK_SIZE, OVERLAP
from embedding_strategy import embed_chunks, EMBED_MODEL
from index_strategy import build_faiss_index
from generation_strategy import answer_question
import streamlit as st

PDF_PATH = os.environ.get("PDF_PATH")
if not PDF_PATH:
    st.error("PDF_PATH environment variable not set. Please set it before running the app.")
    st.stop()

cache_key = build_cache_key(
    PDF_PATH,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP,
    embed_model=EMBED_MODEL,
)
paths = cache_paths(cache_key)


if cache_exists(paths):
    st.write("Loading embeddings + FAISS index from cache...")
    embedded_chunks = load_embedded_chunks(paths["chunks"])
    index = load_faiss_index(paths["index"])
else:
    st.write("First run: extracting PDF, chunking, embedding, building FAISS index...")
    pages = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_pages(pages)
    embedded_chunks = embed_chunks(chunks)
    index, vectors = build_faiss_index(embedded_chunks)

    save_embedded_chunks(paths["chunks"], embedded_chunks)
    save_faiss_index(paths["index"], index)
    save_meta(paths["meta"], {
        "pdf_path": PDF_PATH,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "embed_model": EMBED_MODEL,
    })
    st.write("Cache saved. Next run will be fast.")


st.write("Hi! How can I help you? (type 'exit' to quit)")

if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


question = st.text_input("Q:", key="question_input").strip()

col1, col2 = st.columns([1, 1])
ask = col1.button("Ask")
clear = col2.button("Clear Chat")

if clear:
    st.session_state.chat = []
    st.session_state.last_question = ""
    st.rerun()

if ask:
    if question.lower() in {"exit", "quit"}:
        st.write("Goodbye!")
    elif not question:
        st.warning("Type a question first.")
    else:
        result = answer_question(question, index, embedded_chunks, top_k=10)
        citations = [f"Page {ch['page_number']} | {ch['chunk_id']}" for ch in result["retrieved_chunks"]]
        st.session_state.chat.append({
            "q": question,
            "a": result["answer"],
            "citations": citations
        })
        st.session_state.last_question = question

        st.rerun()


# Render chat history
for turn in st.session_state.chat[::-1]:
    st.markdown(f"**Q:** {turn['q']}")
    st.markdown(f"**A:** {turn['a']}")
    with st.expander("Citations"):
        for c in turn["citations"]:
            st.write(c)
    st.markdown("---")
