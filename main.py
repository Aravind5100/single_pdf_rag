import os
import sys

# ✅ ADDED: persistence helpers (create persistence.py as I described earlier)
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
from chunk_strategy import chunk_pages
from embedding_strategy import embed_chunks

# ✅ ADDED: import EMBED_MODEL if you have it; otherwise keep the string below
try:
    from embedding_strategy import EMBED_MODEL
except Exception:
    EMBED_MODEL = "nomic-embed-text"  # fallback

from index_strategy import build_faiss_index  # debug_find_term
from generation_strategy import answer_question
import streamlit as st


# ✅ Keep your existing PDF path (you can change to handbook later)
pdf_path = "/Users/aravindkompalli/Downloads/IT_ENDS_WITH_US.pdf"


# ✅ ADDED: Chunking params MUST match your chunk_strategy defaults
# If your chunk_strategy uses different defaults, update these to match exactly.
CHUNK_SIZE = 900
OVERLAP = 350


# ✅ ADDED: Build a cache key based on:
# - PDF content hash
# - chunk params
# - embedding model
cache_key = build_cache_key(
    pdf_path,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP,
    embed_model=EMBED_MODEL,
)
paths = cache_paths(cache_key)


# ✅ ADDED: Load from cache if present, else build and save once
if cache_exists(paths):
    # FAST path
    st.write("✅ Loading embeddings + FAISS index from cache...")
    embedded_chunks = load_embedded_chunks(paths["chunks"])
    index = load_faiss_index(paths["index"])

    # These are not strictly needed once cached, but keep variables available for debugging
    pages = []
    chunks = []

else:
    # SLOW path (first run only)
    st.write("🛠️ First run: extracting PDF, chunking, embedding, building FAISS index...")
    pages = extract_text_from_pdf(pdf_path)

    #print("Total pages:", len(pages))
    #print("\n--- Page preview ---\n")
    #print(pages[5]["text"][:])

    chunks = chunk_pages(pages)

    # print("Total chunks:", len(chunks))
    # print("\n--- Chunk preview ---\n")
    # #print(chunks[750]["text"][:])
    # print("Page:", chunks[633]["page_number"])
    # print("ID:", chunks[633]["chunk_id"])

    # print("chunk 4:", chunks[633]["text"][:],"\n")
    # print("chunk 5:", chunks[634]["text"][:],"\n")
    # print("chunk 6:", chunks[635]["text"][:],"\n")

    embedded_chunks = embed_chunks(chunks)
    #debug_find_term(embedded_chunks, "atlas", max_hits=10)

    #lengths = set(len(chunk["embedding"]) for chunk in embedded_chunks)
    #print("Unique embedding lengths:", lengths)

    #print("Embedding length:", len(embedded_chunks[0]["embedding"]))
    #print("First 5 numbers:", embedded_chunks[0]["embedding"][:5])

    index, vectors = build_faiss_index(embedded_chunks)

    # ✅ ADDED: Save cache for next runs
    save_embedded_chunks(paths["chunks"], embedded_chunks)
    save_faiss_index(paths["index"], index)
    save_meta(paths["meta"], {
        "pdf_path": pdf_path,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "embed_model": EMBED_MODEL
    })

    st.write("✅ Cache saved. Next run will be fast.")


# -----------------------------
# ✅ STREAMLIT-SAFE UI
# -----------------------------
# IMPORTANT:
# Streamlit reruns the script top-to-bottom on every interaction.
# So a `while True:` loop + `st.text_input` will not behave like a terminal app.
# We'll keep your loop (unchanged) but prevent it from running,
# and implement a Streamlit-native interaction below.

st.write("\nHi Ark! How Can I Help You? (type 'exit' to quit)\n")


# ✅ ADDED: session-state chat history (optional but very useful)
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"q":..., "a":..., "citations":[...]}

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# ✅ ADDED: a stable input key prevents StreamlitDuplicateElementId
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
        st.write("👋 Exiting (in Streamlit, just stop asking questions).")
    elif not question:
        st.warning("Type a question first.")
    else:
        # ✅ Your existing answer pipeline call
        result = answer_question(question, index, embedded_chunks, top_k=10)

        # Save to chat history
        citations = [f"Page {ch['page_number']} | {ch['chunk_id']}" for ch in result["retrieved_chunks"]]
        st.session_state.chat.append({
            "q": question,
            "a": result["answer"],
            "citations": citations
        })
        st.session_state.last_question = question

        # Clear input (Streamlit way)
        st.rerun()


# Render chat history
for turn in st.session_state.chat[::-1]:
    st.markdown(f"**Q:** {turn['q']}")
    st.markdown(f"**A:** {turn['a']}")
    with st.expander("Citations"):
        for c in turn["citations"]:
            st.write(c)
    st.markdown("---")


# -----------------------------
# ✅ YOUR ORIGINAL LOOP (PRESERVED)
# -----------------------------
# NOTE: This is terminal-style code and doesn't work correctly in Streamlit due to reruns.
# We keep it exactly as-is, but prevent execution.

if False:
    while True:
       question = st.text_input("Q: ").strip()
       if question.lower() in {"exit", "quit"}:
           break

       result = answer_question(question, index, embedded_chunks, top_k=10)

       print("\nA:", result["answer"])
       print("\nCitations:")
       for ch in result["retrieved_chunks"]:
           print(f"Page {ch['page_number']} | {ch['chunk_id']}")
       print()
