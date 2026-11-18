"""Streamlit page for the framework versions (separate from the baseline main.py).

Run from the repository root:
    PDF_PATH=/path/to/file.pdf streamlit run lc_rag/app_lc.py
"""
import os
import sys
from pathlib import Path

# Streamlit puts this file's folder on sys.path, not the repo root, so add the root
# (the folder above lc_rag/) to import the top-level modules and the lc_rag package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from lc_rag.chain import LCRag  # noqa: E402
from lc_rag.graph import GraphRag  # noqa: E402
from ollama_check import OllamaNotReady, check_ollama  # noqa: E402

st.title("Single-PDF RAG: LangChain / LangGraph")

PDF_PATH = os.environ.get("PDF_PATH")
if not PDF_PATH:
    st.error("PDF_PATH environment variable not set. Please set it before running the app.")
    st.stop()

try:
    check_ollama()
except OllamaNotReady as exc:
    st.error(str(exc))
    st.stop()


@st.cache_resource(show_spinner="Loading chunks and FAISS store (embeds on the first run)...")
def get_pipelines(pdf_path: str):
    rag = LCRag(pdf_path)
    return rag, GraphRag(rag)


rag, graph_rag = get_pipelines(PDF_PATH)

mode = st.radio("Pipeline", ["LangChain chain", "LangGraph (grade, rewrite, verify)"], horizontal=True)
k = st.slider("Chunks to retrieve (k)", min_value=3, max_value=12, value=10)

if "lc_chat" not in st.session_state:
    st.session_state.lc_chat = []

question = st.text_input("Q:", key="lc_question").strip()
col1, col2 = st.columns([1, 1])
ask = col1.button("Ask")
clear = col2.button("Clear Chat")

if clear:
    st.session_state.lc_chat = []
    st.rerun()

if ask:
    if not question:
        st.warning("Type a question first.")
    else:
        pipeline = graph_rag if mode.startswith("LangGraph") else rag
        with st.spinner("Thinking..."):
            result = pipeline.answer(question, k=k)
        st.session_state.lc_chat.append({"mode": mode, "q": question, "result": result})
        st.rerun()

for turn in st.session_state.lc_chat[::-1]:
    result = turn["result"]
    st.markdown(f"**Q:** {turn['q']}  \n*{turn['mode']}*")
    st.markdown(f"**A:** {result['answer']}")
    with st.expander("Path taken (trace)"):
        for line in result["trace"]:
            st.write(line)
    with st.expander("Citations"):
        for ch in result["retrieved_chunks"]:
            st.write(f"Page {ch['page_number']} | {ch['chunk_id']}")
    st.markdown("---")
