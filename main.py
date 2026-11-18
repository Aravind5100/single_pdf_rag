import os

from artifacts import load_or_build_artifacts
from generation_strategy import answer_question
from ollama_check import check_ollama, OllamaNotReady
import streamlit as st

PDF_PATH = os.environ.get("PDF_PATH")
if not PDF_PATH:
    st.error("PDF_PATH environment variable not set. Please set it before running the app.")
    st.stop()

try:
    check_ollama()
except OllamaNotReady as exc:
    st.error(str(exc))
    st.stop()

embedded_chunks, index = load_or_build_artifacts(PDF_PATH, log=st.write)


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
