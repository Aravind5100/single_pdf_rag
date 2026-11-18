"""LangChain version of the RAG pipeline: hybrid retrieval -> grounded prompt -> ChatOllama."""
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from generation_strategy import LLM_MODEL
from lc_rag.ingest import load_or_build
from lc_rag.prompts import ANSWER_PROMPT, docs_to_context, is_refusal
from lc_rag.retriever import HybridRetriever


def make_llm() -> BaseChatModel:
    # temperature 0 and the 120 s timeout match generation_strategy.generate_answer.
    return ChatOllama(model=LLM_MODEL, temperature=0, client_kwargs={"timeout": 120})


def chunk_dicts(docs: List[Document]) -> List[Dict[str, Any]]:
    """Plain-dict view of retrieved chunks, the same shape the baseline returns."""
    return [
        {"page_number": d.metadata["page_number"], "chunk_id": d.metadata["chunk_id"], "text": d.page_content}
        for d in docs
    ]


class LCRag:
    """Holds the vector store, hybrid retriever and answer chain for one PDF."""

    def __init__(
        self,
        pdf_path: str,
        llm: Optional[BaseChatModel] = None,
        embeddings: Optional[Embeddings] = None,
        log: Callable[[str], None] = lambda msg: None,
    ):
        self.chunks, self.store = load_or_build(pdf_path, embeddings=embeddings, log=log)
        self.retriever = HybridRetriever(self.chunks, self.store)
        self.llm = llm or make_llm()
        self.chain = ANSWER_PROMPT | self.llm | StrOutputParser()

    def retrieve(self, query: str, k: int) -> List[Document]:
        return self.retriever.retrieve(query, k)

    def generate(self, docs: List[Document], question: str) -> str:
        """Answer `question` from `docs` with the grounded prompt (also used by the graph)."""
        return self.chain.invoke({"context": docs_to_context(docs), "question": question}).strip()

    def answer(self, question: str, k: int = 10) -> Dict[str, Any]:
        docs = self.retrieve(question, k)
        answer = self.generate(docs, question)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunk_dicts(docs),
            "refused": is_refusal(answer),
            "trace": [f"retrieve: {len(docs)} chunks", "generate"],
        }
