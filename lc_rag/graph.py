"""LangGraph version: retrieve -> grade -> (rewrite -> retrieve)* -> generate -> verify -> (refuse).

    START -> retrieve -> grade_context
    grade_context --relevant, or retries exhausted--> generate
    grade_context --not relevant and retries < MAX--> rewrite_query
    rewrite_query --new query--> retrieve
    rewrite_query --same query as before--> generate   (another search would find nothing new)
    generate -> verify
    verify --supported--> END
    verify --not supported--> refuse -> END

The grader and verifier are small temperature-0 yes/no calls to the local model.
Every node appends a line to state["trace"] (and logs it), so the path a
question took can be read back after the run.
"""
import logging
import operator
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from generation_strategy import LLM_MODEL
from lc_rag.chain import LCRag, chunk_dicts
from lc_rag.prompts import REFUSAL_MESSAGE, docs_to_context, is_refusal

log = logging.getLogger("lc_rag.graph")

MAX_RETRIES = 2          # at most 2 query rewrites, so at most 3 retrievals
RECURSION_LIMIT = 25     # second guard: LangGraph aborts a run that exceeds this many steps


class RagState(TypedDict, total=False):
    question: str                       # the user's original question (never changed)
    k: int                              # chunks to retrieve
    rewritten_query: str                # latest rewritten search query ("" until a rewrite happens)
    chunks: List[Document]              # chunks from the latest retrieval
    relevant: bool                      # grader decision for the latest chunks
    answer: str
    retries: int                        # how many rewrites have happened so far
    stalled: bool                       # the latest rewrite repeated the previous query
    supported: bool                     # verifier decision for the answer
    trace: Annotated[List[str], operator.add]   # each node appends; operator.add concatenates


GRADE_PROMPT = """You are checking retrieved context for a question answering system.

Question: {question}

Context:
{context}

Does the context contain information that directly helps answer the question?
Reply with exactly one word: yes or no."""

VERIFY_PROMPT = """You are checking an answer against its source context.

Question: {question}

Context:
{context}

Answer: {answer}

Is every factual claim in the answer supported by the context?
Reply with exactly one word: yes or no."""

REWRITE_PROMPT = """Rewrite the question below as a short search query (at most 12 words, keywords only)
that is likely to match the wording used in the document. Return only the query.

Question: {question}
Previous query that did not find good context: {previous}"""


def parse_yes_no(text: str) -> bool:
    """Strict parse: True only if the reply's first word is 'yes'. Anything else is 'no'."""
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and words[0] == "yes"


def same_query(a: str, b: str) -> bool:
    """True if two queries are the same ignoring case, punctuation and spacing."""
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    return norm(a) == norm(b)


def clean_query(text: str, fallback: str) -> str:
    """First non-empty line, quotes and a leading 'query:' stripped; fall back if empty or too long."""
    for line in text.splitlines():
        line = re.sub(r"^\s*(search\s+)?query\s*:\s*", "", line.strip(), flags=re.I).strip(" \"'`")
        if line:
            return line if len(line) <= 200 else fallback
    return fallback


def make_judge_llm() -> BaseChatModel:
    # Same model, temperature 0; num_predict caps the reply at a few tokens (yes/no only).
    return ChatOllama(model=LLM_MODEL, temperature=0, num_predict=8, client_kwargs={"timeout": 120})


def build_graph(rag: LCRag, judge_llm: Optional[BaseChatModel] = None, max_retries: int = MAX_RETRIES):
    """Compile the StateGraph. `rag` supplies retrieval, generation and the rewrite model."""
    judge = judge_llm or make_judge_llm()

    def ask_yes_no(prompt: str) -> tuple:
        raw = judge.invoke(prompt).content.strip()
        return parse_yes_no(raw), raw

    def note(msg: str) -> Dict[str, List[str]]:
        log.info(msg)
        return {"trace": [msg]}

    # ---- nodes ----

    def retrieve(state: RagState) -> Dict[str, Any]:
        query = state.get("rewritten_query") or state["question"]
        docs = rag.retrieve(query, state["k"])
        pages = sorted({d.metadata["page_number"] for d in docs})
        return {"chunks": docs, **note(f"retrieve: query={query!r} -> {len(docs)} chunks, pages {pages}")}

    def grade_context(state: RagState) -> Dict[str, Any]:
        relevant, raw = ask_yes_no(GRADE_PROMPT.format(
            question=state["question"], context=docs_to_context(state["chunks"])))
        return {"relevant": relevant,
                **note(f"grade_context: relevant={'yes' if relevant else 'no'} (raw={raw!r}, retries={state['retries']}/{max_retries})")}

    def rewrite_query(state: RagState) -> Dict[str, Any]:
        previous = state.get("rewritten_query") or state["question"]
        raw = rag.llm.invoke(REWRITE_PROMPT.format(question=state["question"], previous=previous)).content
        query = clean_query(raw, fallback=state["question"])
        stalled = same_query(query, previous)
        suffix = "; unchanged, stopping retries" if stalled else ""
        return {"rewritten_query": query, "retries": state["retries"] + 1, "stalled": stalled,
                **note(f"rewrite_query: {previous!r} -> {query!r} (retries now {state['retries'] + 1}){suffix}")}

    def generate(state: RagState) -> Dict[str, Any]:
        answer = rag.generate(state["chunks"], state["question"])
        return {"answer": answer, **note(f"generate: {len(answer)} chars, generator_refused={is_refusal(answer)}")}

    def verify(state: RagState) -> Dict[str, Any]:
        if is_refusal(state["answer"]):
            # A refusal makes no claims to check. Route it to the refuse node so the
            # final text is the fixed refusal string.
            return {"supported": False, **note("verify: generator already refused, skipping the check")}
        supported, raw = ask_yes_no(VERIFY_PROMPT.format(
            question=state["question"], context=docs_to_context(state["chunks"]), answer=state["answer"]))
        return {"supported": supported, **note(f"verify: supported={'yes' if supported else 'no'} (raw={raw!r})")}

    def refuse(state: RagState) -> Dict[str, Any]:
        return {"answer": REFUSAL_MESSAGE, **note("refuse: returning the fixed refusal string")}

    # ---- conditional edges ----

    def route_after_grade(state: RagState) -> str:
        if state["relevant"] or state["retries"] >= max_retries:
            return "generate"
        return "rewrite_query"

    def route_after_rewrite(state: RagState) -> str:
        # A rewrite that repeats the previous query cannot retrieve anything new, so
        # skip the repeat retrieval and grading call and answer with what we have.
        return "generate" if state["stalled"] else "retrieve"

    def route_after_verify(state: RagState) -> str:
        return END if state["supported"] else "refuse"

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_context", grade_context)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_context")
    graph.add_conditional_edges("grade_context", route_after_grade,
                                {"generate": "generate", "rewrite_query": "rewrite_query"})
    graph.add_conditional_edges("rewrite_query", route_after_rewrite,
                                {"retrieve": "retrieve", "generate": "generate"})
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {END: END, "refuse": "refuse"})
    graph.add_edge("refuse", END)
    return graph.compile()


class GraphRag:
    """Runs the compiled graph and returns the same result shape as LCRag.answer."""

    def __init__(self, rag: LCRag, judge_llm: Optional[BaseChatModel] = None, max_retries: int = MAX_RETRIES):
        self.rag = rag
        self.graph = build_graph(rag, judge_llm=judge_llm, max_retries=max_retries)

    def answer(self, question: str, k: int = 10) -> Dict[str, Any]:
        final = self.graph.invoke(
            {"question": question, "k": k, "rewritten_query": "", "retries": 0, "stalled": False, "trace": []},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        return {
            "question": question,
            "answer": final["answer"],
            "retrieved_chunks": chunk_dicts(final["chunks"]),
            "refused": is_refusal(final["answer"]),
            "trace": final["trace"],
            "supported": final["supported"],
            "retries": final["retries"],
            "rewritten_query": final["rewritten_query"],
        }
