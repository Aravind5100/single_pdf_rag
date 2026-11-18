"""LangGraph version: retrieve -> grade -> (rewrite -> retrieve)* -> generate -> verify -> (refuse).

    START -> retrieve -> grade_context
    grade_context --relevant, or retries exhausted--> generate
    grade_context --not relevant and retries < MAX--> rewrite_query
    rewrite_query --new query--> retrieve
    rewrite_query --a query already searched--> generate   (another search would find nothing new)
    generate -> verify
    verify --supported--> END
    verify --not supported--> refuse -> END
    (verify first checks that every number in the answer appears in the context, then asks the model)

The grader and verifier are small temperature-0 yes/no calls to the local model.
Every node appends a line to state["trace"] (and logs it), so the path a
question took can be read back after the run. Nodes also record how long they
took (state["timings"]), and the state records why the loop stopped
(stop_reason) and who refused (refusal_source).
"""
import logging
import operator
import re
import time
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from generation_strategy import LLM_MODEL
from lc_rag.chain import LCRag, chunk_dicts
from lc_rag.grounding import ungrounded_numbers
from lc_rag.prompts import REFUSAL_MESSAGE, docs_to_context, is_refusal

log = logging.getLogger("lc_rag.graph")

MAX_RETRIES = 2          # at most 2 query rewrites, so at most 3 retrievals
RECURSION_LIMIT = 25     # minimum step limit; second guard: LangGraph aborts a run that exceeds this many steps


def recursion_limit_for(max_retries: int) -> int:
    """Step limit that always exceeds the longest possible path.

    The longest path is retrieve, grade (then rewrite, retrieve, grade) once per retry, then generate,
    verify, refuse: 3 * max_retries + 5 steps. One extra step of margin, and never below RECURSION_LIMIT.
    """
    return max(RECURSION_LIMIT, 3 * max_retries + 6)


class RagState(TypedDict, total=False):
    question: str                       # the user's original question (never changed)
    k: int                              # chunks to retrieve
    queries: Annotated[List[str], operator.add]  # every query searched, the question first
    rewritten_query: str                # latest rewritten search query ("" until a rewrite happens)
    chunks: List[Document]              # chunks from the latest retrieval
    relevant: bool                      # grader decision for the latest chunks
    answer: str
    retries: int                        # how many rewrites have happened so far
    stalled: bool                       # the latest rewrite repeated a query that was already searched
    stop_reason: str                    # why the loop stopped: relevant | retries_exhausted | stalled
    supported: bool                     # verifier decision for the answer
    refusal_source: str                 # who refused: generator | grounding_check | verifier | step_limit | ""
    trace: Annotated[List[str], operator.add]   # each node appends; operator.add concatenates
    timings: Annotated[List[str], operator.add]  # "node=seconds" for every node execution


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


def same_text(a: str, b: str) -> bool:
    """Exact match after collapsing whitespace. Case and punctuation differences count as different
    queries, because they can change what the embedding search returns."""
    return " ".join(a.split()) == " ".join(b.split())


def seen_before(query: str, queries: List[str]) -> bool:
    """True if `query` is identical to any query already searched (including the original question)."""
    return any(same_text(query, q) for q in queries)


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

    def timed(name: str, fn):
        """Wrap a node so its wall-clock seconds are appended to state["timings"]."""
        def wrapper(state: RagState) -> Dict[str, Any]:
            start = time.perf_counter()
            out = dict(fn(state))
            out["timings"] = [f"{name}={time.perf_counter() - start:.3f}"]
            return out
        return wrapper

    # ---- nodes ----

    def retrieve(state: RagState) -> Dict[str, Any]:
        query = state["queries"][-1]
        docs = rag.retrieve(query, state["k"])
        pages = sorted({d.metadata["page_number"] for d in docs})
        return {"chunks": docs, **note(f"retrieve: query={query!r} -> {len(docs)} chunks, pages {pages}")}

    def grade_context(state: RagState) -> Dict[str, Any]:
        relevant, raw = ask_yes_no(GRADE_PROMPT.format(
            question=state["question"], context=docs_to_context(state["chunks"])))
        stop = "relevant" if relevant else ("retries_exhausted" if state["retries"] >= max_retries else "")
        return {"relevant": relevant, "stop_reason": stop,
                **note(f"grade_context: relevant={'yes' if relevant else 'no'} (raw={raw!r}, retries={state['retries']}/{max_retries})")}

    def rewrite_query(state: RagState) -> Dict[str, Any]:
        previous = state["queries"][-1]
        raw = rag.llm.invoke(REWRITE_PROMPT.format(question=state["question"], previous=previous)).content
        query = clean_query(raw, fallback=state["question"])
        stalled = seen_before(query, state["queries"])
        if not stalled:
            suffix = ""
        elif same_text(query, previous):
            suffix = "; unchanged, stopping retries"
        else:
            suffix = "; repeats an earlier query, stopping retries"
        return {"rewritten_query": query, "queries": [query], "retries": state["retries"] + 1, "stalled": stalled,
                "stop_reason": "stalled" if stalled else "",
                **note(f"rewrite_query: {previous!r} -> {query!r} (retries now {state['retries'] + 1}){suffix}")}

    def generate(state: RagState) -> Dict[str, Any]:
        answer = rag.generate(state["chunks"], state["question"])
        return {"answer": answer, **note(f"generate: {len(answer)} chars, generator_refused={is_refusal(answer)}")}

    def verify(state: RagState) -> Dict[str, Any]:
        if is_refusal(state["answer"]):
            # A refusal makes no claims to check. Route it to the refuse node so the
            # final text is the fixed refusal string.
            return {"supported": False, "refusal_source": "generator",
                    **note("verify: generator already refused, skipping the check")}
        context = docs_to_context(state["chunks"])
        # Compare against the chunk text only: docs_to_context adds "[Page N]" labels, and a retrieved
        # page's number must not count as grounding for an answer.
        chunk_text = "\n".join(d.page_content for d in state["chunks"])
        missing = ungrounded_numbers(state["answer"], chunk_text, question=state["question"])
        if missing:
            # A number in the answer that is nowhere in the context is invented. Reject without asking the model.
            return {"supported": False, "refusal_source": "grounding_check",
                    **note(f"verify: numbers {missing} in the answer are not in the context, rejecting without the model check")}
        supported, raw = ask_yes_no(VERIFY_PROMPT.format(
            question=state["question"], context=context, answer=state["answer"]))
        return {"supported": supported, "refusal_source": "" if supported else "verifier",
                **note(f"verify: supported={'yes' if supported else 'no'} (raw={raw!r})")}

    def refuse(state: RagState) -> Dict[str, Any]:
        return {"answer": REFUSAL_MESSAGE, **note("refuse: returning the fixed refusal string")}

    # ---- conditional edges ----

    def route_after_grade(state: RagState) -> str:
        if state["relevant"] or state["retries"] >= max_retries:
            return "generate"
        return "rewrite_query"

    def route_after_rewrite(state: RagState) -> str:
        # A rewrite identical to a query already searched cannot retrieve anything new, so
        # skip the repeat retrieval and grading call and answer with what we have.
        return "generate" if state["stalled"] else "retrieve"

    def route_after_verify(state: RagState) -> str:
        return END if state["supported"] else "refuse"

    graph = StateGraph(RagState)
    for name, fn in [("retrieve", retrieve), ("grade_context", grade_context), ("rewrite_query", rewrite_query),
                     ("generate", generate), ("verify", verify), ("refuse", refuse)]:
        graph.add_node(name, timed(name, fn))

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


def node_seconds(timings: List[str]) -> Dict[str, float]:
    """Total seconds per node from 'node=seconds' entries (a node that ran twice is summed)."""
    out: Dict[str, float] = {}
    for entry in timings:
        name, _, secs = entry.partition("=")
        out[name] = round(out.get(name, 0.0) + float(secs), 3)
    return out


class GraphRag:
    """Runs the compiled graph and returns the same result shape as LCRag.answer.

    If a run ever exceeds the step limit it returns the fixed refusal instead of raising.
    """

    def __init__(self, rag: LCRag, judge_llm: Optional[BaseChatModel] = None, max_retries: int = MAX_RETRIES,
                 recursion_limit: Optional[int] = None):
        self.rag = rag
        self.graph = build_graph(rag, judge_llm=judge_llm, max_retries=max_retries)
        self.recursion_limit = recursion_limit or recursion_limit_for(max_retries)

    def answer(self, question: str, k: int = 10) -> Dict[str, Any]:
        initial = {"question": question, "k": k, "queries": [question], "rewritten_query": "", "retries": 0,
                   "stalled": False, "stop_reason": "", "refusal_source": "", "trace": [], "timings": []}
        try:
            final = self.graph.invoke(initial, config={"recursion_limit": self.recursion_limit})
        except GraphRecursionError:
            log.warning("graph exceeded its step limit (%s); returning the fixed refusal", self.recursion_limit)
            return {
                "question": question, "answer": REFUSAL_MESSAGE, "retrieved_chunks": [], "refused": True,
                "trace": [f"graph: step limit {self.recursion_limit} exceeded, returning the fixed refusal string"],
                "supported": False, "retries": None, "rewritten_query": "", "queries": [question],
                "stop_reason": "step_limit", "refusal_source": "step_limit", "node_seconds": {},
            }
        return {
            "question": question,
            "answer": final["answer"],
            "retrieved_chunks": chunk_dicts(final["chunks"]),
            "refused": is_refusal(final["answer"]),
            "trace": final["trace"],
            "supported": final["supported"],
            "retries": final["retries"],
            "rewritten_query": final["rewritten_query"],
            "queries": final["queries"],
            "stop_reason": final.get("stop_reason", ""),
            "refusal_source": final.get("refusal_source", ""),
            "node_seconds": node_seconds(final.get("timings", [])),
        }
