"""Prompt, refusal string and context formatting shared with the baseline.

The system prompt and the context layout are imported from the hand-written
generation_strategy module so the LangChain and LangGraph versions send the
model exactly the same text as the baseline.
"""
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from generation_strategy import SYSTEM_PROMPT, format_context

# The fixed refusal sentence from rule 4 of SYSTEM_PROMPT (curly apostrophe).
# tests/test_lc_prompts.py asserts this stays inside SYSTEM_PROMPT, so the two
# cannot drift apart.
REFUSAL_MESSAGE = "I don’t have enough information in the provided document to answer this question."

# Same layout as generation_strategy.build_prompt, with placeholders.
PROMPT_TEMPLATE = SYSTEM_PROMPT + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"

# The baseline sends one block of text through /api/generate, so the closest
# equivalent is a single human message (no separate system message).
ANSWER_PROMPT = ChatPromptTemplate.from_messages([("human", PROMPT_TEMPLATE)])


def docs_to_context(docs: Iterable[Document]) -> str:
    """Render documents with the baseline's own formatter: '[Page N]\\n<text>'."""
    return format_context(
        [{"page_number": d.metadata["page_number"], "text": d.page_content} for d in docs]
    )


def _normalize(text: str) -> str:
    return " ".join(text.replace("’", "'").lower().split())


_REFUSAL_CORE = _normalize(REFUSAL_MESSAGE)


def is_refusal(answer: str) -> bool:
    """True if the answer contains the fixed refusal sentence.

    Curly and straight apostrophes are treated as equal, because the model may
    reproduce the sentence with either.
    """
    return _REFUSAL_CORE in _normalize(answer)
