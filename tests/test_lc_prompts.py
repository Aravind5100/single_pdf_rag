from langchain_core.documents import Document

import generation_strategy
from lc_rag.prompts import ANSWER_PROMPT, REFUSAL_MESSAGE, docs_to_context, is_refusal


def test_refusal_message_is_the_one_inside_the_baseline_prompt():
    assert REFUSAL_MESSAGE in generation_strategy.SYSTEM_PROMPT


def test_langchain_prompt_is_byte_identical_to_the_baseline_prompt():
    docs = [
        Document(page_content="Pets are not allowed.", metadata={"page_number": 19}),
        Document(page_content="Fish tanks are fine.", metadata={"page_number": 22}),
    ]
    question = "Are pets allowed?"

    lc_text = ANSWER_PROMPT.format_messages(context=docs_to_context(docs), question=question)[0].content
    baseline_text = generation_strategy.build_prompt(
        generation_strategy.format_context([
            {"page_number": 19, "text": "Pets are not allowed."},
            {"page_number": 22, "text": "Fish tanks are fine."},
        ]),
        question,
    )

    assert lc_text == baseline_text


def test_is_refusal_accepts_curly_and_straight_apostrophes():
    assert is_refusal(REFUSAL_MESSAGE)
    assert is_refusal(REFUSAL_MESSAGE.replace("’", "'"))
    assert is_refusal("  " + REFUSAL_MESSAGE.upper() + " [Page 3]")


def test_is_refusal_rejects_real_answers():
    assert not is_refusal("Pets are not allowed in residence halls [Page 19].")
    assert not is_refusal("")
