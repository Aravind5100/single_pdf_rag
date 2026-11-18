"""Routing tests for the LangGraph pipeline using scripted fake models (no Ollama)."""
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from lc_rag.graph import GraphRag, clean_query, parse_yes_no, same_query
from lc_rag.prompts import REFUSAL_MESSAGE


class FakeRag:
    """Stands in for LCRag: records retrieve calls and returns a scripted answer."""

    def __init__(self, answer="Pets are not allowed [Page 19].", rewrite="pets residence halls policy"):
        self.answer = answer
        # `rewrite` is one reply (returned every time) or a list (returned in order).
        self.llm = FakeListChatModel(responses=[rewrite] if isinstance(rewrite, str) else rewrite)
        self.retrieve_queries = []

    def retrieve(self, query, k):
        self.retrieve_queries.append(query)
        return [Document(page_content="context text", metadata={"page_number": 19, "chunk_id": "19_chunk_1"})]

    def generate(self, docs, question):
        return self.answer


def run(judge_replies, rag=None, max_retries=2):
    rag = rag or FakeRag()
    judge = FakeListChatModel(responses=judge_replies)
    out = GraphRag(rag, judge_llm=judge, max_retries=max_retries).answer("Are pets allowed?", k=6)
    return rag, out


def steps(out):
    return [line.split(":")[0] for line in out["trace"]]


def test_happy_path_goes_straight_through():
    rag, out = run(["yes", "yes"])      # grade, verify

    assert steps(out) == ["retrieve", "grade_context", "generate", "verify"]
    assert out["answer"] == "Pets are not allowed [Page 19]."
    assert out["supported"] is True and out["retries"] == 0 and not out["refused"]
    assert rag.retrieve_queries == ["Are pets allowed?"]


def test_irrelevant_context_triggers_a_rewrite_then_recovers():
    rag, out = run(["no", "yes", "yes"])   # grade no, grade yes, verify yes

    assert steps(out) == ["retrieve", "grade_context", "rewrite_query", "retrieve",
                          "grade_context", "generate", "verify"]
    assert rag.retrieve_queries == ["Are pets allowed?", "pets residence halls policy"]
    assert out["retries"] == 1 and out["rewritten_query"] == "pets residence halls policy"
    assert out["answer"] == "Pets are not allowed [Page 19]."


def test_retry_cap_stops_the_loop_after_two_distinct_rewrites():
    # Grader says "no" every time and each rewrite is different: 3 retrievals, 2 rewrites,
    # then generate anyway; the verifier (also "no") sends it to refuse.
    rag, out = run(["no"], rag=FakeRag(rewrite=["first query", "second query"]))

    assert rag.retrieve_queries == ["Are pets allowed?", "first query", "second query"]
    assert steps(out).count("rewrite_query") == 2
    assert out["retries"] == 2
    assert steps(out)[-3:] == ["generate", "verify", "refuse"]
    assert out["answer"] == REFUSAL_MESSAGE and out["refused"]


def test_repeated_rewrite_stops_the_loop_without_another_search_or_grade():
    # The fake returns the same rewrite every time, so the second rewrite repeats the first.
    rag, out = run(["no"])

    assert rag.retrieve_queries == ["Are pets allowed?", "pets residence halls policy"]   # no third search
    assert steps(out) == ["retrieve", "grade_context", "rewrite_query", "retrieve", "grade_context",
                          "rewrite_query", "generate", "verify", "refuse"]                # no third grade
    assert "unchanged, stopping retries" in out["trace"][5]
    assert out["retries"] == 2


def test_rewrite_identical_to_the_question_stops_at_the_first_rewrite():
    rag, out = run(["no"], rag=FakeRag(rewrite="are pets allowed"))    # same as the question, ignoring case

    assert rag.retrieve_queries == ["Are pets allowed?"]
    assert steps(out)[:4] == ["retrieve", "grade_context", "rewrite_query", "generate"]
    assert out["retries"] == 1


def test_max_retries_zero_never_rewrites():
    rag, out = run(["no", "yes"], max_retries=0)    # grade no -> generate, verify yes

    assert "rewrite_query" not in steps(out)
    assert len(rag.retrieve_queries) == 1
    assert out["answer"] == "Pets are not allowed [Page 19]."


def test_unsupported_answer_is_replaced_by_the_fixed_refusal():
    _, out = run(["yes", "no"])          # grade yes, verify no

    assert steps(out) == ["retrieve", "grade_context", "generate", "verify", "refuse"]
    assert out["answer"] == REFUSAL_MESSAGE
    assert out["supported"] is False and out["refused"]


def test_generator_refusal_skips_the_verifier_and_uses_the_fixed_string():
    rag = FakeRag(answer=REFUSAL_MESSAGE.replace("’", "'") + " [Page 3]")
    judge_calls = []

    class CountingJudge(FakeListChatModel):
        def _call(self, *args, **kwargs):
            judge_calls.append(1)
            return super()._call(*args, **kwargs)

    out = GraphRag(rag, judge_llm=CountingJudge(responses=["yes"])).answer("Parking fines?", k=6)

    assert len(judge_calls) == 1                      # only the grader ran
    assert steps(out)[-3:] == ["generate", "verify", "refuse"]
    assert out["answer"] == REFUSAL_MESSAGE           # normalized to the exact fixed string


def test_trace_lines_name_the_decision_taken():
    _, out = run(["no", "yes", "yes"])
    joined = "\n".join(out["trace"])

    assert "relevant=no" in joined and "relevant=yes" in joined
    assert "supported=yes" in joined and "retries now 1" in joined


# ---- helpers ----

def test_parse_yes_no_is_strict():
    assert parse_yes_no("yes") and parse_yes_no("Yes.") and parse_yes_no("YES, it is supported")
    assert not parse_yes_no("no") and not parse_yes_no("No, not supported")
    assert not parse_yes_no("") and not parse_yes_no("maybe") and not parse_yes_no("I think yes")


def test_same_query_ignores_case_punctuation_and_spacing():
    assert same_query("Capital of France", "capital  of france!")
    assert not same_query("capital of France", "capital of Germany")


def test_clean_query_strips_noise_and_falls_back():
    assert clean_query('"pets residence halls"\nextra line', "orig") == "pets residence halls"
    assert clean_query("Query: pet policy", "orig") == "pet policy"
    assert clean_query("   \n  ", "orig") == "orig"
    assert clean_query("x" * 300, "orig") == "orig"
