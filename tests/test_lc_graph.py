"""Routing tests for the LangGraph pipeline using scripted fake models (no Ollama)."""
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from lc_rag.graph import (GraphRag, clean_query, node_seconds, parse_yes_no, recursion_limit_for,
                          same_text, seen_before)
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
    rag, out = run(["no"], rag=FakeRag(rewrite="Are pets allowed?"))    # exactly the question

    assert rag.retrieve_queries == ["Are pets allowed?"]
    assert steps(out)[:4] == ["retrieve", "grade_context", "rewrite_query", "generate"]
    assert out["retries"] == 1 and out["stop_reason"] == "stalled"


def test_a_case_or_punctuation_change_is_a_new_query_and_is_searched():
    # Different text can embed differently, so it is not treated as a repeat.
    rag, out = run(["no"], rag=FakeRag(rewrite=["are pets allowed", "are pets allowed!"]))

    assert rag.retrieve_queries == ["Are pets allowed?", "are pets allowed", "are pets allowed!"]
    assert out["stop_reason"] == "retries_exhausted"


def test_rewrite_that_cycles_back_to_the_original_question_stops_the_loop():
    # A -> B -> A: the second rewrite repeats the ORIGINAL question, not the previous query.
    rag, out = run(["no"], rag=FakeRag(rewrite=["first query", "Are pets allowed?"]))

    assert rag.retrieve_queries == ["Are pets allowed?", "first query"]          # no third search
    assert "repeats an earlier query, stopping retries" in out["trace"][5]
    assert out["stop_reason"] == "stalled" and out["queries"] == ["Are pets allowed?", "first query", "Are pets allowed?"]


def test_empty_rewrite_falls_back_to_the_question_and_is_caught_at_rewrite_two():
    # clean_query falls back to the original question for a blank reply; at rewrite 2 that differs from the
    # previous query but repeats the first one, which the old previous-query-only check missed.
    rag, out = run(["no"], rag=FakeRag(rewrite=["first query", "   "]))

    assert rag.retrieve_queries == ["Are pets allowed?", "first query"]
    assert out["stop_reason"] == "stalled"


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


def test_same_text_collapses_whitespace_but_not_case_or_punctuation():
    assert same_text("capital  of\nFrance", "capital of France")
    assert not same_text("Capital of France", "capital of France")
    assert not same_text("capital of France", "capital of France!")


def test_seen_before_checks_every_prior_query():
    assert seen_before("b", ["a", "b", "c"]) and not seen_before("d", ["a", "b", "c"])


def test_clean_query_strips_noise_and_falls_back():
    assert clean_query('"pets residence halls"\nextra line', "orig") == "pets residence halls"
    assert clean_query("Query: pet policy", "orig") == "pet policy"
    assert clean_query("   \n  ", "orig") == "orig"
    assert clean_query("x" * 300, "orig") == "orig"


# ---- state fields, timings, step limit ----

def test_happy_path_state_fields():
    _, out = run(["yes", "yes"])
    assert out["stop_reason"] == "relevant" and out["refusal_source"] == "" and out["queries"] == ["Are pets allowed?"]


def test_retries_exhausted_stop_reason_and_verifier_refusal_source():
    _, out = run(["no"], rag=FakeRag(rewrite=["first query", "second query"]))   # grader and verifier always "no"
    assert out["stop_reason"] == "retries_exhausted"
    assert out["refusal_source"] == "verifier"          # the generator answered; the verifier overrode it


def test_generator_refusal_is_attributed_to_the_generator():
    rag = FakeRag(answer=REFUSAL_MESSAGE)
    _, out = run(["yes"], rag=rag)
    assert out["refusal_source"] == "generator" and out["refused"]


def test_node_seconds_sums_repeated_nodes():
    assert node_seconds(["retrieve=0.100", "grade_context=0.250", "retrieve=0.200"]) == {
        "retrieve": 0.3, "grade_context": 0.25}


def test_every_executed_node_is_timed():
    _, out = run(["no", "yes", "yes"])      # retrieve, grade, rewrite, retrieve, grade, generate, verify
    assert set(out["node_seconds"]) == {"retrieve", "grade_context", "rewrite_query", "generate", "verify"}
    assert all(v >= 0 for v in out["node_seconds"].values())


def test_recursion_limit_always_exceeds_the_longest_path():
    for max_retries in range(0, 30):
        assert recursion_limit_for(max_retries) > 3 * max_retries + 5
    assert recursion_limit_for(2) == 25                  # never below the default minimum


def test_a_large_max_retries_still_finishes_without_hitting_the_step_limit():
    rewrites = [f"distinct query {i}" for i in range(10)]
    rag, out = run(["no"], rag=FakeRag(rewrite=rewrites), max_retries=10)

    assert out["retries"] == 10 and len(rag.retrieve_queries) == 11
    assert out["stop_reason"] == "retries_exhausted" and out["refused"]


def test_exceeding_the_step_limit_returns_the_fixed_refusal_instead_of_raising():
    rag = FakeRag()
    judge = FakeListChatModel(responses=["yes"])
    out = GraphRag(rag, judge_llm=judge, recursion_limit=3).answer("Are pets allowed?", k=6)

    assert out["answer"] == REFUSAL_MESSAGE and out["refused"]
    assert out["stop_reason"] == "step_limit" and "step limit 3 exceeded" in out["trace"][0]


# ---- deterministic number check inside verify ----

def test_invented_number_is_rejected_by_the_number_check_without_calling_the_model_verifier():
    rag = FakeRag(answer="A $40 registration fee applies. [Page 19]")        # context text has no '40'
    calls = []

    class Judge(FakeListChatModel):
        def _call(self, *args, **kwargs):
            calls.append(1)
            return super()._call(*args, **kwargs)

    out = GraphRag(rag, judge_llm=Judge(responses=["yes"])).answer("Are pets allowed?", k=6)

    assert len(calls) == 1                                      # only the grader ran, not the verifier
    assert out["answer"] == REFUSAL_MESSAGE and out["refusal_source"] == "grounding_check"
    assert "numbers ['40'] in the answer are not in the context" in " ".join(out["trace"])


def test_a_number_that_is_in_the_context_passes_through_to_the_model_verifier():
    class RagWithNumber(FakeRag):
        def retrieve(self, query, k):
            self.retrieve_queries.append(query)
            return [Document(page_content="A $40 fee applies.", metadata={"page_number": 19, "chunk_id": "19_chunk_1"})]

    rag = RagWithNumber(answer="A $40 fee applies. [Page 19]")
    _, out = run(["yes", "yes"], rag=rag)                       # grade yes, verify yes

    assert out["refusal_source"] == "" and out["answer"] == "A $40 fee applies. [Page 19]"


def test_a_page_label_number_does_not_ground_an_invented_number_in_the_graph():
    rag = FakeRag(answer="A $19 fee applies.")          # the fake chunk is labeled page 19 but its text has no digits
    out = GraphRag(rag, judge_llm=FakeListChatModel(responses=["yes"])).answer("Are pets allowed?", k=6)

    assert out["refusal_source"] == "grounding_check" and out["answer"] == REFUSAL_MESSAGE


def test_a_number_from_the_question_is_not_rejected_by_the_number_check():
    rag = FakeRag(answer="Residents under 18 are covered. [Page 19]")
    out = GraphRag(rag, judge_llm=FakeListChatModel(responses=["yes"])).answer("Are residents under 18 covered?", k=6)

    assert out["refusal_source"] == "" and out["answer"].startswith("Residents under 18")
