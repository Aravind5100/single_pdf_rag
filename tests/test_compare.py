"""Tests for the pure logic in eval/compare.py (no Ollama, no PDF)."""
import pytest

from eval.compare import ordered_pages, parse_trace, render_tables, score_record, summarize
from lc_rag.prompts import REFUSAL_MESSAGE

HAPPY_TRACE = [
    "retrieve: query='q' -> 6 chunks, pages [9, 19]",
    "grade_context: relevant=yes (raw='yes', retries=0/2)",
    "generate: 90 chars, generator_refused=False",
    "verify: supported=yes (raw='yes')",
]
REWRITE_TRACE = [
    "retrieve: query='q' -> 6 chunks, pages [2, 3]",
    "grade_context: relevant=no (raw='no', retries=0/2)",
    "rewrite_query: 'q' -> 'q2' (retries now 1)",
    "retrieve: query='q2' -> 6 chunks, pages [19]",
    "grade_context: relevant=yes (raw='yes', retries=1/2)",
    "generate: 90 chars, generator_refused=False",
    "verify: supported=no (raw='no')",
    "refuse: returning the fixed refusal string",
]
SKIPPED_TRACE = [
    "retrieve: query='q' -> 6 chunks, pages [4]",
    "grade_context: relevant=no (raw='no', retries=2/2)",
    "generate: 81 chars, generator_refused=True",
    "verify: generator already refused, skipping the check",
    "refuse: returning the fixed refusal string",
]


def test_parse_trace_happy_path():
    info = parse_trace(HAPPY_TRACE)
    assert info == {"grades": [True], "rewrites": 0, "generator_refused": False,
                    "verify": "yes", "first_pages": [9, 19]}


def test_parse_trace_rewrite_and_verifier_rejection():
    info = parse_trace(REWRITE_TRACE)
    assert info["grades"] == [False, True] and info["rewrites"] == 1
    assert info["verify"] == "no"
    assert info["first_pages"] == [2, 3]        # the FIRST retrieval, not the rewritten one


def test_parse_trace_skipped_verifier():
    info = parse_trace(SKIPPED_TRACE)
    assert info["verify"] == "skipped" and info["generator_refused"] is True


def test_parse_trace_none_for_versions_without_a_trace():
    assert parse_trace(None) is None


def test_ordered_pages_dedupes_in_retrieval_order():
    chunks = [{"page_number": 22}, {"page_number": 19}, {"page_number": 22}, {"page_number": 21}]
    assert ordered_pages(chunks) == [22, 19, 21]


def row(qid, answer_type, gt):
    return {"id": qid, "question": "q", "answer_type": answer_type, "ground_truth_pages": gt, "k": 6}


def result(answer, pages, trace=None):
    return {"answer": answer, "trace": trace,
            "retrieved_chunks": [{"page_number": p, "chunk_id": f"{p}_chunk_1"} for p in pages]}


def test_score_record_retrieval_metrics_for_answerable_and_none_for_negative():
    hit = score_record("chain", row("Q1", "policy", [19]), result("Pets are banned [Page 19].", [22, 19]), 2.0)
    assert hit["recall_at_k"] == 1.0 and hit["mrr"] == 0.5 and not hit["refused"]

    neg = score_record("chain", row("Q9", "negative", []), result(REFUSAL_MESSAGE, [5]), 1.0)
    assert neg["recall_at_k"] is None and neg["mrr"] is None and neg["refused"]


@pytest.fixture
def graph_records():
    def rec(qid, typ, gt, answer, pages, trace, latency):
        return score_record("graph", row(qid, typ, gt), result(answer, pages, trace), latency)

    return [
        rec("A1", "fact", [19], "Banned [Page 19]", [19], HAPPY_TRACE, 10.0),             # hit, grader yes
        rec("A2", "fact", [19], REFUSAL_MESSAGE, [19], REWRITE_TRACE, 20.0),              # first retrieval missed [19]
        rec("N1", "negative", [], REFUSAL_MESSAGE, [4], SKIPPED_TRACE, 30.0),             # grader no, correct refusal
        rec("N2", "negative", [], "Made up answer", [9], HAPPY_TRACE, 40.0),              # grader yes, verifier passed
    ]


def test_summarize_counts_rates_and_latency(graph_records):
    s = summarize(graph_records)

    assert (s["n_answerable"], s["n_negative"]) == (2, 2)
    assert s["recall_at_k"] == 1.0                           # both answerables ended with page 19 retrieved
    assert s["refused_negatives"] == 1 and s["refusal_correctness"] == 0.5
    assert s["refused_answerables"] == 1 and s["over_refusal"] == 0.5
    assert s["latency_mean_s"] == 25.0 and s["latency_median_s"] == 25.0 and s["latency_total_s"] == 100.0


def test_summarize_graph_decisions_use_the_first_grade_and_first_retrieval(graph_records):
    g = summarize(graph_records)["graph"]

    assert g["questions_with_rewrite"] == 1 and g["mean_rewrites"] == 0.25
    assert (g["grader_first_answerable_yes"], g["grader_first_answerable_no"]) == (1, 1)
    # Only A1's FIRST retrieval contained the gold page, and the grader said yes there.
    assert (g["grader_first_answerable_with_hit_yes"], g["grader_first_answerable_with_hit_no"]) == (1, 0)
    assert (g["grader_first_negative_yes"], g["grader_first_negative_no"]) == (1, 1)
    assert g["answerable_verifier_rejected"] == 1
    assert (g["negative_generator_refused"], g["negative_verifier_rejected"], g["negative_verifier_passed"]) == (1, 0, 1)


def test_baseline_style_records_have_no_graph_section():
    recs = [score_record("baseline", row("A1", "fact", [19]), result("Banned [Page 19]", [19]), 3.0)]
    assert "graph" not in summarize(recs)


def test_render_tables_shows_counts_next_to_rates(graph_records):
    text = render_tables({"graph": summarize(graph_records)})

    assert "Refusal correctness on negatives (n=2) | 1/2 (50%)" in text
    assert "Over-refusal on answerables (n=2) | 1/2 (50%)" in text
    assert "Mean latency per question (s) | 25.0" in text
    assert "LangGraph decisions" in text
