"""Tests for the pure logic in eval/compare.py (no Ollama, no PDF)."""
import argparse
from pathlib import Path

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


# ---- key facts, repeats, output paths ----

from eval.compare import add_repeat_stats, fact_found, normalize_text, out_paths, score_key_facts


def test_normalize_text_straightens_apostrophes_and_drops_commas():
    assert normalize_text("  Don’t, STOP\n now ") == "don't stop now"


def test_fact_found_accepts_any_alternative_case_insensitively():
    answer = normalize_text("Quiet hours start at 10 P.M., sunday to thursday.")
    assert fact_found("10:00 p.m.|10 p.m.", answer)
    assert fact_found("SUNDAY", answer)
    assert not fact_found("midnight|12:00 a.m.", answer)
    assert not fact_found("", answer)                     # an empty alternative never matches


def test_score_key_facts_none_without_facts_and_per_fact_otherwise():
    assert score_key_facts(None, "anything") is None
    assert score_key_facts([], "anything") is None
    assert score_key_facts(["fish", "10 gallons", "service animal"], "Only fish in a tank of 10 gallons") == [True, True, False]


def facts_row(qid, facts):
    r = row(qid, "fact", [19])
    r["key_facts"] = facts
    return r


def test_score_record_adds_key_fact_fields_for_answerables_only():
    ok = score_record("chain", facts_row("Q1", ["fish", "10 gallons"]), result("Fish in a 10 gallons tank [Page 19]", [19]), 1.0)
    assert ok["key_fact_hits"] == [True, True] and ok["key_fact_recall"] == 1.0

    partial = score_record("chain", facts_row("Q2", ["fish", "10 gallons"]), result("Only fish", [19]), 1.0)
    assert partial["key_fact_recall"] == 0.5

    neg_row = row("N1", "negative", [])
    neg_row["key_facts"] = ["ignored"]
    neg = score_record("chain", neg_row, result(REFUSAL_MESSAGE, [4]), 1.0)
    assert neg["key_fact_hits"] is None and neg["key_fact_recall"] is None


def test_summarize_reports_key_fact_recall_and_fully_correct_count():
    recs = [
        score_record("chain", facts_row("A", ["alpha", "beta"]), result("alpha beta", [19]), 1.0),     # 2/2
        score_record("chain", facts_row("B", ["alpha", "beta"]), result("alpha only", [19]), 1.0),     # 1/2
        score_record("chain", row("C", "fact", [19]), result("no facts defined", [19]), 1.0),
    ]
    s = summarize(recs)

    assert s["n_with_key_facts"] == 2
    assert s["key_fact_recall"] == 0.75
    assert s["answers_with_all_key_facts"] == 1


def make_run(latencies, answers):
    return [{"id": f"Q{i}", "latency_s": lat, "answer": ans} for i, (lat, ans) in enumerate(zip(latencies, answers))]


def test_add_repeat_stats_uses_all_repeats_for_latency_and_checks_answer_stability():
    run1 = make_run([10.0, 20.0], ["a", "b"])
    run2 = make_run([14.0, 24.0], ["a", "CHANGED"])
    out = add_repeat_stats({"latency_mean_s": 15.0}, [run1, run2])

    assert out["repeats"] == 2
    assert out["latency_mean_s"] == 17.0
    assert out["latency_mean_per_repeat_s"] == [15.0, 19.0]
    assert (out["results_stable_across_repeats"], out["n_questions"]) == (1, 2)


def test_add_repeat_stats_leaves_a_single_run_unchanged():
    summary = {"latency_mean_s": 15.0}
    assert add_repeat_stats(summary, [make_run([10.0], ["a"])]) == {"latency_mean_s": 15.0}


def test_render_tables_shows_key_fact_and_repeat_rows():
    recs = [score_record("chain", facts_row("A", ["alpha", "beta"]), result("alpha beta", [19]), 2.0)]
    s = summarize(recs)
    s = add_repeat_stats(s, [make_run([2.0], ["alpha beta"]), make_run([4.0], ["alpha beta"])])
    text = render_tables({"chain": s})

    assert "Key-fact recall, mean (n=1 answerable) | 1.000" in text
    assert "Answers containing all key facts | 1/1" in text
    assert "3.0 (runs: 2.0, 4.0)" in text
    assert "identical results in every repeat (answer, chunks, refusal, grades) | 1/1" in text


def test_out_paths_follow_the_prefix():
    p = out_paths("eval/compare_adversarial")
    assert str(p["results"]) == "eval/compare_adversarial_results.jsonl"
    assert str(p["summary"]) == "eval/compare_adversarial_summary.json"
    assert str(p["tables"]) == "eval/compare_adversarial_tables.md"


# ---- verifier check (pure logic) ----

from eval.verifier_check import render as render_verifier, summarize_verdicts


def test_summarize_verdicts_counts_accepted_and_rejected_per_kind():
    results = [
        {"kind": "control", "verifier_said_yes": True},
        {"kind": "control", "verifier_said_yes": True},
        {"kind": "contradiction", "verifier_said_yes": False},
        {"kind": "contradiction", "verifier_said_yes": True},
        {"kind": "fabrication", "verifier_said_yes": False},
    ]
    s = summarize_verdicts(results)

    assert s["control"] == {"n": 2, "accepted": 2, "rejected": 0, "flagged_by_number_check": 0, "rejected_by_either": 0}
    assert s["contradiction"] == {"n": 2, "accepted": 1, "rejected": 1, "flagged_by_number_check": 0, "rejected_by_either": 1}
    assert s["fabrication"] == {"n": 1, "accepted": 0, "rejected": 1, "flagged_by_number_check": 0, "rejected_by_either": 1}
    assert "| Seeded contradiction | 2 | 1 | 0 | 1 | reject |" in render_verifier(s)


def test_summarize_verdicts_counts_the_number_check_separately_and_in_combination():
    results = [
        {"kind": "fabrication", "verifier_said_yes": True, "verifier_raw": "yes", "grounding_issues": ["40"]},   # only the number check
        {"kind": "fabrication", "verifier_said_yes": False, "verifier_raw": "no", "grounding_issues": []},        # only the LLM
        {"kind": "fabrication", "verifier_said_yes": True, "verifier_raw": "yes", "grounding_issues": []},        # neither
    ]
    f = summarize_verdicts(results)["fabrication"]
    assert (f["rejected"], f["flagged_by_number_check"], f["rejected_by_either"]) == (1, 1, 2)


# ---- reviewer findings: whole-word matching, citations, normalization, repeats, rescore, CLI ----

import json

import pytest

from eval.compare import main as compare_main, positive_int, rescore, strip_citations


def test_no_matches_only_the_whole_word_not_not_or_enough():
    # Regression: 'no ' used to lose its space and match inside 'not', 'know' and 'enough'.
    assert score_key_facts(["no|not an official"], "No, it is not the channel") == [True]
    assert score_key_facts(["no"], "I don't have enough information to answer.") == [False]
    assert score_key_facts(["no"], "It is not allowed and I know that.") == [False]


def test_fixed_refusal_earns_no_key_fact_credit():
    assert score_key_facts(["no|cannot", "roommate"], REFUSAL_MESSAGE) == [False, False]
    assert score_key_facts(["informing|inform|policies"], REFUSAL_MESSAGE) == [False]   # not 'information'


def test_page_citations_cannot_satisfy_a_fact():
    assert strip_citations("Two lock-outs [Page 29], see (Page 12) and Page 17.").split() == ["Two", "lock-outs", ",", "see", "and", "."]
    assert score_key_facts(["two", "2", "25"], "Free lock-outs apply. [Page 25] [Page 2]") == [False, False, False]
    assert score_key_facts(["25"], "A $25.00 fee applies. [Page 29]") == [True]


def test_normalization_handles_hyphens_commas_and_parentheses():
    assert score_key_facts(["844-459-7405"], "Call (844) 459-7405 for help.") == [True]
    assert score_key_facts(["1500"], "one 1,500 ml bottle of wine") == [True]
    assert score_key_facts(["twelve 12"], "Twelve 12-oz beers are the limit") == [True]
    assert score_key_facts(["993-2380"], "(703) 993-2380") == [True]


def test_stems_must_be_listed_explicitly():
    assert score_key_facts(["policy|policies"], "residence hall policies apply") == [True]
    assert score_key_facts(["polic"], "residence hall policies apply") == [False]


def dataset_rows_with_key_facts():
    path = Path(__file__).resolve().parent.parent / "eval" / "eval_dataset_extended.jsonl"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return [r for r in rows if r.get("key_facts")]


def test_every_key_fact_is_found_in_the_gold_answer():
    for r in dataset_rows_with_key_facts():
        hits = score_key_facts(r["key_facts"], r["ground_truth_answer"])
        assert all(hits), f"{r['id']}: key facts not found in the gold answer: " \
                          f"{[f for f, h in zip(r['key_facts'], hits) if not h]}"


def test_no_key_fact_is_satisfied_by_the_refusal_sentence():
    for r in dataset_rows_with_key_facts():
        assert not any(score_key_facts(r["key_facts"], REFUSAL_MESSAGE)), r["id"]


def test_repeat_stability_looks_at_chunks_refusal_and_grades_not_just_the_answer():
    base = {"id": "Q1", "latency_s": 1.0, "answer": "a", "retrieved_chunk_ids": ["1_chunk_1"], "refused": False,
            "graph": {"grades": [True]}}
    same = add_repeat_stats({}, [[dict(base)], [dict(base)]])
    assert same["results_stable_across_repeats"] == 1

    for change in ({"retrieved_chunk_ids": ["2_chunk_1"]}, {"refused": True}, {"graph": {"grades": [False]}}):
        out = add_repeat_stats({}, [[dict(base)], [{**base, **change}]])
        assert out["results_stable_across_repeats"] == 0, change


def test_repeat_latency_total_covers_all_repeats():
    out = add_repeat_stats({"latency_total_s": 999.0},
                           [make_run([10.0, 20.0], ["a", "b"]), make_run([14.0, 24.0], ["a", "b"])])
    assert out["latency_total_s"] == 68.0 and out["latency_median_s"] == 17.0


def test_rescore_recomputes_key_facts_from_saved_answers(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({**row("Q1", "fact", [19]), "key_facts": ["alpha", "beta"]}) + "\n")
    prefix = str(tmp_path / "cmp")
    rec = score_record("chain", row("Q1", "fact", [19]), result("alpha only", [19]), 2.0)   # scored with no key_facts
    rec["repeat"] = 0
    Path(prefix + "_results.jsonl").write_text(json.dumps(rec) + "\n")
    Path(prefix + "_summary.json").write_text(json.dumps({"meta": {"dataset": "x"}, "summary": {}}))

    summary = rescore(prefix, dataset)

    assert summary["chain"]["key_fact_recall"] == 0.5 and summary["chain"]["answers_with_all_key_facts"] == 0
    saved = json.loads(Path(prefix + "_results.jsonl").read_text())
    assert saved["key_fact_hits"] == [True, False]
    assert json.loads(Path(prefix + "_summary.json").read_text())["meta"]["key_facts_rescored_from_saved_answers"]


def test_positive_int_rejects_zero_and_negative():
    assert positive_int("3") == 3
    for bad in ("0", "-2"):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(bad)


def test_non_default_runs_must_choose_an_output_prefix(capsys):
    with pytest.raises(SystemExit) as err:
        compare_main(["--k", "10"])
    assert "--out-prefix" in str(err.value)


# ---- structured graph fields (stop reason, refusal source, node timings) ----

def graph_result(answer, trace, **fields):
    r = result(answer, [19], trace)
    r.update(fields)
    return r


def test_graph_record_carries_structured_fields_when_present_and_none_when_absent():
    with_fields = score_record("graph", row("Q1", "fact", [19]),
                               graph_result("a", HAPPY_TRACE, stop_reason="relevant", refusal_source="",
                                            node_seconds={"retrieve": 0.1, "generate": 3.0}), 5.0)
    assert with_fields["graph"]["stop_reason"] == "relevant"
    assert with_fields["graph"]["node_seconds"] == {"retrieve": 0.1, "generate": 3.0}

    old_style = score_record("graph", row("Q1", "fact", [19]), graph_result("a", HAPPY_TRACE), 5.0)
    assert old_style["graph"]["stop_reason"] is None and old_style["graph"]["node_seconds"] is None


def test_summarize_reports_sources_stop_reasons_and_mean_node_seconds():
    recs = [
        score_record("graph", row("A", "fact", [19]),
                     graph_result("a", HAPPY_TRACE, stop_reason="relevant", refusal_source="",
                                  node_seconds={"retrieve": 1.0, "generate": 4.0}), 5.0),
        score_record("graph", row("N", "negative", []),
                     graph_result(REFUSAL_MESSAGE, SKIPPED_TRACE, stop_reason="retries_exhausted",
                                  refusal_source="generator", node_seconds={"retrieve": 3.0, "grade_context": 2.0}), 5.0),
        score_record("graph", row("M", "negative", []),
                     graph_result(REFUSAL_MESSAGE, REWRITE_TRACE, stop_reason="stalled",
                                  refusal_source="verifier", node_seconds={"retrieve": 2.0}), 5.0),
    ]
    g = summarize(recs)["graph"]

    assert (g["refused_by_generator"], g["refused_by_verifier"]) == (1, 1)
    assert g["stop_reasons"] == {"relevant": 1, "retries_exhausted": 1, "stalled": 1}
    assert g["mean_node_seconds"] == {"generate": 1.33, "grade_context": 0.67, "retrieve": 2.0}


def test_summarize_works_on_results_saved_before_these_fields_existed():
    recs = [score_record("graph", row("A", "fact", [19]), graph_result("a", HAPPY_TRACE), 5.0)]
    g = summarize(recs)["graph"]
    assert "refused_by_generator" not in g and "stop_reasons" not in g and "mean_node_seconds" not in g
    render_tables({"graph": summarize(recs)})        # must not raise


# ---- verifier check helpers ----

from eval.verifier_check import build_candidates, is_clean_reply


def test_is_clean_reply_accepts_only_a_bare_yes_or_no():
    assert is_clean_reply("yes") and is_clean_reply("No.") and is_clean_reply("  YES!\n")
    assert not is_clean_reply("Yes, it is supported") and not is_clean_reply("") and not is_clean_reply("maybe")


def test_build_candidates_appends_the_invented_sentence_to_the_control():
    cases = [
        {"id": "Q1", "kind": "contradiction", "answer": "Yes, cats are allowed."},
        {"id": "Q1", "kind": "fabrication", "append": "A $40 fee applies.", "invented": "$40"},
    ]
    candidates, skipped = build_candidates("Only fish are allowed.", cases, context="[Page 19] fish tanks only")

    assert skipped == []
    assert candidates == [
        {"kind": "control", "answer": "Only fish are allowed."},
        {"kind": "contradiction", "answer": "Yes, cats are allowed."},
        {"kind": "fabrication", "answer": "Only fish are allowed. A $40 fee applies."},
    ]


def test_build_candidates_skips_a_case_whose_invented_text_is_in_the_context():
    cases = [{"id": "Q1", "kind": "fabrication", "append": "A $40 fee applies.", "invented": "$40"}]
    candidates, skipped = build_candidates("control", cases, context="the fee is $40 per year")

    assert [c["kind"] for c in candidates] == ["control"]
    assert skipped[0]["id"] == "Q1" and "$40" in skipped[0]["reason"]


def test_summarize_verdicts_counts_unclean_replies():
    results = [{"kind": "control", "verifier_said_yes": True, "verifier_raw": "yes"},
               {"kind": "fabrication", "verifier_said_yes": False, "verifier_raw": "No, it adds a claim"}]
    assert summarize_verdicts(results)["unclean_replies"] == 1


# ---- second review round: matcher edge cases, trace parsing, rescore, CLI ----

GROUNDING_TRACE = [
    "retrieve: query='q' -> 6 chunks, pages [19]",
    "grade_context: relevant=yes (raw='yes', retries=0/2)",
    "generate: 40 chars, generator_refused=False",
    "verify: numbers ['40'] in the answer are not in the context, rejecting without the model check",
    "refuse: returning the fixed refusal string",
]


def test_parse_trace_marks_a_number_check_rejection_as_grounding_not_an_llm_verdict():
    assert parse_trace(GROUNDING_TRACE)["verify"] == "grounding"


def test_number_check_rejections_are_counted_apart_from_llm_verifier_rejections():
    rec = score_record("graph", row("A", "fact", [19]),
                       graph_result("x", GROUNDING_TRACE, refusal_source="grounding_check",
                                    stop_reason="relevant", node_seconds={"generate": 1.0}), 5.0)
    g = summarize([rec])["graph"]

    assert g["answerable_grounding_rejected"] == 1 and g["answerable_verifier_rejected"] == 0
    assert g["refused_by_grounding_check"] == 1


def test_step_limit_refusals_are_counted_so_the_sources_add_up():
    rec = score_record("graph", row("N", "negative", []),
                       graph_result(REFUSAL_MESSAGE, ["graph: step limit 25 exceeded, returning the fixed refusal string"],
                                    refusal_source="step_limit", stop_reason="step_limit", node_seconds={}), 5.0)
    g = summarize([rec])["graph"]

    assert g["refused_by_step_limit"] == 1
    assert g["refused_by_generator"] + g["refused_by_verifier"] + g["refused_by_grounding_check"] + g["refused_by_step_limit"] == 1


def test_matcher_handles_dashes_and_am_pm_spellings():
    assert score_key_facts(["844-459-7405"], "Call 844\u2013459\u20137405") == [True]
    assert score_key_facts(["10:00 p.m.|10 p.m."], "Quiet hours start at 10:00 PM") == [True]
    assert score_key_facts(["8:00 a.m."], "until 8:00 AM") == [True]


def test_a_fact_number_does_not_match_inside_a_decimal():
    assert score_key_facts(["25"], "The fee is $100.25") == [False]
    assert score_key_facts(["25"], "The fee is $25.00 per lock-out") == [True]


def test_more_citation_forms_cannot_satisfy_a_fact():
    for citation in ("[Pages 18 and 25]", "[Page 19\u201325]", "p. 25", "(page 25)"):
        assert score_key_facts(["25"], f"No fee applies. {citation}") == [False], citation


def test_rescore_defaults_to_the_dataset_recorded_in_the_saved_summary(tmp_path):
    dataset = tmp_path / "other.jsonl"
    dataset.write_text(json.dumps({**row("Z9", "fact", [19]), "key_facts": ["alpha"]}) + "\n")
    prefix = str(tmp_path / "run")
    rec = score_record("chain", row("Z9", "fact", [19]), result("alpha", [19]), 1.0)
    rec["repeat"] = 0
    Path(prefix + "_results.jsonl").write_text(json.dumps(rec) + "\n")
    Path(prefix + "_summary.json").write_text(json.dumps({"meta": {"dataset": str(dataset)}, "summary": {}}))

    summary = rescore(prefix)                       # no dataset argument: it must read meta["dataset"]

    assert summary["chain"]["key_fact_recall"] == 1.0
    assert not Path(prefix + "_results.jsonl.tmp").exists()      # the temp file was swapped in


def test_a_default_run_passes_the_output_guard_and_a_prefixed_one_too(monkeypatch):
    monkeypatch.delenv("PDF_PATH", raising=False)
    for argv in ([], ["--k", "10", "--out-prefix", "eval/compare_k10"]):
        with pytest.raises(SystemExit) as err:
            compare_main(argv)
        assert "PDF_PATH" in str(err.value) and "--out-prefix" not in str(err.value), argv
