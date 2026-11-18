"""Compare the hand-written baseline, the LangChain chain and the LangGraph pipeline.

Run from the repository root:
    PDF_PATH=/path/to/file.pdf python -m eval.compare
    PDF_PATH=... python -m eval.compare --versions baseline chain --limit 5
    PDF_PATH=... python -m eval.compare --repeats 2          # repeat the run to see latency variation
    PDF_PATH=... python -m eval.compare --k 10 --versions baseline chain --out-prefix eval/compare_k10
    PDF_PATH=... python -m eval.compare --dataset eval/eval_dataset_adversarial.jsonl --out-prefix eval/compare_adversarial

For every question each version runs its full pipeline (retrieve, and generate
or refuse). The retrieved chunks of that same run are scored, so the retrieval
numbers describe what the pipeline really returned.

Metrics per version
  * Recall@K and MRR on answerable questions (page level, same functions as eval_runner).
  * Refusal correctness: share of negative questions whose answer is the fixed refusal sentence.
  * Over-refusal: share of answerable questions whose answer is the fixed refusal sentence.
  * Mean and median latency per question, in seconds, for the whole pipeline call.
  * Key-fact recall: for answerable questions whose dataset row has `key_facts`, the share of those
    facts found in the answer (case-insensitive whole-word match, page citations ignored; "a|b" means
    either is accepted).
    This is a cheap objective check for facts the answer must contain. It cannot catch a wrong claim
    that sits next to the right facts, so it is not a full correctness judgment (see eval/COMPARISON.md).
    Existing results can be re-scored from their saved answers with --rescore (no models are run).
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from eval.eval_runner import (
    build_rag_artifacts,
    load_jsonl,
    recall_at_k,
    reciprocal_rank,
    write_json,
)
from generation_strategy import LLM_MODEL, answer_question
from lc_rag.grounding import strip_citations
from lc_rag.prompts import is_refusal
from ollama_check import OllamaNotReady, check_ollama

DEFAULT_DATASET = Path("eval/eval_dataset_extended.jsonl")
DEFAULT_PREFIX = "eval/compare"


def out_paths(prefix: str) -> Dict[str, Path]:
    return {"results": Path(f"{prefix}_results.jsonl"), "summary": Path(f"{prefix}_summary.json"),
            "tables": Path(f"{prefix}_tables.md")}


VERSIONS = ["baseline", "chain", "graph"]
LABELS = {"baseline": "Hand-written", "chain": "LangChain", "graph": "LangGraph"}

AnswerFn = Callable[[str, int], Dict[str, Any]]


# ---------------- version runners ----------------

def make_runners(pdf_path: str, versions: List[str]) -> Dict[str, AnswerFn]:
    runners: Dict[str, AnswerFn] = {}

    if "baseline" in versions:
        embedded_chunks, index = build_rag_artifacts(pdf_path)

        def baseline(question: str, k: int) -> Dict[str, Any]:
            r = answer_question(question, index, embedded_chunks, top_k=k)
            return {"answer": r["answer"], "retrieved_chunks": r["retrieved_chunks"], "trace": None}

        runners["baseline"] = baseline

    if "chain" in versions or "graph" in versions:
        from lc_rag.chain import LCRag
        from lc_rag.graph import GraphRag

        rag = LCRag(pdf_path, log=print)
        if "chain" in versions:
            runners["chain"] = rag.answer
        if "graph" in versions:
            runners["graph"] = GraphRag(rag).answer

    return runners


# ---------------- scoring ----------------

def ordered_pages(chunks: List[Dict[str, Any]]) -> List[int]:
    seen, pages = set(), []
    for c in chunks:
        p = int(c["page_number"])
        if p not in seen:
            seen.add(p)
            pages.append(p)
    return pages


def parse_trace(trace: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """Pull the graph's decisions out of its trace lines (the format is defined in lc_rag/graph.py)."""
    if trace is None:
        return None
    info: Dict[str, Any] = {"grades": [], "rewrites": 0, "generator_refused": None, "verify": None,
                            "first_pages": None}
    for line in trace:
        if line.startswith("retrieve:") and info["first_pages"] is None:
            m = re.search(r"pages \[([\d, ]*)\]", line)
            info["first_pages"] = [int(x) for x in m.group(1).split(",") if x.strip()] if m else []
        elif line.startswith("grade_context:"):
            info["grades"].append("relevant=yes" in line)
        elif line.startswith("rewrite_query:"):
            info["rewrites"] += 1
        elif line.startswith("generate:"):
            info["generator_refused"] = "generator_refused=True" in line
        elif line.startswith("verify:"):
            if "skipping the check" in line:
                info["verify"] = "skipped"
            elif "rejecting without the model check" in line:
                info["verify"] = "grounding"          # the number check rejected it; the LLM verifier never ran
            else:
                info["verify"] = "yes" if "supported=yes" in line else "no"
    return info


def normalize_text(text: str) -> str:
    """Lowercase, straighten apostrophes, write 'a.m.'/'p.m.' as 'am'/'pm', remove thousands commas
    ('1,500' -> '1500'), and turn other commas, brackets and hyphens or dashes into spaces
    ('844-459-7405' -> '844 459 7405')."""
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"\b([ap])\.m\.?", r"\1m", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[,()\[\]\-\u2013\u2014]", " ", text)
    return " ".join(text.split())


def fact_found(fact: str, answer_norm: str) -> bool:
    """True if any '|'-separated alternative of `fact` appears in the normalized answer as a whole word
    or phrase (so 'no' does not match 'not' or 'enough'). Write stems as explicit alternatives."""
    for alt in fact.split("|"):
        norm = normalize_text(alt)
        # (?<![\w.]) also stops "25" from matching inside a decimal such as "100.25".
        if norm and re.search(r"(?<![\w.])" + re.escape(norm) + r"(?!\w)", answer_norm):
            return True
    return False


def score_key_facts(facts: Optional[List[str]], answer: str) -> Optional[List[bool]]:
    if not facts:
        return None
    answer_norm = normalize_text(strip_citations(answer))
    return [fact_found(f, answer_norm) for f in facts]


def graph_info(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Decisions parsed from the trace, plus the structured fields the graph returns (absent in old results)."""
    info = parse_trace(result.get("trace"))
    if info is not None:
        for key in ("stop_reason", "refusal_source", "node_seconds"):
            info[key] = result.get(key)
    return info


def score_record(version: str, row: Dict[str, Any], result: Dict[str, Any], latency: float) -> Dict[str, Any]:
    negative = row["answer_type"] == "negative"
    pages = ordered_pages(result["retrieved_chunks"])
    gt = row["ground_truth_pages"]
    fact_hits = None if negative else score_key_facts(row.get("key_facts"), result["answer"])
    return {
        "version": version,
        "id": row["id"],
        "question": row["question"],
        "answer_type": row["answer_type"],
        "k": int(row.get("k", 6)),
        "ground_truth_pages": gt,
        "answer": result["answer"],
        "refused": is_refusal(result["answer"]),
        "retrieved_pages": pages,
        "retrieved_chunk_ids": [c["chunk_id"] for c in result["retrieved_chunks"]],
        "recall_at_k": None if negative else recall_at_k(pages, gt),
        "mrr": None if negative else reciprocal_rank(pages, gt),
        "key_fact_hits": fact_hits,
        "key_fact_recall": None if fact_hits is None else sum(fact_hits) / len(fact_hits),
        "latency_s": round(latency, 3),
        "graph": graph_info(result) if version == "graph" else None,
        "trace": result.get("trace"),
    }


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one version's per-question records. Counts are included next to every rate."""
    ans = [r for r in records if r["answer_type"] != "negative"]
    neg = [r for r in records if r["answer_type"] == "negative"]
    lat = [r["latency_s"] for r in records]
    out: Dict[str, Any] = {
        "n_answerable": len(ans),
        "n_negative": len(neg),
        "recall_at_k": _mean([r["recall_at_k"] for r in ans]),
        "mrr": _mean([r["mrr"] for r in ans]),
        "refused_negatives": sum(r["refused"] for r in neg),
        "refusal_correctness": _mean([float(r["refused"]) for r in neg]),
        "refused_answerables": sum(r["refused"] for r in ans),
        "over_refusal": _mean([float(r["refused"]) for r in ans]),
        "latency_mean_s": _mean(lat),
        "latency_median_s": statistics.median(lat) if lat else None,
        "latency_total_s": sum(lat),
    }
    with_facts = [r for r in ans if r.get("key_fact_recall") is not None]
    out["n_with_key_facts"] = len(with_facts)
    out["key_fact_recall"] = _mean([r["key_fact_recall"] for r in with_facts])
    out["answers_with_all_key_facts"] = sum(r["key_fact_recall"] == 1.0 for r in with_facts)

    graph = [r["graph"] for r in records if r.get("graph")]
    if graph:
        ans_g = [(r, r["graph"]) for r in ans if r.get("graph")]
        neg_g = [(r, r["graph"]) for r in neg if r.get("graph")]
        # "Hit" = a gold page was in the FIRST retrieval, the same context the grader's first decision saw.
        hit = [(r, g) for r, g in ans_g if set(r["ground_truth_pages"]) & set(g["first_pages"] or [])]
        out["graph"] = {
            "mean_rewrites": _mean([g["rewrites"] for g in graph]),
            "questions_with_rewrite": sum(g["rewrites"] > 0 for g in graph),
            # Grader, judged on its FIRST decision (before any rewrite).
            "grader_first_answerable_yes": sum(g["grades"][0] for _, g in ans_g if g["grades"]),
            "grader_first_answerable_no": sum(not g["grades"][0] for _, g in ans_g if g["grades"]),
            "grader_first_negative_yes": sum(g["grades"][0] for _, g in neg_g if g["grades"]),
            "grader_first_negative_no": sum(not g["grades"][0] for _, g in neg_g if g["grades"]),
            "grader_first_answerable_with_hit_yes": sum(g["grades"][0] for _, g in hit if g["grades"]),
            "grader_first_answerable_with_hit_no": sum(not g["grades"][0] for _, g in hit if g["grades"]),
            # Verifier, judged on non-refusal generator answers.
            "answerable_generator_refused": sum(bool(g["generator_refused"]) for _, g in ans_g),
            "answerable_verifier_rejected": sum(g["verify"] == "no" for _, g in ans_g),
            "negative_generator_refused": sum(bool(g["generator_refused"]) for _, g in neg_g),
            "negative_verifier_rejected": sum(g["verify"] == "no" for _, g in neg_g),
            "negative_verifier_passed": sum(g["verify"] == "yes" for _, g in neg_g),
            # Rejections by the deterministic number check (counted apart from the LLM verifier's).
            "answerable_grounding_rejected": sum(g["verify"] == "grounding" for _, g in ans_g),
            "negative_grounding_rejected": sum(g["verify"] == "grounding" for _, g in neg_g),
        }
        # Structured fields (only present in results produced after the graph started returning them).
        sources = [g.get("refusal_source") for g in graph if g.get("refusal_source") is not None]
        if sources:
            out["graph"]["refused_by_generator"] = sum(s == "generator" for s in sources)
            out["graph"]["refused_by_verifier"] = sum(s == "verifier" for s in sources)
            out["graph"]["refused_by_grounding_check"] = sum(s == "grounding_check" for s in sources)
            out["graph"]["refused_by_step_limit"] = sum(s == "step_limit" for s in sources)
        reasons = [g.get("stop_reason") for g in graph if g.get("stop_reason")]
        if reasons:
            out["graph"]["stop_reasons"] = {r: reasons.count(r) for r in sorted(set(reasons))}
        timed = [g["node_seconds"] for g in graph if g.get("node_seconds")]
        if timed:
            names = sorted({n for t in timed for n in t})
            # mean seconds per question spent in each node (0 for questions where the node did not run)
            out["graph"]["mean_node_seconds"] = {n: round(sum(t.get(n, 0.0) for t in timed) / len(timed), 2) for n in names}
    return out


def _signature(r: Dict[str, Any]) -> tuple:
    """What must match between repeats for a question's result to count as stable."""
    graph = r.get("graph") or {}
    return (r["answer"], tuple(r.get("retrieved_chunk_ids") or ()), r.get("refused"), tuple(graph.get("grades") or ()))


def add_repeat_stats(summary: Dict[str, Any], runs: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Fold several repeats of the same version into its summary.

    `summary` was computed from the first repeat (all non-latency metrics), so metrics are only
    trustworthy if repeats agree; `results_stable_across_repeats` counts the questions whose answer,
    retrieved chunks, refusal flag and graph grades were identical in every repeat. Latency fields are
    recomputed over all repeats. With a single repeat the summary is returned unchanged.
    """
    if len(runs) < 2:
        return summary
    all_lat = [r["latency_s"] for run in runs for r in run]
    summary["latency_mean_s"] = _mean(all_lat)
    summary["latency_median_s"] = statistics.median(all_lat)
    summary["latency_total_s"] = sum(all_lat)                  # all repeats, to match the mean and median
    summary["repeats"] = len(runs)
    summary["latency_mean_per_repeat_s"] = [_mean([r["latency_s"] for r in run]) for run in runs]
    by_id: Dict[str, set] = {}
    for run in runs:
        for r in run:
            by_id.setdefault(r["id"], set()).add(_signature(r))
    summary["results_stable_across_repeats"] = sum(len(v) == 1 for v in by_id.values())
    summary["n_questions"] = len(by_id)
    return summary


# ---------------- markdown tables ----------------

def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _num(x: Optional[float], digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_tables(summary: Dict[str, Dict[str, Any]]) -> str:
    versions = [v for v in VERSIONS if v in summary]
    head = "| Metric | " + " | ".join(LABELS[v] for v in versions) + " |\n|---|" + "---|" * len(versions) + "\n"

    def row(name: str, fn: Callable[[Dict[str, Any]], str]) -> str:
        return f"| {name} | " + " | ".join(fn(summary[v]) for v in versions) + " |\n"

    any_s = summary[versions[0]]
    text = head
    text += row(f"Recall@K (n={any_s['n_answerable']} answerable)", lambda s: _num(s["recall_at_k"]))
    text += row("MRR", lambda s: _num(s["mrr"]))
    text += row(f"Refusal correctness on negatives (n={any_s['n_negative']})",
                lambda s: f"{s['refused_negatives']}/{s['n_negative']} ({_pct(s['refusal_correctness'])})")
    text += row(f"Over-refusal on answerables (n={any_s['n_answerable']})",
                lambda s: f"{s['refused_answerables']}/{s['n_answerable']} ({_pct(s['over_refusal'])})")
    if any_s.get("n_with_key_facts"):
        text += row(f"Key-fact recall, mean (n={any_s['n_with_key_facts']} answerable)", lambda s: _num(s["key_fact_recall"]))
        text += row("Answers containing all key facts",
                    lambda s: f"{s['answers_with_all_key_facts']}/{s['n_with_key_facts']}")

    def latency_cell(s: Dict[str, Any]) -> str:
        cell = _num(s["latency_mean_s"], 1)
        if s.get("repeats"):
            cell += " (runs: " + ", ".join(_num(x, 1) for x in s["latency_mean_per_repeat_s"]) + ")"
        return cell

    text += row("Mean latency per question (s)", latency_cell)
    text += row("Median latency per question (s)", lambda s: _num(s["latency_median_s"], 1))
    if any_s.get("repeats"):
        text += row("Questions with identical results in every repeat (answer, chunks, refusal, grades)",
                    lambda s: f"{s['results_stable_across_repeats']}/{s['n_questions']}")

    if "graph" in summary and "graph" in summary["graph"]:
        g = summary["graph"]["graph"]
        text += (
            "\nLangGraph decisions\n\n| Decision | Count |\n|---|---|\n"
            f"| Questions that triggered at least one rewrite | {g['questions_with_rewrite']} |\n"
            f"| Mean rewrites per question | {_num(g['mean_rewrites'], 2)} |\n"
            f"| Grader, first decision, answerable questions: yes / no | {g['grader_first_answerable_yes']} / {g['grader_first_answerable_no']} |\n"
            f"| Grader, first decision, answerable questions whose gold page was retrieved: yes / no | {g['grader_first_answerable_with_hit_yes']} / {g['grader_first_answerable_with_hit_no']} |\n"
            f"| Grader, first decision, negative questions: yes / no | {g['grader_first_negative_yes']} / {g['grader_first_negative_no']} |\n"
            f"| Answerable: generator already refused / verifier rejected an answer | {g['answerable_generator_refused']} / {g['answerable_verifier_rejected']} |\n"
            f"| Negative: generator already refused / verifier rejected / verifier passed | {g['negative_generator_refused']} / {g['negative_verifier_rejected']} / {g['negative_verifier_passed']} |\n"
        )
        if "refused_by_generator" in g:
            text += (f"| Final refusals by source: generator / number check / LLM verifier / step limit | "
                     f"{g['refused_by_generator']} / {g.get('refused_by_grounding_check', 0)} / "
                     f"{g['refused_by_verifier']} / {g.get('refused_by_step_limit', 0)} |\n")
        if "stop_reasons" in g:
            text += "| Why the retrieve/grade loop stopped | " + ", ".join(f"{k}: {v}" for k, v in g["stop_reasons"].items()) + " |\n"
        if "mean_node_seconds" in g:
            text += ("\nLangGraph: mean seconds per question spent in each node\n\n| Node | Seconds |\n|---|---|\n"
                     + "".join(f"| {n} | {s:.2f} |\n" for n, s in g["mean_node_seconds"].items()))
    return text


# ---------------- rescore ----------------

def rescore(prefix: str, dataset_path: Optional[Path] = None) -> Dict[str, Any]:
    """Recompute the key-fact fields and all summaries from the saved results of an earlier run.

    No model is called: only the saved answers are re-scored against the dataset's current key_facts.
    The dataset defaults to the one recorded in the saved summary, so this works for any --out-prefix.
    """
    paths = out_paths(prefix)
    saved = json.loads(paths["summary"].read_text(encoding="utf-8"))
    dataset_path = dataset_path or Path(saved["meta"]["dataset"])
    rows = {r["id"]: r for r in load_jsonl(dataset_path)}
    records = load_jsonl(paths["results"])
    runs: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for rec in records:
        row = rows[rec["id"]]
        negative = row["answer_type"] == "negative"
        hits = None if negative else score_key_facts(row.get("key_facts"), rec["answer"])
        rec["key_fact_hits"] = hits
        rec["key_fact_recall"] = None if hits is None else sum(hits) / len(hits)
        runs.setdefault(rec["version"], {}).setdefault(rec.get("repeat", 0), []).append(rec)

    summary = {}
    for version in [v for v in VERSIONS if v in runs]:
        per_repeat = [runs[version][i] for i in sorted(runs[version])]
        summary[version] = add_repeat_stats(summarize(per_repeat[0]), per_repeat)

    meta = saved["meta"]
    meta["key_facts_rescored_from_saved_answers"] = True
    tmp = paths["results"].with_suffix(".jsonl.tmp")          # write beside, then swap in, so a crash cannot truncate it
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, paths["results"])
    write_json(paths["summary"], {"meta": meta, "summary": summary})
    tables = render_tables(summary)
    paths["tables"].write_text(tables, encoding="utf-8")
    print(tables)
    print(f"Re-scored {paths['results']} (no models were run)")
    return summary


# ---------------- run ----------------

def run(pdf_path: str, dataset_path: Path, versions: List[str], limit: Optional[int],
        repeats: int = 1, prefix: str = DEFAULT_PREFIX, k_override: Optional[int] = None) -> Dict[str, Any]:
    rows = load_jsonl(dataset_path)
    if limit:
        rows = rows[:limit]
    if k_override:
        rows = [{**r, "k": k_override} for r in rows]      # same k for every question
    runners = make_runners(pdf_path, versions)
    paths = out_paths(prefix)

    paths["results"].parent.mkdir(parents=True, exist_ok=True)
    paths["results"].write_text("", encoding="utf-8")
    runs_by_version: Dict[str, List[List[Dict[str, Any]]]] = {v: [] for v in versions}

    for repeat in range(repeats):
        tag = f" (repeat {repeat + 1}/{repeats})" if repeats > 1 else ""
        for version in versions:
            fn = runners[version]
            print(f"\n=== {LABELS[version]}{tag}: warm-up (not timed) ===")
            fn("What is this document about?", 6)      # loads models so the first question is not charged for it

            print(f"=== {LABELS[version]}{tag}: {len(rows)} questions ===")
            recs = []
            for row in rows:
                k = int(row.get("k", 6))
                start = time.perf_counter()
                result = fn(row["question"], k)
                latency = time.perf_counter() - start
                rec = score_record(version, row, result, latency)
                rec["repeat"] = repeat
                rec["dataset"] = str(dataset_path)
                recs.append(rec)
                with paths["results"].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{row['id']:>4} {latency:6.1f}s refused={rec['refused']!s:5} "
                      f"recall={rec['recall_at_k']} pages={rec['retrieved_pages']}")
            runs_by_version[version].append(recs)

    # Metrics come from the first repeat (temperature 0 makes them repeatable); latency uses all repeats.
    summary = {v: add_repeat_stats(summarize(runs[0]), runs) for v, runs in runs_by_version.items()}
    with open(pdf_path, "rb") as f:
        pdf_sha = hashlib.sha256(f.read()).hexdigest()
    meta = {
        "dataset": str(dataset_path),
        "num_questions": len(rows),
        "repeats": repeats,
        "k_override": k_override,
        "pdf_file": os.path.basename(pdf_path),
        "pdf_sha256": pdf_sha,
        "llm_model": LLM_MODEL,
        "package_versions": {p: md.version(p) for p in
                             ["langchain-core", "langchain-classic", "langchain-community", "langchain-ollama",
                              "langchain-text-splitters", "langgraph", "rank-bm25", "faiss-cpu"]},
    }
    write_json(paths["summary"], {"meta": meta, "summary": summary})
    tables = render_tables(summary)
    paths["tables"].write_text(tables, encoding="utf-8")
    print("\n" + tables)
    print(f"Wrote {paths['results']}, {paths['summary']}, {paths['tables']}")
    return summary


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.compare", description=__doc__.split("\n")[0])
    p.add_argument("--dataset", type=Path, default=None,
                   help=f"Dataset to run (default {DEFAULT_DATASET}). With --rescore the default is the dataset "
                        "recorded in the saved summary.")
    p.add_argument("--versions", nargs="+", choices=VERSIONS, default=VERSIONS)
    p.add_argument("--limit", type=positive_int, default=None, help="Only the first N questions (smoke test).")
    p.add_argument("--k", type=positive_int, default=None,
                   help="Retrieve this many chunks for every question instead of the dataset's per-row k "
                        "(the Streamlit apps use 10).")
    p.add_argument("--repeats", type=positive_int, default=1, help="Run everything N times to see latency variation.")
    p.add_argument("--rescore", action="store_true",
                   help="Do not run any model: re-score the saved results of --out-prefix against the dataset's key_facts.")
    p.add_argument("--out-prefix", default=DEFAULT_PREFIX,
                   help="Output files are <prefix>_results.jsonl, _summary.json and _tables.md.")
    args = p.parse_args(argv)

    if args.rescore:
        rescore(args.out_prefix, args.dataset)
        return 0

    dataset = args.dataset or DEFAULT_DATASET
    non_default = (args.k or args.limit or dataset != DEFAULT_DATASET or sorted(args.versions) != sorted(VERSIONS))
    if non_default and args.out_prefix == DEFAULT_PREFIX:
        sys.exit("This run differs from the default (--k, --limit, --dataset or --versions), so it would overwrite "
                 f"the main results in {DEFAULT_PREFIX}_*. Pass --out-prefix to write it elsewhere.")

    pdf_path = os.environ.get("PDF_PATH")
    if not pdf_path:
        sys.exit("PDF_PATH environment variable not set. Example:\n"
                 "  PDF_PATH=/path/to/file.pdf python -m eval.compare")
    if not os.path.isfile(pdf_path):
        sys.exit(f"PDF_PATH does not point to a file: {pdf_path}")
    try:
        check_ollama()
    except OllamaNotReady as exc:
        sys.exit(str(exc))

    run(pdf_path, dataset, args.versions, args.limit, repeats=args.repeats, prefix=args.out_prefix,
        k_override=args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
