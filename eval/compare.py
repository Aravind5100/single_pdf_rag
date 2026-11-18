"""Compare the hand-written baseline, the LangChain chain and the LangGraph pipeline.

Run from the repository root:
    PDF_PATH=/path/to/file.pdf python -m eval.compare
    PDF_PATH=... python -m eval.compare --versions baseline chain --limit 5
    PDF_PATH=... python -m eval.compare --repeats 2          # repeat the run to see latency variation
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
    facts found in the answer (case-insensitive substring match; "a|b" means either is accepted).
    This is a cheap objective check for facts the answer must contain. It cannot catch a wrong claim
    that sits next to the right facts, so it is not a full correctness judgment (see eval/COMPARISON.md).
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
            else:
                info["verify"] = "yes" if "supported=yes" in line else "no"
    return info


def normalize_text(text: str) -> str:
    """Lowercase, straighten apostrophes, drop commas, collapse whitespace."""
    return " ".join(text.lower().replace("\u2019", "'").replace(",", " ").split())


def fact_found(fact: str, answer_norm: str) -> bool:
    """A fact is found if any of its '|'-separated alternatives is a substring of the normalized answer."""
    return any(alt.strip() and normalize_text(alt) in answer_norm for alt in fact.split("|"))


def score_key_facts(facts: Optional[List[str]], answer: str) -> Optional[List[bool]]:
    if not facts:
        return None
    answer_norm = normalize_text(answer)
    return [fact_found(f, answer_norm) for f in facts]


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
        "graph": parse_trace(result.get("trace")) if version == "graph" else None,
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
        }
    return out


def add_repeat_stats(summary: Dict[str, Any], runs: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Fold several repeats of the same version into its summary (latency over all repeats, answer stability).

    `summary` was computed from the first repeat. With a single repeat it is returned unchanged.
    """
    if len(runs) < 2:
        return summary
    all_lat = [r["latency_s"] for run in runs for r in run]
    summary["latency_mean_s"] = _mean(all_lat)
    summary["latency_median_s"] = statistics.median(all_lat)
    summary["repeats"] = len(runs)
    summary["latency_mean_per_repeat_s"] = [_mean([r["latency_s"] for r in run]) for run in runs]
    by_id: Dict[str, set] = {}
    for run in runs:
        for r in run:
            by_id.setdefault(r["id"], set()).add(r["answer"])
    summary["answers_stable_across_repeats"] = sum(len(v) == 1 for v in by_id.values())
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
        text += row("Questions with an identical answer in every repeat",
                    lambda s: f"{s['answers_stable_across_repeats']}/{s['n_questions']}")

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
    return text


# ---------------- run ----------------

def run(pdf_path: str, dataset_path: Path, versions: List[str], limit: Optional[int],
        repeats: int = 1, prefix: str = DEFAULT_PREFIX) -> Dict[str, Any]:
    rows = load_jsonl(dataset_path)
    if limit:
        rows = rows[:limit]
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.compare", description=__doc__.split("\n")[0])
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--versions", nargs="+", choices=VERSIONS, default=VERSIONS)
    p.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke test).")
    p.add_argument("--repeats", type=int, default=1, help="Run everything N times to see latency variation.")
    p.add_argument("--out-prefix", default=DEFAULT_PREFIX,
                   help="Output files are <prefix>_results.jsonl, _summary.json and _tables.md.")
    args = p.parse_args(argv)

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

    run(pdf_path, args.dataset, args.versions, args.limit, repeats=args.repeats, prefix=args.out_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
