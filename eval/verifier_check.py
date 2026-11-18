"""Test the LangGraph verifier with answers whose correctness is known.

The comparison run never exercised the verifier: the generator refused every negative question
itself and the verifier said "yes" to every real answer, so it was never shown a bad answer.
This script shows it three kinds of answer for each of 12 questions, against the real retrieved context:

  * control        the model's own answer, which is supported by that context
  * contradiction  a seeded answer that contradicts the handbook (see eval/verifier_cases.jsonl)
  * fabrication    the control answer with ONE invented sentence appended (eval/verifier_cases.jsonl). Because
                   everything else is the supported control answer, a "no" can only be a reaction to the
                   invented detail. The script checks that the invented text really is absent from the
                   retrieved context, and skips (and reports) any case where it is not.

A reliable verifier says "yes" to controls and "no" to the other two. Each answer is also run through the
deterministic number check (lc_rag/grounding.py), so the table shows the LLM verifier alone, the number
check alone, and the two together. Caution: most of the invented details here are numbers, which is what
the number check can see, so its result on this set overstates how it would do on invented facts in general.

Run from the repository root:
    PDF_PATH=/path/to/file.pdf python -m eval.verifier_check
"""
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from eval.eval_runner import load_jsonl, write_json
from generation_strategy import format_context
from lc_rag.chain import LCRag
from lc_rag.grounding import ungrounded_numbers
from lc_rag.graph import VERIFY_PROMPT, make_judge_llm, parse_yes_no
from ollama_check import OllamaNotReady, check_ollama

DATASET = Path("eval/eval_dataset_extended.jsonl")
CASES = Path("eval/verifier_cases.jsonl")
OUT = Path("eval/verifier_check_results.json")
KINDS = ["control", "contradiction", "fabrication"]


def is_clean_reply(raw: str) -> bool:
    """True if the verifier replied with just 'yes' or 'no' (any case, trailing punctuation allowed)."""
    return re.fullmatch(r"\s*(yes|no)\W*", raw, flags=re.I) is not None


def build_candidates(control: str, cases: List[Dict[str, Any]], context: str):
    """Answers to show the verifier for one question: the control, then each seeded case.

    Contradictions are used as written. A fabrication is the control answer plus the case's `append`
    sentence. Returns (candidates, skipped); a case whose invented text already appears in the retrieved
    context is skipped, because that answer would not actually be unsupported.
    """
    candidates = [{"kind": "control", "answer": control}]
    skipped = []
    for c in cases:
        if c.get("invented") and c["invented"].lower() in context.lower():
            skipped.append({"id": c["id"], "kind": c["kind"],
                            "reason": f"invented text {c['invented']!r} appears in the retrieved context"})
            continue
        answer = f"{control} {c['append']}" if c["kind"] == "fabrication" else c["answer"]
        candidates.append({"kind": c["kind"], "answer": answer})
    return candidates, skipped


def summarize_verdicts(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per kind: how many answers the verifier accepted ('yes'). Controls should be accepted, the rest rejected."""
    out: Dict[str, Any] = {}
    for kind in KINDS:
        rows = [r for r in results if r["kind"] == kind]
        accepted = sum(r["verifier_said_yes"] for r in rows)
        flagged = sum(bool(r.get("grounding_issues")) for r in rows)
        either = sum((not r["verifier_said_yes"]) or bool(r.get("grounding_issues")) for r in rows)
        out[kind] = {"n": len(rows), "accepted": accepted, "rejected": len(rows) - accepted,
                     "flagged_by_number_check": flagged, "rejected_by_either": either}
    out["unclean_replies"] = sum(not is_clean_reply(r.get("verifier_raw", "yes")) for r in results)
    return out


def render(summary: Dict[str, Any]) -> str:
    rows = [("Control (supported answer)", "control", "accept"),
            ("Seeded contradiction", "contradiction", "reject"),
            ("Control answer plus one invented sentence", "fabrication", "reject")]
    text = ("| Answer shown | n | LLM verifier: rejected | Number check: flagged | Either: rejected | Correct behavior |\n"
            "|---|---|---|---|---|---|\n")
    for label, kind, want in rows:
        s = summary[kind]
        text += (f"| {label} | {s['n']} | {s['rejected']} | {s['flagged_by_number_check']} | "
                 f"{s['rejected_by_either']} | {want} |\n")
    text += (f"\nReplies that were not a clean 'yes' or 'no' (treated as 'no' by the strict parser): "
             f"{summary['unclean_replies']}\n")
    return text


def run(pdf_path: str) -> Dict[str, Any]:
    questions = {r["id"]: r for r in load_jsonl(DATASET)}
    cases = load_jsonl(CASES)
    rag = LCRag(pdf_path, log=print)
    judge = make_judge_llm()

    results: List[Dict[str, Any]] = []
    skipped_all: List[Dict[str, Any]] = []
    for qid in dict.fromkeys(c["id"] for c in cases):          # one retrieval per question, in file order
        row = questions[qid]
        docs = rag.retrieve(row["question"], int(row.get("k", 6)))
        context = format_context([{"page_number": d.metadata["page_number"], "text": d.page_content} for d in docs])
        control = rag.generate(docs, row["question"])
        candidates, skipped = build_candidates(control, [c for c in cases if c["id"] == qid], context)
        skipped_all += skipped
        for s in skipped:
            print(f"{qid:>4} SKIPPED {s['kind']}: {s['reason']}")

        for cand in candidates:
            raw = judge.invoke(VERIFY_PROMPT.format(question=row["question"], context=context,
                                                    answer=cand["answer"])).content.strip()
            res = {"id": qid, "kind": cand["kind"], "answer": cand["answer"], "verifier_raw": raw,
                   "verifier_said_yes": parse_yes_no(raw),
                   "grounding_issues": ungrounded_numbers(cand["answer"], context)}
            results.append(res)
            print(f"{qid:>4} {cand['kind']:13} verifier={'yes' if res['verifier_said_yes'] else 'no ':3} "
                  f"numbers={res['grounding_issues'] or '-'} {cand['answer'][:50]!r}")

    summary = summarize_verdicts(results)
    write_json(OUT, {"summary": summary, "skipped_cases": skipped_all, "results": results})
    print("\n" + render(summary))
    print(f"Wrote {OUT}")
    return summary


def main() -> int:
    pdf_path = os.environ.get("PDF_PATH")
    if not pdf_path:
        sys.exit("PDF_PATH environment variable not set. Example:\n"
                 "  PDF_PATH=/path/to/file.pdf python -m eval.verifier_check")
    if not os.path.isfile(pdf_path):
        sys.exit(f"PDF_PATH does not point to a file: {pdf_path}")
    try:
        check_ollama()
    except OllamaNotReady as exc:
        sys.exit(str(exc))
    run(pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
