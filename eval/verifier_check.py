"""Test the LangGraph verifier with answers whose correctness is known.

The comparison run never exercised the verifier: the generator refused every negative question
itself and the verifier said "yes" to every real answer, so it was never shown a bad answer.
This script shows it three kinds of answer for each of 12 questions, against the real retrieved context:

  * control        the model's own answer, which is supported by that context
  * contradiction  a seeded answer that contradicts the handbook (see eval/verifier_cases.jsonl)
  * fabrication    a correct answer plus an invented detail. The script checks that the invented
                   text really is absent from the retrieved context, so it is unsupported by design.

A reliable verifier says "yes" to controls and "no" to the other two.

Run from the repository root:
    PDF_PATH=/path/to/file.pdf python -m eval.verifier_check
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from eval.eval_runner import load_jsonl, write_json
from generation_strategy import format_context
from lc_rag.chain import LCRag
from lc_rag.graph import VERIFY_PROMPT, make_judge_llm, parse_yes_no
from ollama_check import OllamaNotReady, check_ollama

DATASET = Path("eval/eval_dataset_extended.jsonl")
CASES = Path("eval/verifier_cases.jsonl")
OUT = Path("eval/verifier_check_results.json")
KINDS = ["control", "contradiction", "fabrication"]


def summarize_verdicts(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per kind: how many answers the verifier accepted ('yes'). Controls should be accepted, the rest rejected."""
    out: Dict[str, Any] = {}
    for kind in KINDS:
        rows = [r for r in results if r["kind"] == kind]
        accepted = sum(r["verifier_said_yes"] for r in rows)
        out[kind] = {"n": len(rows), "accepted": accepted, "rejected": len(rows) - accepted}
    return out


def render(summary: Dict[str, Any]) -> str:
    c, x, f = summary["control"], summary["contradiction"], summary["fabrication"]
    return (
        "| Answer shown to the verifier | n | Accepted (yes) | Rejected (no) | Correct behavior |\n"
        "|---|---|---|---|---|\n"
        f"| Control (supported answer) | {c['n']} | {c['accepted']} | {c['rejected']} | accept |\n"
        f"| Seeded contradiction | {x['n']} | {x['accepted']} | {x['rejected']} | reject |\n"
        f"| Correct answer plus an invented detail | {f['n']} | {f['accepted']} | {f['rejected']} | reject |\n"
    )


def run(pdf_path: str) -> Dict[str, Any]:
    questions = {r["id"]: r for r in load_jsonl(DATASET)}
    cases = load_jsonl(CASES)
    rag = LCRag(pdf_path, log=print)
    judge = make_judge_llm()

    results: List[Dict[str, Any]] = []
    for qid in dict.fromkeys(c["id"] for c in cases):          # one retrieval per question, in file order
        row = questions[qid]
        docs = rag.retrieve(row["question"], int(row.get("k", 6)))
        context = format_context([{"page_number": d.metadata["page_number"], "text": d.page_content} for d in docs])

        candidates = [{"kind": "control", "answer": rag.generate(docs, row["question"])}]
        for c in cases:
            if c["id"] == qid:
                if c.get("invented") and c["invented"].lower() in context.lower():
                    sys.exit(f"{qid}: the 'invented' text {c['invented']!r} appears in the retrieved context, "
                             "so that answer is not unsupported. Pick a different invented detail.")
                candidates.append({"kind": c["kind"], "answer": c["answer"]})

        for cand in candidates:
            raw = judge.invoke(VERIFY_PROMPT.format(question=row["question"], context=context,
                                                    answer=cand["answer"])).content.strip()
            res = {"id": qid, "kind": cand["kind"], "answer": cand["answer"], "verifier_raw": raw,
                   "verifier_said_yes": parse_yes_no(raw)}
            results.append(res)
            print(f"{qid:>4} {cand['kind']:13} verifier={'yes' if res['verifier_said_yes'] else 'no ':3} {cand['answer'][:70]!r}")

    summary = summarize_verdicts(results)
    write_json(OUT, {"summary": summary, "results": results})
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
