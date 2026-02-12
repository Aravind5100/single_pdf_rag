import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
from extract_pdf_text import extract_text_from_pdf
from chunk_strategy import chunk_pages
from embedding_strategy import embed_chunks
from index_strategy import build_faiss_index, hybrid_retrieve_indices
from generation_strategy import generate_answer


EVAL_PATH = Path("eval/eval_dataset.jsonl")
OUT_RESULTS = Path("eval/eval_results.jsonl")
OUT_SUMMARY = Path("eval/eval_summary.json")

#Loads JSONL data from a file and returns it as a list of dictionaries. Each line in the file should be a valid JSON object.
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

#Writes a list of dictionaries to a file in JSONL format. This File will contain the detailed results of each evaluation case, including the question, retrieved pages, and computed metrics. Each dictionary in the list is written as a separate line in the output file, formatted as JSON.
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

#Writes a dictionary to a file in JSON format. This File will contain the summary of the evaluation results, including average metrics and other relevant information. The output is formatted with indentation for readability.
def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------- METRICS ----------------
def recall_at_k(retrieved_pages: List[int], gt_pages: List[int]) -> float:
    """1 if ANY ground-truth page appears in retrieved_pages, else 0.
    For negative questions (gt_pages == []), recall isn't meaningful; we return None later.
    """
    if not gt_pages:
        return float("nan")
    return 1.0 if any(p in set(retrieved_pages) for p in gt_pages) else 0.0


def reciprocal_rank(retrieved_pages: List[int], gt_pages: List[int]) -> float:
    """1/rank of first occurrence of any GT page; 0 if none.
    Rank is 1-based.
    """
    if not gt_pages:
        return float("nan")
    gt: Set[int] = set(gt_pages)
    for i, p in enumerate(retrieved_pages, start=1):
        if p in gt:
            return 1.0 / i
    return 0.0


# ------------- YOUR PIPELINE HOOKS -------------

def build_rag_artifacts(pdf_path: str):
    """Builds chunks, embeddings, and FAISS index once (offline step)."""
    pages = extract_text_from_pdf(pdf_path)         
    chunks = chunk_pages(pages)                     
    embedded_chunks = embed_chunks(chunks)          
    index, vectors = build_faiss_index(embedded_chunks)

    # We return everything we need to retrieve chunks later
    return embedded_chunks, index


def retrieve_top_k(
    question: str,
    embedded_chunks: List[Dict[str, Any]],
    index,
    k: int
) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
    """
    Runs YOUR retrieval and returns:
      - retrieved_pages (ordered, with duplicates removed in order)
      - retrieved_chunk_ids
      - retrieved_chunks (full chunk dicts)
    """

    # Your hybrid retrieval returns indices into embedded_chunks
    indices = hybrid_retrieve_indices(index, embedded_chunks, question, top_k=k)

    retrieved_chunks = [embedded_chunks[i] for i in indices]
    retrieved_chunk_ids = [c["chunk_id"] for c in retrieved_chunks]

    # pages in retrieval order, dedup while preserving order
    seen = set()
    retrieved_pages = []
    for c in retrieved_chunks:
        p = int(c["page_number"])
        if p not in seen:
            seen.add(p)
            retrieved_pages.append(p)

    return retrieved_pages, retrieved_chunk_ids, retrieved_chunks


# ---------------- MAIN EVAL LOOP ----------------

def run_eval(pdf_path: str) -> None:
    dataset = load_jsonl(EVAL_PATH)

    print("Building RAG artifacts once")
    pdf_path = "/Users/aravindkompalli/Downloads/NEWHANDBOOK_AUG8_2025_Final_optimized.pdf"
    embedded_chunks, index = build_rag_artifacts(pdf_path)

    results = []
    valid_recall = []
    valid_rr = []

    for row in dataset:
        qid = row["id"]
        question = row["question"]
        answer_type = row["answer_type"]
        gt_pages = row["ground_truth_pages"]
        k = int(row.get("k", 6))

        retrieved_pages, retrieved_chunk_ids, _ = retrieve_top_k(
            question=question,
            embedded_chunks=embedded_chunks,
            index=index,
            k=k
        )

        # Metrics: for negative questions, retrieval metrics are not the point
        if answer_type == "negative":
            rec = float("nan")
            rr = float("nan")
        else:
            rec = recall_at_k(retrieved_pages, gt_pages)
            rr = reciprocal_rank(retrieved_pages, gt_pages)
            valid_recall.append(rec)
            valid_rr.append(rr)

        result = {
            "id": qid,
            "question": question,
            "answer_type": answer_type,
            "k": k,
            "ground_truth_pages": gt_pages,
            "retrieved_pages": retrieved_pages,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "metrics": {
                "recall_at_k": None if rec != rec else rec,  # NaN check
                "mrr": None if rr != rr else rr,
            },
        }
        results.append(result)

        print(f"{qid}: Recall@{k}={result['metrics']['recall_at_k']}  MRR={result['metrics']['mrr']}  retrieved_pages={retrieved_pages}")

    # Aggregate metrics
    avg_recall = sum(valid_recall) / len(valid_recall) if valid_recall else None
    mrr = sum(valid_rr) / len(valid_rr) if valid_rr else None

    summary = {
        "pdf_path": pdf_path,
        "num_cases": len(dataset),
        "num_scored_cases": len(valid_recall),
        "avg_recall_at_k": avg_recall,
        "mrr": mrr,
    }

    write_jsonl(OUT_RESULTS, results)
    write_json(OUT_SUMMARY, summary)

    print("\n✅ Wrote:")
    print(f" - {OUT_RESULTS}")
    print(f" - {OUT_SUMMARY}")
    print("\nSummary:", summary)


if __name__ == "__main__":
    # Put your handbook PDF path here:
    # Example: "docs/NEWHANDBOOK_AUG8_2025_Final_optimized.pdf"
    PDF_PATH = "docs/NEWHANDBOOK_AUG8_2025_Final_optimized.pdf"
    run_eval(PDF_PATH)
