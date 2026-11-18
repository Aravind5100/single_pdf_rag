"""Command-line entry point for the framework versions.

Run from the repository root:
    PDF_PATH=/path/to/file.pdf python -m lc_rag.cli "Are pets allowed?"
    PDF_PATH=/path/to/file.pdf python -m lc_rag.cli            # interactive
    PDF_PATH=/path/to/file.pdf python -m lc_rag.cli --mode graph "Are pets allowed?"
"""
import argparse
import os
import sys

from lc_rag.chain import LCRag
from lc_rag.graph import GraphRag
from ollama_check import OllamaNotReady, check_ollama


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lc_rag.cli", description="Ask questions about a PDF (LangChain version).")
    p.add_argument("question", nargs="?", help="Question to ask. Omit for an interactive prompt.")
    p.add_argument("--mode", choices=["chain", "graph"], default="chain",
                   help="chain: plain LangChain pipeline. graph: LangGraph with grading, rewrite and verification.")
    p.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve (default 10, same as the UI).")
    return p


def print_result(result: dict) -> None:
    print("\n".join(f"  [{step}]" for step in result["trace"]))
    print(f"\nAnswer: {result['answer']}\n")
    print("Citations:")
    for ch in result["retrieved_chunks"]:
        print(f"  Page {ch['page_number']} | {ch['chunk_id']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    pdf_path = os.environ.get("PDF_PATH")
    if not pdf_path:
        sys.exit("PDF_PATH environment variable not set. Example:\n"
                 "  PDF_PATH=/path/to/file.pdf python -m lc_rag.cli \"Your question\"")
    if not os.path.isfile(pdf_path):
        sys.exit(f"PDF_PATH does not point to a file: {pdf_path}")
    try:
        check_ollama()
    except OllamaNotReady as exc:
        sys.exit(str(exc))

    rag = LCRag(pdf_path, log=print)
    pipeline = GraphRag(rag) if args.mode == "graph" else rag

    if args.question:
        print_result(pipeline.answer(args.question, k=args.k))
        return 0

    print("Ask a question (blank line or 'exit' to quit).")
    while True:
        try:
            question = input("Q: ").strip()
        except EOFError:
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        print_result(pipeline.answer(question, k=args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
