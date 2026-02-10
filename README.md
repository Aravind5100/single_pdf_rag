# Single PDF RAG (From Scratch)

A minimal Retrieval-Augmented Generation (RAG) pipeline built from scratch to answer questions from a single PDF.

## Pipeline
1. Extract PDF text (page-level)
2. Chunking with overlap
3. Embed chunks (local embeddings via Ollama)
4. Vector search with FAISS
5. Answer generation with citations + refusal logic

## How to run
- Start Ollama
- Place your PDF locally (not committed)
- Run:
  python main.py

## Notes
- Evaluation will be added in a future iteration.
- Cache/index files are ignored via .gitignore.
