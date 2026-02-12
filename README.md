# 📘 Single-PDF RAG System (Evaluated & Persistent)

A **production-style Retrieval-Augmented Generation (RAG) system** that enables users to ask questions over a single PDF document with **measurable retrieval quality**, **persistent caching**, and an interactive **Streamlit UI**.

This repository is intentionally engineered beyond a demo: retrieval is evaluated, artifacts are cached, and system behavior is explainable and reproducible.

---

## ✨ Highlights

- 📄 **Single-PDF ingestion** (handbooks, policies, manuals)
- ✂️ **Page-aware chunking with overlap**
- 🧠 **Local embeddings** via Ollama (`nomic-embed-text`)
- 🔍 **Hybrid retrieval** (FAISS vector search + keyword fallback)
- 📊 **Formal evaluation** using Recall@K and MRR
- 💾 **Persistent cache** (embeddings + FAISS index)
- 🖥️ **Streamlit UI** with citations and chat history
- 🛑 **Hallucination-safe generation** (context-grounded answers only)

---

## 🧠 Why this project

Most RAG examples:
- skip evaluation
- rebuild embeddings on every run
- rely on opaque APIs
- cannot explain retrieval failures

This project focuses on:
- **measurable retrieval quality**
- **repeatability**
- **debuggability**
- **real-world system design**

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[PDF Document] --> B[Text Extraction<br/>(page-aware)]
    B --> C[Chunking<br/>(overlap + metadata)]
    C --> D[Local Embeddings<br/>(Ollama)]
    D --> E[FAISS Vector Index]
    E --> F[Hybrid Retrieval<br/>(vector + keyword)]
    F --> G[Answer Generation<br/>(grounded)]
    G --> H[Streamlit UI]

    F --> I[Evaluation]
    I -->|Recall@K / MRR| J[Evaluation Reports]

.
├── extract_pdf_text.py      # PDF → page-level text
├── chunk_strategy.py        # Chunking + normalization
├── embedding_strategy.py   # Local embeddings via Ollama
├── index_strategy.py       # FAISS index + hybrid retrieval
├── generation_strategy.py  # Grounded answer generation
├── persistence.py          # Embedding + FAISS caching
├── eval_runner.py          # Evaluation pipeline
├── main.py                 # Streamlit app entry point
│
├── eval/
│   ├── eval_dataset.jsonl  # Ground-truth evaluation set
│   ├── eval_results.jsonl  # Per-question results
│   └── eval_summary.json   # Aggregate metrics
│
├── cache/                  # Persistent embeddings + FAISS
└── docs/                   # PDFs used by the system

📊 Evaluation Methodology
Evaluation is retrieval-first, which is critical for RAG systems.
Implemented metrics
Recall@K
Determines whether at least one correct source page is retrieved in the top-K results.
MRR (Mean Reciprocal Rank)
Measures how highly the first correct page is ranked.
Ground truth is defined manually in eval/eval_dataset.jsonl using:
fixed questions
known correct page numbers
negative (refusal) cases
This ensures evaluation is:
objective
reproducible
actionable for debugging
💾 Persistence Design
To avoid re-embedding on every run, the system caches:
embedded chunks (JSONL)
FAISS index (binary)
Cache keys are derived from:
PDF content hash
chunking parameters
embedding model name
This guarantees correctness while providing a significant speed-up on subsequent runs.
🖥️ Streamlit UI
The Streamlit interface provides:
interactive Q&A
answer citations (page + chunk)
expandable context inspection
session-based chat history
Run the app with:
streamlit run main.py
⚙️ Setup & Usage
1️⃣ Start Ollama
ollama serve
ollama pull nomic-embed-text
2️⃣ Add a PDF
Place your PDF inside:
docs/
Update pdf_path in main.py to point to the document.
3️⃣ Run the app
streamlit run main.py
First run builds the cache
Subsequent runs load instantly
🧪 Run Evaluation
python eval_runner.py
Outputs:
eval/eval_results.jsonl
eval/eval_summary.json
🔮 Future Work
Multi-PDF ingestion
Query rewriting and reranking
Automated refusal correctness scoring
Dockerization
Extended evaluation dashboards
📜 License
MIT License

---

## ✅ That’s it

You now have:
- a **GitHub-ready README**
- a **system diagram that renders on GitHub**
- **zero downloads required**
- **single copy–paste**

If you want next:
- GitHub repo description + tags
- Screenshots checklist
- Interview explanation (2-minute version)

Just tell me.
