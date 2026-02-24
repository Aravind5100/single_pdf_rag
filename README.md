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
    A["PDF Document"] --> B["Text Extraction<br/>(page-aware)"]
    B --> C["Chunking<br/>(overlap + metadata)"]
    C --> D["Local Embeddings<br/>(Ollama)"]
    D --> E["FAISS Vector Index"]
    E --> F["Hybrid Retrieval<br/>(vector + keyword)"]
    F --> G["Answer Generation<br/>(grounded)"]
    G --> H["Streamlit UI"]
    F --> I["Evaluation"]
    I -->|Recall@K / MRR| J["Evaluation Reports"]
```

### Project Structure

```
.
├── extract_pdf_text.py      # PDF → page-level text
├── chunk_strategy.py        # Chunking + normalization
├── embedding_strategy.py    # Local embeddings via Ollama
├── index_strategy.py        # FAISS index + hybrid retrieval
├── generation_strategy.py   # Grounded answer generation
├── persistence.py           # Embedding + FAISS caching
├── main.py                  # Streamlit app entry point
│
├── eval/
│   ├── eval_dataset.jsonl   # Ground-truth evaluation set
│   ├── eval_results.jsonl   # Per-question results
│   └── eval_summary.json    # Aggregate metrics
│
├── cache/                   # Persistent embeddings + FAISS
└── requirements.txt
```

---

## 📊 Evaluation Methodology

Evaluation is retrieval-first, which is critical for RAG systems.

**Implemented metrics:**

- **Recall@K** — Determines whether at least one correct source page is retrieved in the top-K results.
- **MRR (Mean Reciprocal Rank)** — Measures how highly the first correct page is ranked.

Ground truth is defined manually in `eval/eval_dataset.jsonl` using:
- Fixed questions
- Known correct page numbers
- Negative (refusal) cases

This ensures evaluation is objective, reproducible, and actionable for debugging.

---

## 💾 Persistence Design

To avoid re-embedding on every run, the system caches:
- Embedded chunks (JSONL)
- FAISS index (binary)

Cache keys are derived from:
- PDF content hash
- Chunking parameters
- Embedding model name

This guarantees correctness while providing a significant speed-up on subsequent runs.

---

## 🖥️ Streamlit UI

The Streamlit interface provides:
- Interactive Q&A
- Answer citations (page + chunk)
- Expandable context inspection
- Session-based chat history

Run the app with:

```bash
streamlit run main.py
```

---

## ⚙️ Setup & Usage

### 1. Start Ollama

```bash
ollama serve
ollama pull nomic-embed-text
```

### 2. Add a PDF

Set the `PDF_PATH` environment variable, or update the default path in `main.py`.

### 3. Run the app

```bash
streamlit run main.py
```

- First run builds the cache.
- Subsequent runs load instantly.

---

## 🧪 Run Evaluation

```bash
python eval_runner.py
```

Outputs:
- `eval/eval_results.jsonl`
- `eval/eval_summary.json`

---

## 🔮 Future Work

- Multi-PDF ingestion
- Query rewriting and reranking
- Automated refusal correctness scoring
- Dockerization
- Extended evaluation dashboards

---

## 📜 License

MIT License
