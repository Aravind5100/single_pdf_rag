# Comparison: hand-written vs LangChain vs LangGraph

All numbers below come from one real run of `python -m eval.compare` on 2026-11-18. The raw
per-question results are in `eval/compare_results.jsonl` and the aggregates in
`eval/compare_summary.json`. Nothing here was tuned to improve a score.

## Setup

| | |
|---|---|
| Document | `NEWHANDBOOK_AUG8_2025_Final_optimized.pdf` (35 pages, sha256 `476a0394...eca325`) |
| Questions | 26 in `eval/eval_dataset_extended.jsonl`: 16 answerable and 10 negative (unanswerable) |
| Answerable questions | the original Q1 to Q9, plus 7 new ones (Q11 to Q17) |
| Negative questions | the original Q10, plus 9 new ones. Each negative was checked by keyword search of the full PDF text to confirm the answer is absent. Kinds: 3 "hard" (topic present, answer absent), 5 adjacent, 1 off-domain, plus the original parking question (Q10) |
| Models | `llama3.1:8b` at temperature 0 for generation, grading and verification; `nomic-embed-text` for embeddings, both through local Ollama |
| Retrieval depth | per question `k` from the dataset (6 for all new questions) |
| Packages | langchain-core 1.8.3, langchain-classic 1.0.8, langchain-community 0.4.2, langchain-ollama 1.1.0, langchain-text-splitters 1.1.3, langgraph 1.2.22, rank-bm25 0.2.2, faiss-cpu 1.13.2 |

Cross-check: the hand-written pipeline's MRR on the original 9 answerable questions in this run is
0.8056, the same as the recorded `eval/baseline_summary.json`.

## Results

| Metric | Hand-written | LangChain | LangGraph |
|---|---|---|---|
| Recall@K (n=16 answerable) | 1.000 | 1.000 | 1.000 |
| MRR | 0.859 | 0.969 | 0.969 |
| Refusal correctness on negatives (n=10) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Over-refusal on answerables (n=16) | 0/16 (0%) | 0/16 (0%) | 0/16 (0%) |
| Mean latency per question (s) | 8.6 | 9.4 | 23.3 |
| Median latency per question (s) | 7.7 | 8.8 | 21.8 |

LangGraph decisions across the 26 questions:

| Decision | Count |
|---|---|
| Questions that triggered at least one rewrite | 8 (all 8 were negative questions) |
| Mean rewrites per question | 0.58 |
| Grader, first decision, answerable questions: yes / no | 16 / 0 |
| Grader, first decision, negative questions: yes / no | 2 / 8 |
| Answerable: generator already refused / verifier rejected an answer | 0 / 0 |
| Negative: generator already refused / verifier rejected / verifier passed | 10 / 0 / 0 |

How the metrics are defined:

- **Recall@K** is 1 if any gold page is among the retrieved pages. **MRR** is 1 over the rank of the
  first gold page in the page-deduplicated retrieved list. Both are page level and use the chunks the
  pipeline actually returned in that run (for LangGraph, the final retrieval after any rewrite).
- **Refused** means the answer contains the fixed refusal sentence (apostrophe-insensitive).
  Refusal correctness is the refused share of negatives. Over-refusal is the refused share of answerables.
- **Latency** is wall-clock time for the whole pipeline call (retrieve, then generate, plus grade,
  rewrite and verify for LangGraph) after an untimed warm-up. It was measured once per question.

## Analysis

### Retrieval: the framework versions ranked slightly better, on 3 questions

Recall@K is 1.0 for all three, so none of them missed a gold page. MRR is 0.859 for the hand-written
version and 0.969 for both framework versions. The two framework versions return the same pages as
each other on every answerable question, so they tie.

The whole gap comes from 3 of the 16 questions (Q1, Q2, Q16), where the hand-written version ranked
the gold page 2nd, 4th and 2nd, and LangChain ranked it 1st. On every other answerable question the
MRR is identical, and the framework versions were never worse. The cause is visible in the hand-written
keyword stage. It scores chunks by raw substring counts and puts those hits first. For Q1
("Are pets allowed in the residence halls?") its top keyword hits are pages 22, 19 and 21, because common
words like "are" and "allowed" match everywhere, so page 22 goes first (I checked this on the cached
chunks). BM25 weights rare terms like "pets" more heavily and the ensemble favors chunks that both
retrievers return, which is consistent with the better ranking, but I did not isolate the cause on the
LangChain side.

Caveats: with 16 questions, one question moves MRR by about 0.03 to 0.05, and 3 questions is a small
basis for a claim. I would call this "not worse, possibly a bit better on ranking", not a proven win.
It is also not a like-for-like swap of one component (see "What the LangChain version does differently").

### Refusal: a tie at the ceiling, which this dataset cannot break

All three versions refused all 10 negatives and refused none of the 16 answerable questions. Every
negative answer was exactly the 81-character refusal sentence in every version, so no hedged or partial
refusal slipped past the metric in either direction.

That means the prompt plus the 8B model already refuse correctly here, and the graph's extra safety
nets had nothing to fix. Sample size limits matter: 0 failures in 10 negatives is consistent with a true
failure rate of up to about 26% (exact one-sided 95% bound), and 0 over-refusals in 16 is consistent with
up to about 17%. My negatives are mostly easy or adjacent. I did not include adversarial negatives where
a model is tempted to guess, so this result says little about harder cases.

### Latency: the graph costs about 2.5 times the chain

The hand-written version averaged 8.6 s and LangChain 9.4 s per question. The 0.8 s gap is a single
run and I am not claiming it is real. LangGraph averaged 23.3 s, which is 2.5 times LangChain, about
13.8 s more per question.

The cost is not mainly the rewrites. The 18 questions with no rewrite averaged 23.4 s and the 8 with
rewrites averaged 23.0 s. Every question pays for a grading call over the full context and a
verification call, and answerable questions also have longer generations than the one-sentence refusals.
Latency was measured once, so treat it as indicative.

### Where the graph helped, and where it hurt

**Helped: nothing measurable.**
- On all 16 answerable questions the graph's answer text was identical to the plain chain's.
- The rewrite loop ran only on negative questions, and it never changed an outcome. In 7 of those 8
  cases the rewrite changed which pages were retrieved, but the generator still refused every time.
- The verifier never rejected anything, so it never prevented a bad answer.

**Hurt: latency, and some wasted work.**
- 2.5 times the latency of the chain for the same answers.
- In all 7 cases where the graph rewrote twice, the second rewrite was identical to the first, so the
  second retry could not possibly find anything new. The loop has no check for "the rewrite did not
  change". I did not fix this, because the comparison should reflect the graph as specified, but stopping
  when a rewrite repeats would remove this waste and is the obvious first improvement.

**Visible, which is valuable for learning:** the per-node trace shows exactly why each question took the
path it did (see `LEARNING_NOTES.md`).

### Is the 8B local model a reliable grader?

Only partly, and the evidence is thin.

- **Context grader.** On its first decision it said "yes" for all 16 answerable questions, and in all 16
  the gold page really was in the first retrieval (no false negatives). On the 10 negatives it said "no" 8
  times and "yes" 2 times, so it was right on 24 of 26 (92%). Both errors were "hard" negatives where the
  topic is in the handbook but the specific answer is not (room rate, smoking fine amount). A third hard
  negative (meal plan price) flipped from "no" to "yes" after a rewrite. So it handles clearly
  off-topic context well and is shaky exactly where grading is hardest. No harm followed, because the
  generator refused anyway.
- **Verifier.** It was barely exercised. On the 10 negatives the generator refused on its own every time,
  so the verifier step was skipped by design (a refusal makes no claim to check). On the 16 answerable
  questions it said "yes" every time. So there is no evidence that it can catch an unsupported answer.
  There is a concrete reason for doubt: for Q15 (can an of-age guest keep alcohol in an under-21
  resident's room?) the answer in all three versions says no and omits the handbook's exception "unless
  their of-age roommate is present". The verifier passed that incomplete answer. I read the other answers
  only in truncated form, and they looked consistent with the gold answers, but I did not score
  correctness.

To test the verifier properly you would need answers with seeded errors. I did not do that.

## What the LangChain version does differently

I could not reproduce the hand-written behavior exactly, so the two are not a controlled swap of one
component. The differences:

| Area | Hand-written | LangChain version |
|---|---|---|
| Hybrid merge | Keyword hits first, then vector hits, with a quota of `max(2, k // 2)` keyword slots | `EnsembleRetriever` fuses BM25 and FAISS with weighted reciprocal rank fusion (0.5 / 0.5). LangChain has no built-in keyword-first merge, so this cannot be reproduced exactly. Each retriever returns k, and the fused list is trimmed to k |
| Keyword scoring | Raw substring counts | BM25. I reused the baseline's tokenizer and stopwords so the terms match |
| Zero-score keyword hits | Dropped | Stock `BM25Retriever` returns arbitrary documents when nothing matches, so I subclassed it to drop zero scores |
| Chunking | Fixed 900/350 sliding window, 204 chunks | `RecursiveCharacterTextSplitter` at 900/350 on word boundaries, 184 chunks. Chunk ids follow the same `page_chunk_n` pattern but the boundaries differ |
| Page text | PyMuPDF, normalized | `PyMuPDFLoader`, then the baseline's `normalize_for_search`, so the indexed text is the same |
| Embeddings call | `/api/embeddings`, one request per chunk | `OllamaEmbeddings` (`/api/embed`, batched). Vectors are L2-normalized in both |
| Generation call | `/api/generate` with one text prompt | `ChatOllama` (`/api/chat`) with the same text as a single human message. The prompt is byte-identical (a test pins this) |
| Answer text | | Identical to the baseline on only 5 of 16 answerable questions, because the retrieved context and the call path differ. The content is similar |

Other things to know:

- `EnsembleRetriever` lives in `langchain-classic` in LangChain 1.x, and `langchain-community`
  prints a deprecation notice saying it is being sunset. The pins are in `requirements-lc.txt`.
- The LangGraph version reuses the LangChain pipeline for retrieval and generation, so it inherits all
  of the above.

## Limits of this comparison

- **Small samples.** 16 answerable and 10 negative questions, one document, one model, one run. Small
  differences are not statistically meaningful.
- **Answer correctness is not scored.** I read the answers (truncated) as a sanity check and found at
  least one incomplete answer (Q15). No automatic correctness metric exists in this repo.
- **Latency was measured once per question.** The baseline and LangChain share an identical prompt, so
  Ollama's prompt caching could favor whichever ran later. I ran versions in separate blocks (baseline,
  then chain, then graph) rather than interleaved to limit that, but did not measure the effect.
- **The refusal metric** matches the exact fixed sentence. A model that refused in other words would be
  counted as answering. It did not happen here, but it could with other models or prompts.
- **My negatives are mostly easy.** They do not test a model that is tempted to invent an answer.

## Reproduce

```bash
pip install -r requirements-lc.txt
PDF_PATH=/path/to/NEWHANDBOOK_AUG8_2025_Final_optimized.pdf python -m eval.compare
```

Use `--versions baseline chain` to skip the slow graph version, and `--limit 5` for a quick check.
