# Comparison: hand-written vs LangChain vs LangGraph

All numbers below come from real runs of the code on this branch on 2026-11-18. Raw per-question results
are in `eval/compare_*_results.jsonl` and aggregates in `eval/compare_*_summary.json`. Nothing was tuned
to improve a score.

An earlier version of this file was written from a first run. Since then a code review and an
architecture review changed the graph and the evaluation tooling, and every run below was repeated on
the final code. The final graph's answers and retrieved pages are identical to the first run on all 26
questions, so the conclusions about accuracy did not move. What changed is how much is now known about
latency, the verifier, and whether the retry loop ever helps.

## Setup

| | |
|---|---|
| Document | `NEWHANDBOOK_AUG8_2025_Final_optimized.pdf` (35 pages, sha256 `476a0394...eca325`) |
| Main dataset | 26 questions in `eval/eval_dataset_extended.jsonl`: 16 answerable (each with `key_facts`) and 10 negative |
| Adversarial dataset | 8 more negatives in `eval/eval_dataset_adversarial.jsonl`: the topic is covered but the specific detail is absent (for example "how much is charged for a lock replacement"). Absence was verified by keyword search of the full PDF text |
| Models | `llama3.1:8b` at temperature 0 for generation, grading, rewriting and verification; `nomic-embed-text` for embeddings, both through local Ollama |
| Retrieval depth | per-question `k` from the dataset (6, or 8 for two questions) for the main run; `k=10` and `k=2` runs described below |
| Packages | langchain-core 1.8.3, langchain-classic 1.0.8, langchain-community 0.4.2, langchain-ollama 1.1.0, langchain-text-splitters 1.1.3, langgraph 1.2.22, rank-bm25 0.2.2, faiss-cpu 1.13.2 |

## Main results (26 questions, 2 repeats)

| Metric | Hand-written | LangChain | LangGraph |
|---|---|---|---|
| Recall@K (n=16 answerable) | 1.000 | 1.000 | 1.000 |
| MRR | 0.859 | 0.969 | 0.969 |
| Refusal correctness on negatives (n=10) | 10/10 | 10/10 | 10/10 |
| Over-refusal on answerables (n=16) | 0/16 | 0/16 | 0/16 |
| Key-fact recall, mean (n=16) | 0.865 | 0.865 | 0.865 |
| Answers containing all key facts | 12/16 | 12/16 | 12/16 |
| Mean latency per question (s), runs 1 and 2 | 11.0 (10.5, 11.5) | 10.1 (8.1, 12.1) | 25.1 (24.9, 25.2) |
| Median latency per question (s) | 9.2 | 9.0 | 23.6 |
| Questions with identical results in both repeats | 26/26 | 26/26 | 26/26 |

How the metrics are defined:

- **Recall@K** is 1 if any gold page is among the retrieved pages; **MRR** is 1 over the rank of the first
  gold page. Both are page level and use the chunks the pipeline actually returned in that run.
- **Refused** means the answer contains the fixed refusal sentence (apostrophe-insensitive).
- **Key facts** are short phrases a correct answer must contain, written from the gold answer and the
  handbook text (`a|b` accepts either). Matching is whole-word, case-insensitive, with page citations
  ignored and hyphens, commas inside numbers and parentheses normalized. It is a cheap objective check,
  not a correctness judgment: it cannot catch a wrong claim that sits next to the right facts.
- **Latency** is wall-clock time for the whole pipeline call after an untimed warm-up. "Runs" are two full
  repeats of the whole comparison. Temperature 0 made every question's answer, retrieved chunks, refusal
  flag and graph grades identical across the two repeats.

## Analysis

### Retrieval: the framework versions ranked slightly better, on 3 questions

Recall@K is 1.0 for all three. MRR is 0.859 for the hand-written version and 0.969 for both framework
versions, which return the same pages on every answerable question. The whole gap comes from 3 of the 16
questions (Q1, Q2, Q16), where the hand-written version ranked the gold page 2nd, 4th and 2nd. I checked the
hand-written side on the cached chunks: its keyword stage scores raw substring counts and puts those hits
first, so for Q1 (top keyword hits are pages 22, 19, 21) common words like "are" and "allowed" push page
22 ahead. BM25 weights rare terms like "pets" more, and the ensemble favors chunks both retrievers return,
which is consistent with the better ranking, but I did not isolate the cause on the LangChain side.

With 16 questions one question moves MRR by about 0.03 to 0.05, so read this as "not worse, possibly a bit
better", not a proven win.

### Answer correctness: the same four answers fall short in every version

On the key-fact check all three get 12 of 16 answers fully right, and it is the same four in every version:

| Question | Missing | Likely reason |
|---|---|---|
| Q2 (which animals are allowed) | service animals | the answer lists only fish |
| Q6 (guest responsibilities) | informing guests of policies | the answer says residents are responsible but not that they must inform |
| Q7 (prohibited items) | all 5 expected items (candles, propane, hoverboards, pressure cookers, air conditioners) | page 12 is a two-column permitted/prohibited table; the extracted text runs the columns together with the headings out of order ("Prohibited" appears only at the very end, and candles sit under a "Permitted Flammable Items" label), so all three versions list items from other pages. This is my best explanation and I have not confirmed it against the PDF layout |
| Q15 (alcohol, under-21 resident) | the "unless the of-age roommate is present" exception | the answer says no and omits the exception |

At the UI's depth (`k=10`, hand-written and LangChain only) LangChain gets 14 of 16 fully right
(Q2 and Q15 now correct) and the hand-written version stays at 12 of 16. Key-fact recall is 0.917 against
0.865, MRR 0.969 against 0.846, and latency 18.2 s against 16.6 s. More context helped LangChain here and
not the hand-written version.

The key-fact matcher had a bug that a code review caught (a trailing space made `"no "` match "not" and
"enough", and `"2"` could match `[Page 29]`). I fixed it and re-scored every saved answer with the old and the
new logic: they agree on every record, so the numbers above did not change. The check remains an upper
bound on correctness, not proof of it.

### Refusal: a tie at the ceiling, and a metric that is too strict

All three refused all 10 negatives and refused none of the 16 answerable questions. Zero failures in 10 is
consistent with a true failure rate up to about 26% (exact one-sided 95% bound), and 0 of 16 over-refusals
with up to about 17%.

The 8 adversarial negatives are harder. Strictly scored, hand-written and LangChain refuse 7 of 8 and
LangGraph 8 of 8. The one difference is A1 ("how much is charged for a lock replacement"):

- Hand-written: "...they will be charged for a lock replacement." (true, but it does not answer the question)
- LangChain: the same sentence plus "However, the specific amount is not mentioned in the provided context."
- LangGraph: it generated the LangChain answer, its LLM verifier rejected it, and the fixed refusal replaced it.

**No version invented an amount.** The strict metric counts the first two as failures only because they did
not use the exact refusal sentence, and LangGraph's 8/8 comes from the verifier overriding an honest answer.
So the real hallucination rate on these 8 is 0 of 8 for every version, and the graph's refusal advantage is a
scoring artifact, not evidence that it is safer.

### Latency: the graph costs about 2.5 times the chain, and grading is the biggest piece

Hand-written and LangChain are indistinguishable: 11.0 s against 10.1 s, but LangChain's own two runs were
8.1 s and 12.1 s, so run-to-run variation is larger than the difference. LangGraph averages 25.1 s, about
2.5 times LangChain and about 15 s more per question, and was steady across its two runs (24.9 s and 25.2 s).

Where the graph's time goes (mean seconds per question, over all 26):

| Node | Seconds | Share |
|---|---|---|
| grade_context | 10.70 | 43% |
| generate | 7.55 | 30% |
| verify | 6.23 | 25% |
| rewrite_query | 0.37 | 1% |
| retrieve | 0.02 | 0% |

A single grading call averages 8.9 s and a model verify call about 10 s, about as much as generation, even
though both reply with one word. The prompts contain the full retrieved context, so reading it appears to
dominate. A repeated grade (a question graded twice or more) costs about 7.4 s per call, roughly 17%
cheaper than a first one: some prompt reuse, far from free. The architecture review guessed repeated
grading would be nearly free, and the measurement says otherwise.

### Where the graph helped, and where it hurt

**Helped: nothing measurable on accuracy.**
- On all 16 answerable questions the graph's answers equal the plain chain's.
- The retry loop never ran on an answerable question in the main run (all 8 rewrites were on negatives) and
  never changed an outcome.
- **Rescue test (`k=2`).** At 2 chunks the first retrieval misses the gold page more often, so this is the
  case the grade-and-rewrite loop exists for. One question missed (Q5, gold page 11, retrieved page 10).
  The grader still said "relevant", so no rewrite happened and both versions then refused. The loop
  rescued nothing. The grader said "yes" for all 16 answerable questions, including that miss.

**Hurt:**
- At `k=2` the graph over-refused 2 of 16 answerable questions against the chain's 1 of 16, and fully
  matched 11 of 16 key facts against 12. The extra case is Q15: the chain gave a correct, complete answer
  (including the roommate exception) and the verifier rejected it, so the user got a refusal instead.
- 2.5 times the latency for the same answers.
- Retry waste, now fixed: before the stall check existed, all 7 questions that used both retries repeated
  the same query a second time. The fix removes one search and one grading call on those 7, and answers,
  pages and grades are unchanged on all 26. Measured latency on those questions did not improve
  (23.3 s before, 24.5 s after, inside the run-to-run noise), so I cannot claim a speed benefit.

**Visible, which is valuable for learning:** the per-node trace shows why each question took the path it did.

### The verifier and the number check

The first comparison never exercised the verifier. `eval/verifier_check.py` now shows it three kinds of
answer for 12 questions, against the real retrieved context. A fabrication is exactly the model's own
supported answer plus one appended invented sentence, so any "no" is a reaction to the invented detail.
The script checks the invented text is absent from the context.

| Answer shown | n | LLM verifier rejected | Number check flagged | Either rejected | Right behavior |
|---|---|---|---|---|---|
| Control (supported answer) | 12 | 0 | 0 | 0 | accept |
| Seeded contradiction | 12 | 12 | 8 | 12 | reject |
| Control plus one invented sentence | 12 | 4 | 8 | 9 | reject |

- The LLM verifier alone is good at blatant contradictions (12 of 12) and weak at an appended invented
  detail: it accepted 8 of 12. An earlier hand-written version of this test showed 10 of 12 rejected, which
  was inflated because those answers differed from the context in other ways too.
- The **number check** (`lc_rag/grounding.py`, now part of the graph's verify step) rejects any answer
  containing a number that appears nowhere in the retrieved context. It flagged 0 of the 12 controls, and in
  all final graph runs it never rejected an answer (0 of 26 + 8 + 26 runs), so it caused no over-refusal.
  Combined with the LLM verifier, 9 of 12 fabrications are caught.
- **Caveat:** most of my invented details are numbers, which is exactly what this check sees, so 9 of 12
  overstates how it would do on invented facts in general. The 3 misses are Q5 ("alcohol permit") and Q9
  ("student panel vote"), which contain no numbers, and Q14 ("30 days"), because a printed page-header
  number "30" is inside that page's chunk text so the check sees "30" in the context.
- It would not catch an omission (Q15) or a wrong claim that reuses a real number.

### Is the 8B local model a reliable grader?

For clearly off-topic context, yes. For the hard case, no.

- **Easy cases:** on the 10 main negatives it said "no" 8 times and "yes" 2 times; on the 16 answerable
  questions it said "yes" every time.
- **Hard cases:** on the 8 adversarial negatives (topic covered, detail absent) it said "yes" 5 of 8 times,
  and at `k=2` it said "yes" to the one retrieval that missed the gold page. It measures topical relevance,
  not whether the question is answerable.
- That barely matters for correctness here, because the generator and verifier decide the outcome and a
  rewrite cannot find a detail the document does not contain. The grader mostly acts as a latency switch.

## What the LangChain version does differently

I could not reproduce the hand-written behavior exactly, so the two are not a controlled swap of one
component.

| Area | Hand-written | LangChain version |
|---|---|---|
| Hybrid merge | Keyword hits first, then vector hits, with a quota of `max(2, k // 2)` keyword slots | `EnsembleRetriever` fuses BM25 and FAISS with weighted reciprocal rank fusion (0.5 / 0.5). LangChain has no built-in keyword-first merge, so this cannot be reproduced exactly |
| Keyword scoring | Raw substring counts | BM25, with the baseline's tokenizer and stopwords |
| Zero-score keyword hits | Dropped | Stock `BM25Retriever` returns arbitrary documents when nothing matches, so I subclassed it to drop zero scores |
| Chunking | Fixed 900/350 sliding window, 204 chunks | `RecursiveCharacterTextSplitter` at 900/350 on word boundaries, 184 chunks |
| Page text | PyMuPDF, normalized | `PyMuPDFLoader`, then the baseline's `normalize_for_search` |
| Embeddings call | `/api/embeddings`, one request per chunk | `OllamaEmbeddings` (`/api/embed`, batched). Vectors are L2-normalized in both |
| Generation call | `/api/generate` with one text prompt | `ChatOllama` (`/api/chat`) with the same text as a single human message; the prompt is byte-identical (a test pins this) |
| Answer text | | Identical to the baseline on only 5 of 16 answerable questions in the main run, because the retrieved context and the call path differ |

`EnsembleRetriever` lives in `langchain-classic` in LangChain 1.x, and `langchain-community` prints a
deprecation notice saying it is being sunset. Pins are in `requirements-lc.txt`.

## Limits of this comparison

- **Small samples.** 16 answerable and 10 + 8 negative questions, one document, one model. Small
  differences are not statistically meaningful.
- **Answer correctness is only partly measured.** Key facts check required content, not wrong claims. I read
  answers (truncated) as a sanity check and found incomplete ones (Q2, Q6, Q7, Q15).
- **Latency** is from two repeats. Run-to-run variation was up to about 50% for the faster versions, so
  small differences between them mean nothing.
- **Prompt length.** Ollama's default context is 2048 tokens. Measured prompts were about 1.5k tokens at
  `k=6` to 8 and 1.8k at `k=10`, so nothing was truncated in the evaluated settings. A `k=10` verify prompt
  with a long answer could cross the limit; I did not measure that case.
- **Refusal scoring is exact-sentence matching.** It undercounts honest hedged answers (A1). Whether a
  partial answer that names the missing detail should count as correct is a policy decision, not settled here.
- **The number check and the seeded tests are mine.** Their results depend on the invented details I chose.
- **PDF extraction.** Two-column tables (page 12) extract as a flat run of text with the headings out of
  order. Fixing that needs layout-aware extraction and was not attempted.

## Reproduce

```bash
pip install -r requirements-lc.txt
export PDF_PATH=/path/to/NEWHANDBOOK_AUG8_2025_Final_optimized.pdf
python -m eval.compare --repeats 2                       # main table
python -m eval.compare --dataset eval/eval_dataset_adversarial.jsonl --out-prefix eval/compare_adversarial
python -m eval.compare --k 10 --versions baseline chain --out-prefix eval/compare_k10
python -m eval.compare --k 2 --versions chain graph --out-prefix eval/compare_k2
python -m eval.verifier_check                            # seeded-error test of the verifier
python -m eval.compare --rescore                         # re-score saved answers, no models run
```
