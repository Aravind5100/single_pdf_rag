| Metric | Hand-written | LangChain | LangGraph |
|---|---|---|---|
| Recall@K (n=16 answerable) | 1.000 | 1.000 | 1.000 |
| MRR | 0.859 | 0.969 | 0.969 |
| Refusal correctness on negatives (n=10) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Over-refusal on answerables (n=16) | 0/16 (0%) | 0/16 (0%) | 0/16 (0%) |
| Key-fact recall, mean (n=16 answerable) | 0.865 | 0.865 | 0.865 |
| Answers containing all key facts | 12/16 | 12/16 | 12/16 |
| Mean latency per question (s) | 11.0 (runs: 10.5, 11.5) | 10.1 (runs: 8.1, 12.1) | 25.1 (runs: 24.9, 25.2) |
| Median latency per question (s) | 9.2 | 9.0 | 23.6 |
| Questions with identical results in every repeat (answer, chunks, refusal, grades) | 26/26 | 26/26 | 26/26 |

LangGraph decisions

| Decision | Count |
|---|---|
| Questions that triggered at least one rewrite | 8 |
| Mean rewrites per question | 0.58 |
| Grader, first decision, answerable questions: yes / no | 16 / 0 |
| Grader, first decision, answerable questions whose gold page was retrieved: yes / no | 16 / 0 |
| Grader, first decision, negative questions: yes / no | 2 / 8 |
| Answerable: generator already refused / verifier rejected an answer | 0 / 0 |
| Negative: generator already refused / verifier rejected / verifier passed | 10 / 0 / 0 |
| Final refusals by source: generator / number check / LLM verifier / step limit | 10 / 0 / 0 / 0 |
| Why the retrieve/grade loop stopped | relevant: 19, stalled: 7 |

LangGraph: mean seconds per question spent in each node

| Node | Seconds |
|---|---|
| generate | 7.55 |
| grade_context | 10.70 |
| refuse | 0.00 |
| retrieve | 0.02 |
| rewrite_query | 0.37 |
| verify | 6.23 |
