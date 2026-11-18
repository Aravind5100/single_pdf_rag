| Metric | Hand-written | LangChain | LangGraph |
|---|---|---|---|
| Recall@K (n=0 answerable) | n/a | n/a | n/a |
| MRR | n/a | n/a | n/a |
| Refusal correctness on negatives (n=8) | 7/8 (88%) | 7/8 (88%) | 8/8 (100%) |
| Over-refusal on answerables (n=0) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| Mean latency per question (s) | 9.5 | 9.8 | 18.2 |
| Median latency per question (s) | 9.3 | 9.5 | 16.4 |

LangGraph decisions

| Decision | Count |
|---|---|
| Questions that triggered at least one rewrite | 3 |
| Mean rewrites per question | 0.62 |
| Grader, first decision, answerable questions: yes / no | 0 / 0 |
| Grader, first decision, answerable questions whose gold page was retrieved: yes / no | 0 / 0 |
| Grader, first decision, negative questions: yes / no | 5 / 3 |
| Answerable: generator already refused / verifier rejected an answer | 0 / 0 |
| Negative: generator already refused / verifier rejected / verifier passed | 7 / 1 / 0 |
| Final refusals by source: generator / number check / LLM verifier / step limit | 7 / 0 / 1 / 0 |
| Why the retrieve/grade loop stopped | relevant: 6, stalled: 2 |

LangGraph: mean seconds per question spent in each node

| Node | Seconds |
|---|---|
| generate | 4.23 |
| grade_context | 12.25 |
| refuse | 0.00 |
| retrieve | 0.02 |
| rewrite_query | 0.45 |
| verify | 1.24 |
