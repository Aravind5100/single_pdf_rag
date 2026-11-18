| Metric | LangChain | LangGraph |
|---|---|---|
| Recall@K (n=16 answerable) | 0.938 | 0.938 |
| MRR | 0.938 | 0.938 |
| Refusal correctness on negatives (n=10) | 10/10 (100%) | 10/10 (100%) |
| Over-refusal on answerables (n=16) | 1/16 (6%) | 2/16 (12%) |
| Key-fact recall, mean (n=16 answerable) | 0.833 | 0.771 |
| Answers containing all key facts | 12/16 | 11/16 |
| Mean latency per question (s) | 5.0 | 11.0 |
| Median latency per question (s) | 4.2 | 11.3 |

LangGraph decisions

| Decision | Count |
|---|---|
| Questions that triggered at least one rewrite | 8 |
| Mean rewrites per question | 0.62 |
| Grader, first decision, answerable questions: yes / no | 16 / 0 |
| Grader, first decision, answerable questions whose gold page was retrieved: yes / no | 15 / 0 |
| Grader, first decision, negative questions: yes / no | 2 / 8 |
| Answerable: generator already refused / verifier rejected an answer | 1 / 1 |
| Negative: generator already refused / verifier rejected / verifier passed | 10 / 0 / 0 |
| Final refusals by source: generator / number check / LLM verifier / step limit | 11 / 0 / 1 / 0 |
| Why the retrieve/grade loop stopped | relevant: 18, stalled: 8 |

LangGraph: mean seconds per question spent in each node

| Node | Seconds |
|---|---|
| generate | 4.10 |
| grade_context | 3.88 |
| refuse | 0.00 |
| retrieve | 0.02 |
| rewrite_query | 0.41 |
| verify | 2.56 |
