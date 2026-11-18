| Metric | Hand-written | LangChain | LangGraph |
|---|---|---|---|
| Recall@K (n=16 answerable) | 1.000 | 1.000 | 1.000 |
| MRR | 0.859 | 0.969 | 0.969 |
| Refusal correctness on negatives (n=10) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Over-refusal on answerables (n=16) | 0/16 (0%) | 0/16 (0%) | 0/16 (0%) |
| Mean latency per question (s) | 8.6 | 9.4 | 23.3 |
| Median latency per question (s) | 7.7 | 8.8 | 21.8 |

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
