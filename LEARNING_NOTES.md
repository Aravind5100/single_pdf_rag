# Learning notes: State, nodes and conditional edges

These notes explain the three ideas behind LangGraph in plain language, using the graph in
[`lc_rag/graph.py`](lc_rag/graph.py) as the running example.

## The big picture

The hand-written pipeline and the LangChain chain are straight lines: retrieve, then generate, done.
A graph lets the program **look at what happened and choose the next step**. For example: "the
retrieved text does not look relevant, so rewrite the question and search again." A straight line
cannot do that, a graph can.

LangGraph gives you three building blocks for this.

## 1. State: the shared notebook

**State** is one shared notebook that travels through the whole run. Every step can read it and
write to it. In code it is a typed dictionary, `RagState`:

| Field | What it holds |
|---|---|
| `question` | The user's original question. It never changes. |
| `k` | How many chunks to retrieve. |
| `rewritten_query` | The latest rewritten search query (empty until a rewrite happens). |
| `chunks` | The chunks from the most recent retrieval. |
| `relevant` | The grader's yes/no for those chunks. |
| `answer` | The answer text so far. |
| `retries` | How many rewrites have happened. This is what stops endless loops. |
| `queries` | Every query searched so far, the question first. The stall check looks at all of them. |
| `stalled` | True when the latest rewrite is identical to a query that was already searched, so another search would find nothing new. |
| `stop_reason` | Why the retrieve/grade loop stopped: `relevant`, `retries_exhausted` or `stalled`. |
| `refusal_source` | Who refused: `generator` (it wrote the refusal itself), `grounding_check` (a number in the answer was not in the context), `verifier` (the model overrode an answer) or `step_limit` (the safety net tripped); empty if nobody refused. |
| `timings` | `node=seconds` for every node that ran, so you can see where the time goes. |
| `supported` | The verifier's yes/no for the answer. |
| `trace` | A running log of what each step decided, so you can read the path afterwards. |

A step does not rewrite the whole notebook. It returns only the fields it wants to change, and
LangGraph merges them in. `trace` is special: it is declared with a **reducer** (`operator.add`),
which tells LangGraph "when a step returns new trace lines, append them to the old ones instead of
replacing them."

## 2. Nodes: the steps

A **node** is a plain Python function. It takes the state, does one job, and returns the changes.
Mine are:

| Node | Its one job |
|---|---|
| `retrieve` | Search the PDF (BM25 plus FAISS) using the rewritten query if there is one, else the question. |
| `grade_context` | Ask the model "does this context help answer the question? yes or no". |
| `rewrite_query` | Ask the model for a better search query, add 1 to `retries`, and note whether the new query is identical to one already searched (`stalled`). |
| `generate` | Answer the original question from the chunks, using the grounded prompt. |
| `verify` | First check that every number in the answer appears in the context (a plain string comparison, no model), then ask the model "is every claim in this answer supported by the context? yes or no". |
| `refuse` | Replace the answer with the fixed refusal sentence. |

Nodes do not decide what comes next. They only do their job and update the notebook.

## 3. Edges: the arrows

**Edges** connect nodes. There are two kinds.

**Normal edges** always go to the same place. `retrieve` always goes to `grade_context`.
`generate` always goes to `verify`.

**Conditional edges** run a small **routing function** that reads the state and returns the name of
the next node. This is where decisions live. I have three:

```python
def route_after_grade(state):
    if state["relevant"] or state["retries"] >= max_retries:
        return "generate"
    return "rewrite_query"

def route_after_rewrite(state):
    return "generate" if state["stalled"] else "retrieve"

def route_after_verify(state):
    return END if state["supported"] else "refuse"
```

## The whole graph

```mermaid
flowchart TD
    S([START]) --> R[retrieve]
    R --> G[grade_context]
    G -->|relevant, or retries used up| GEN[generate]
    G -->|not relevant and retries < 2| RW[rewrite_query]
    RW -->|new query| R
    RW -->|same query as before| GEN
    GEN --> V[verify]
    V -->|supported| E([END])
    V -->|not supported| RF[refuse]
    RF --> E
```

## Why the loop cannot run forever

The loop is `retrieve -> grade_context -> rewrite_query -> retrieve`. Three things stop it:

1. **The retry counter.** `rewrite_query` adds 1 to `retries`, and `route_after_grade` refuses to
   loop again once `retries` reaches 2. So there are at most 3 retrievals per question.
2. **The stall check.** If a rewrite produces a query identical to any query already searched (the
   previous rewrite, or the original question), searching again would return the same chunks, so
   `route_after_rewrite` skips straight to `generate`. Matching is exact after collapsing whitespace: a
   change in case or punctuation counts as a new query, because it can change what the embedding search
   returns. (In my first comparison run, all 7 questions that used both retries wasted the second one this
   way, which is why the check exists.)
3. **A recursion limit.** LangGraph is also told to abort any run that takes too many steps. The limit
   is `3 * max_retries + 6` (never below 25), always above the longest possible path of
   `3 * max_retries + 5` steps. If a run ever did hit it, the graph returns the fixed refusal instead of
   raising an error. This is a safety net in case the other two ever have a bug.

## A worked example from a real run

Question: *Does the handbook describe rules or fines related to student parking violations?*
(run with `--k 6`, so each retrieval returns 6 chunks)

```
retrieve      -> 6 chunks
grade_context -> relevant=no  (retries 0/2)
rewrite_query -> "Student parking handbook rules and fines"  (retries now 1)
retrieve      -> 6 chunks
grade_context -> relevant=no  (retries 1/2)
rewrite_query -> "Student parking handbook rules and fines"  (retries now 2, unchanged, so stop retrying)
generate      -> the model already answered with the refusal sentence
verify        -> skipped, because a refusal makes no claims to check
refuse        -> fixed refusal sentence
```

You can read this from the `trace` field. Two things to notice. The graph reached the right answer
(the handbook has no such rules). And the second rewrite produced the same query as the first, which
the stall check caught: before that check existed, the graph searched and graded a third time for
nothing. A graph makes behavior like this visible, which is useful for learning even when it does not
improve the score.

## Mapping the ideas to the code

| Idea | Where |
|---|---|
| State | `RagState` in `lc_rag/graph.py` |
| Nodes | the inner functions `retrieve`, `grade_context`, `rewrite_query`, `generate`, `verify`, `refuse` inside `build_graph` |
| Normal edges | `graph.add_edge(...)` |
| Conditional edges | `graph.add_conditional_edges(...)` with `route_after_grade`, `route_after_rewrite` and `route_after_verify` |
| Retry cap | `max_retries` (default `MAX_RETRIES = 2`) and `recursion_limit_for(max_retries)` (`RECURSION_LIMIT` is only its minimum) |
| Reading the path | `result["trace"]`, printed by `python -m lc_rag.cli --mode graph` and shown in the Streamlit page |

## Try it

```bash
PDF_PATH=/path/to/file.pdf python -m lc_rag.cli --mode graph "Are pets allowed in the residence halls?"
```

Then ask something the document does not cover and watch the trace take the rewrite loop.
