# CLAUDE.md

Instructions for Claude Code in this repository.

## Project

A single-PDF RAG system with three implementations that are compared against each other: a
hand-written baseline (modules in the repo root), a LangChain version and a LangGraph version
(`lc_rag/`). Everything runs locally through Ollama (`llama3.1:8b`, `nomic-embed-text`).

Read `PROJECT_CONTEXT.md` before making changes, and update it at the end of each phase of work.
Re-check every number in it against the `eval/` outputs first.

## Subagents

The agents in `.claude/agents/` are mine. Use them at these points without waiting to be asked:

- **`graph-architect`**: before building or changing the LangGraph graph (state fields, nodes,
  conditional edges). Give it the proposed change and act on its loop-risk and grader-assumption flags
  before writing code.
- **`code-reviewer`**: at the end of each phase, on the diff, before asking me for approval. Report what
  it found, including anything it flags that could alter eval results.
- **`test-runner`**: after code changes, to run pytest and the eval and report exact numbers.

Notes:
- `graph-architect` and `code-reviewer` are read-only, so they can run at any time.
- `test-runner` calls Ollama when it runs the eval. Do not run it while a long `eval.compare` run is in
  progress, because it would distort that run's latency numbers.
- Do not paraphrase an agent's findings into something softer. Quote the problem as it reported it.

## Working agreements

- The hand-written baseline must stay runnable and unchanged in behavior. After any change, run
  `python -m eval.eval_runner` and confirm the output is byte-identical to `eval/baseline_results.jsonl`.
- Never invent a result. Every number must come from a real run of the eval harness.
- No em dashes in docs, comments or commit messages. No absolute machine-specific paths and no secrets in
  code or docs.
- Make small commits with clear messages, one logical change each. Commit or push only when I ask, and use a
  branch rather than committing to `main`.
- Explain steps briefly, and pause for my OK at the end of each phase.

## Commands

Run everything from the repository root with `PDF_PATH` set to the PDF.

```bash
python -m pytest                       # offline tests, no Ollama needed
python -m eval.eval_runner             # baseline retrieval eval
python -m eval.compare                 # three-way comparison (needs Ollama)
```
