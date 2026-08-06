# RAGAS evaluation

Automated quality scoring for the chatbot's RAG pipeline. Every question in
`golden_dataset.json` is pushed through the **real** production path and RAGAS
scores the result on four metrics.

"Real" is the whole point: `run_ragas_eval.py` imports the same functions
`/chat` calls — `main.retrieve_docs()` (k=6 → hidden-repo filter → top 3),
`main.build_prompt()`, `main.SYSTEM_PROMPT`, and the same `gpt-4o-mini` call —
rather than reimplementing them. If retrieval changes, the eval changes with
it. The only difference is `stream=False`, which affects delivery, not content.

## Layout

Everything lives under `backend/evals/`, and the eval virtualenv sits at the
repo root:

```
portfolio-ai/
├── .venv-eval/                     # eval-only venv (gitignored)
└── backend/
    ├── .env                        # OPENAI_API_KEY lives here
    ├── requirements.txt            # app deps
    └── evals/
        ├── run_ragas_eval.py
        ├── golden_dataset.json
        ├── requirements-eval.txt
        └── results/                # CSVs gitignored, JSON summaries committed
```

All commands below are run **from the repo root**.

## Setup (one time)

Eval dependencies get their own venv because ragas needs the langchain 0.3.x
line — reasons and pins are documented in `requirements-eval.txt`. Install both
requirement files in a **single** pip command so the resolver honors those pins
for the app's unpinned dependencies too:

```bash
python -m venv .venv-eval
.venv-eval/bin/pip install -r backend/requirements.txt -r backend/evals/requirements-eval.txt
```

Needs `OPENAI_API_KEY` in `backend/.env` — the same key the app already uses.
The script loads that file explicitly, so it works regardless of where you
launch it from.

## Run

```bash
.venv-eval/bin/python backend/evals/run_ragas_eval.py                 # full run
.venv-eval/bin/python backend/evals/run_ragas_eval.py --limit 2       # cheap smoke run
.venv-eval/bin/python backend/evals/run_ragas_eval.py --judge gpt-4o  # stricter judge
```

| Flag | Default | What it does |
|---|---|---|
| `--dataset` | `backend/evals/golden_dataset.json` | Path to the question set |
| `--judge` | `gpt-4o-mini` | Model RAGAS uses to score. `gpt-4o` is stricter and pricier |
| `--threshold` | `0.7` | Any question scoring below this on any metric gets flagged |
| `--limit` | none | Only run the first N questions |

Cost: roughly 8 questions × (1 generation + ~10–15 small judge calls) on
`gpt-4o-mini` — a few cents per full run.

Because RAGAS metrics are themselves LLM prompts, the judge model matters. The
judge runs at `temperature=0` so repeat runs stay comparable, but scores from a
`gpt-4o-mini` run and a `gpt-4o` run are **not** comparable to each other.

## What gets saved

Results print to the console and land in `backend/evals/results/`:

- `ragas-<timestamp>.csv` — full per-question detail, including the retrieved
  contexts and generated answers. **Gitignored**, because those contexts contain
  chunks of the resume PDF, which is deliberately not committed.
- `ragas-<timestamp>.json` — timestamp, judge model, question count, and the
  metric means. No document content, so this one is committed and gives you a
  score history to diff against code changes.

## Reading the scores (all 0 to 1, higher is better)

| Metric | Question it answers | Low score usually means |
|---|---|---|
| `faithfulness` | Is every claim in the answer backed by the retrieved chunks? | Model is adding facts the chunks don't contain (hallucination) |
| `answer_relevancy` | Does the answer actually address the question? | Vague, padded, or off-topic answers |
| `context_precision` (no-reference) | Are the retrieved chunks relevant and ranked well? | Retrieval pulls noise; try a different k, chunking, or embedding model |
| `context_recall` | Did retrieval fetch everything the reference answer needs? | The right facts exist but weren't retrieved — or aren't indexed at all |

Rule of thumb: **recall and precision problems are retrieval problems** (fix
chunking, k, embeddings). **Faithfulness problems are prompt or model problems**
(tighten the system prompt, lower temperature, upgrade the model).

## Baseline

First recorded run, 8 questions, `gpt-4o-mini` judge:

| Metric | Score |
|---|---|
| faithfulness | 0.852 |
| answer_relevancy | 0.907 |
| context_precision | 0.760 |
| context_recall | 0.938 |

Recall is healthy — the right chunks are getting retrieved. Precision is the
weakest number, which points at ranking rather than coverage: retrieval is
pulling the relevant material *plus* noise. That's the thread to pull first.

## Extending

Add rows to `golden_dataset.json` — a `question` plus a `reference` written from
facts that genuinely exist in the knowledge base (`backend/data/profile.json`,
the resume PDF, visible repo READMEs).

If a fact isn't in the knowledge base, retrieval can never find it, and
`context_recall` will punish the pipeline for a dataset bug rather than a real
one. Hidden repos are filtered before retrieval, so never write a reference
that depends on one.