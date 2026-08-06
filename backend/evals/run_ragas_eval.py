"""
run_ragas_eval.py
=================

Scores the portfolio chatbot's RAG pipeline with RAGAS on a small "golden"
dataset of questions whose true answers are known to live in the knowledge
base (profile.json / resume / GitHub READMEs).

The important design rule: this script does NOT reimplement the pipeline.
It imports the SAME functions the live /chat endpoint uses —
  * main.retrieve_docs()  -> the exact k=6 -> hidden-filter -> top-3 retrieval
  * main.build_prompt() + main.SYSTEM_PROMPT + the same gpt-4o-mini call
so every score describes production behavior, not an approximation of it.

For each question we capture the RAG "evidence triple":
    user_input          the question
    retrieved_contexts  the 3 chunks retrieval actually returned
    response            the answer the model actually generated
plus a human-written `reference` (ground truth) from golden_dataset.json.

RAGAS then judges four things (LLM-as-judge under the hood):
    faithfulness        is every claim in the answer supported by the chunks?
    answer_relevancy    does the answer actually address the question?
    context_precision   were the retrieved chunks relevant / well ranked?
    context_recall      did retrieval fetch everything the reference needs?

Usage (from the repo root, inside the eval venv — see requirements-eval.txt):
    python evals/run_ragas_eval.py
    python evals/run_ragas_eval.py --limit 2          # cheap smoke run
    python evals/run_ragas_eval.py --judge gpt-4o     # stricter judge
Cost: ~8 questions x (1 generation + ~10-15 small judge calls) on gpt-4o-mini
— a few cents per full run.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Path bootstrap: make `import main` / `import rag` work no matter where this
# script is launched from. backend/ modules import each other as top-level
# names (`import projects`), so backend/ itself must be on sys.path.
# ---------------------------------------------------------------------------
EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EVALS_DIR))  # Go up TWO levels
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv  # noqa: E402

# The app keeps its secrets in backend/.env — load it explicitly so the eval
# works even when launched from the repo root.
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

if not os.getenv("OPENAI_API_KEY"):
    sys.exit(
        "OPENAI_API_KEY is not set. Add it to backend/.env (the same key the "
        "app uses) — the eval needs it for generation, embeddings, and the "
        "RAGAS judge calls."
    )

# App modules — imported AFTER the env check so failures are readable.
import main as app_main  # noqa: E402  (FastAPI app; importing it does not start a server)
import rag  # noqa: E402

# RAGAS + LangChain wrappers (see requirements-eval.txt for version pins).
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402


async def generate_answer(question: str, context: str) -> str:
    """Produce the answer EXACTLY as /chat would — same system prompt, same
    user prompt builder, same model and token cap. The only difference is
    stream=False, because the eval needs the final text, and streaming only
    changes delivery, not content."""
    resp = await app_main.client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=250,
        messages=[
            {"role": "system", "content": app_main.SYSTEM_PROMPT},
            {"role": "user", "content": app_main.build_prompt(
                context, question)},
        ],
    )
    return resp.choices[0].message.content or ""


async def collect_samples(dataset_rows: list[dict]) -> list[SingleTurnSample]:
    """Run every golden question through the real pipeline and package the
    results in the shape RAGAS expects."""
    samples = []
    for i, row in enumerate(dataset_rows, start=1):
        question = row["question"]
        print(f"[{i}/{len(dataset_rows)}] {question}")

        # 1) real retrieval (k=6 -> hidden filter -> top 3), then
        # 2) real generation on those exact chunks.
        docs = await app_main.retrieve_docs(question)
        contexts = [d.page_content for d in docs]
        answer = await generate_answer(question, "\n".join(contexts))

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=row["reference"],
            )
        )
    return samples


def main(args: argparse.Namespace) -> None:
    with open(args.dataset, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if args.limit:
        rows = rows[: args.limit]

    # The server loads the FAISS index in its lifespan hook; there is no
    # server here, so we hydrate the same module-level global ourselves.
    # retrieve_docs() then behaves identically to production.
    print("Loading vector store...")
    app_main.vector_db = rag.load_vector_store()

    print(f"Collecting answers for {len(rows)} questions...\n")
    samples = asyncio.run(collect_samples(rows))

    # ------------------------------------------------------------------
    # The judge. RAGAS metrics are prompts run against an LLM, so WHICH
    # model judges matters. gpt-4o-mini keeps a full run at a few cents;
    # pass --judge gpt-4o before an interview demo for stricter scoring.
    # temperature=0 makes repeat runs comparable.
    # ------------------------------------------------------------------
    judge = LangchainLLMWrapper(ChatOpenAI(model=args.judge, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    metrics = [
        Faithfulness(),                        # answer grounded in chunks?
        ResponseRelevancy(),                   # answer addresses question?
        LLMContextPrecisionWithoutReference(),  # retrieved chunks relevant?
        LLMContextRecall(),                    # retrieval missed anything?
    ]

    print("\nScoring with RAGAS...")
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=4),
    )

    # ------------------------------------------------------------------
    # Report: per-question table, per-metric means, and a below-threshold
    # flag list (the rows worth debugging first).
    # ------------------------------------------------------------------
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c in {
        "faithfulness", "answer_relevancy",
        "llm_context_precision_without_reference", "context_recall",
    }]

    table = df[["user_input"] + metric_cols].copy()
    table["user_input"] = table["user_input"].str.slice(0, 48)
    print("\n=== Per-question scores ===")
    print(table.round(3).to_string(index=False))

    print("\n=== Averages ===")
    for col in metric_cols:
        print(f"  {col:42s} {df[col].mean():.3f}")

    weak = df[(df[metric_cols] < args.threshold).any(axis=1)]
    if len(weak):
        print(f"\n=== Questions below {args.threshold} on any metric ===")
        for _, r in weak.iterrows():
            worst = min(metric_cols, key=lambda c: r[c])
            print(f"  - {r['user_input'][:60]}  ({worst}={r[worst]:.2f})")
    else:
        print(f"\nAll questions scored >= {args.threshold} on every metric.")

    # Persist the full run (including contexts and answers) so score changes
    # can be diffed against code changes over time.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(EVALS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"ragas-{stamp}.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "timestamp": stamp,
        "judge_model": args.judge,
        "n_questions": len(rows),
        "means": {c: round(float(df[c].mean()), 4) for c in metric_cols},
    }
    with open(os.path.join(out_dir, f"ragas-{stamp}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS eval for portfolio-ai")
    parser.add_argument("--dataset",
                        default=os.path.join(EVALS_DIR, "golden_dataset.json"))
    parser.add_argument("--judge", default="gpt-4o-mini",
                        help="Judge model for RAGAS metrics (e.g. gpt-4o)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Flag any question scoring below this")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions (cheap smoke run)")
    main(parser.parse_args())
