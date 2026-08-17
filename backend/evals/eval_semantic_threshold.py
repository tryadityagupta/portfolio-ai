"""
eval_semantic_threshold.py
==========================

Picks SEMANTIC_CACHE_THRESHOLD with evidence instead of vibes.

WHAT IT SIMULATES — production lookup semantics exactly:

  The "cached corpus" is every golden_dataset.json question plus every
  `cached` question in semantic_pairs.json. For each labeled `query` we find
  its BEST match over the WHOLE corpus (that is what the FAISS lookup does —
  it doesn't know which cached question you 'meant'), then, per candidate
  threshold, decide hit/miss the same way main.py does.

SCORING — asymmetric on purpose:

  false positive  = served a cached answer we should NOT have
                    (wrong reuse, or right-to-reuse but matched a DIFFERENT
                    cached question) — this is the failure mode that makes
                    the chatbot confidently wrong, so it dominates.
  false negative  = missed a safe reuse — costs one extra gpt-4o-mini
                    generation. Cheap.

  Recommendation printed = the LOWEST threshold with zero false positives
  (lowest ⇒ most cache hits within the safe region).

COST: one real embedding per unique string (~20 strings, text-embedding-3-
small) — well under a cent. Needs OPENAI_API_KEY in backend/.env.

Usage (from backend/):
    python evals/eval_semantic_threshold.py
    python evals/eval_semantic_threshold.py --margin 0.03   # also test a margin
    python evals/eval_semantic_threshold.py --thresholds 0.90 0.92 0.94
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

import numpy as np

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(EVALS_DIR)
sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BACKEND_DIR, ".env"))
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set — add it to backend/.env.")

from rag import get_embeddings  # noqa: E402  (the SAME model the app uses)

DEFAULT_THRESHOLDS = [0.88, 0.90, 0.92, 0.94, 0.95]


def unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype="float32")
    return a / np.linalg.norm(a)


async def embed_all(texts: list[str]) -> dict[str, np.ndarray]:
    emb = get_embeddings()
    out = {}
    for t in texts:
        out[t] = unit(await emb.aembed_query(t))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=DEFAULT_THRESHOLDS)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="also require top - runner_up >= margin (like "
                         "SEMANTIC_CACHE_MIN_MARGIN)")
    args = ap.parse_args()

    with open(os.path.join(EVALS_DIR, "golden_dataset.json")) as f:
        golden_qs = [r["question"] for r in json.load(f)]
    with open(os.path.join(EVALS_DIR, "semantic_pairs.json")) as f:
        pairs = json.load(f)

    corpus = sorted({*golden_qs, *(p["cached"] for p in pairs)})
    everything = sorted({*corpus, *(p["query"] for p in pairs)})

    print(f"Embedding {len(everything)} unique strings "
          f"({len(corpus)} cached-corpus, {len(pairs)} labeled queries)...")
    vecs = asyncio.run(embed_all(everything))

    # Best + runner-up cached match per labeled query — production semantics.
    rows = []
    for p in pairs:
        sims = sorted(((float(vecs[p["query"]] @ vecs[c]), c) for c in corpus),
                      reverse=True)
        (top_sim, top_q), runner_sim = sims[0], (sims[1][0] if len(sims) > 1
                                                 else -1.0)
        rows.append({**p, "top_sim": top_sim, "top_match": top_q,
                     "runner_sim": runner_sim})

    print("\n=== Per-pair similarities (query -> best cached match) ===")
    for r in rows:
        flag = "REUSE-OK " if r["should_reuse"] else "MUST-MISS"
        aimed = "" if r["top_match"] == r["cached"] else \
            f"  !! best match is a DIFFERENT question: {r['top_match'][:44]!r}"
        print(f"  [{flag}] {r['top_sim']:.4f}  {r['query'][:58]!r}{aimed}")

    print(f"\n=== Threshold sweep (margin={args.margin}) ===")
    print(f"  {'thresh':>6} {'hits':>5} {'FP':>3} {'FN':>3}  verdict")
    report, recommended = [], None
    for th in sorted(args.thresholds):
        fp = fn = hits = 0
        for r in rows:
            hit = (r["top_sim"] >= th and
                   (args.margin <= 0.0 or
                    r["top_sim"] - r["runner_sim"] >= args.margin))
            if hit:
                hits += 1
                # A hit is only CORRECT if reuse was safe AND the winning
                # cached question is the one whose answer fits.
                if not (r["should_reuse"] and r["top_match"] == r["cached"]):
                    fp += 1
            elif r["should_reuse"]:
                fn += 1
        verdict = "UNSAFE (serves wrong answers)" if fp else \
            ("safe, most hits" if recommended is None else "safe")
        if fp == 0 and recommended is None:
            recommended = th
        report.append({"threshold": th, "hits": hits,
                       "false_positives": fp, "false_negatives": fn})
        print(f"  {th:>6.2f} {hits:>5} {fp:>3} {fn:>3}  {verdict}")

    if recommended is not None:
        print(f"\nRecommended SEMANTIC_CACHE_THRESHOLD={recommended:.2f} — "
              f"lowest tested value with zero false positives on this set.")
    else:
        print("\nNo tested threshold was false-positive-free — raise the "
              "candidate range and/or add a margin before enabling reuse.")
    print("Correctness outranks hit rate: a false negative costs one "
          "gpt-4o-mini generation; a false positive costs a wrong answer.\n"
          "Cross-check against real traffic: /admin/analytics reports the "
          "similarity distribution of live semantic hits.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(EVALS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"semantic-threshold-{stamp}.json")
    with open(out_path, "w") as f:
        json.dump({"margin": args.margin, "pairs": rows, "sweep": report,
                   "recommended": recommended}, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
