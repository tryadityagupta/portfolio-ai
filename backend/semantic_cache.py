"""
semantic_cache.py — in-memory FAISS index over CACHED-QUESTION embeddings.

WHAT THIS IS (plain English):

  The exact cache (analytics.py) only matches a question after trivial
  cleanup — "What are his SKILLS??" == "what are his skills". This module
  adds the second level: "Which projects has Aditya worked on?" can reuse
  the answer cached for "What projects has Aditya built?" because their
  EMBEDDINGS are nearly identical, even though the strings are not.

  It holds ONE small in-memory FAISS index whose vectors are the embeddings
  of previously cached questions, and whose FAISS ids are literally the
  `CachedAnswer.id` primary keys. A lookup therefore returns a database row
  id directly — no separate id-mapping table to keep in sync.

WHAT THIS IS NOT:

  - It is NOT the document index. rag.py's FAISS index answers "which chunks
    are relevant to this question"; this index answers "have we answered a
    question like this before". Mixing them would let cached questions leak
    into retrieval, so they stay two separate objects on purpose.
  - It is NOT persistent. The vectors live in the `answer_cache` table
    (analytics.py owns persistence); this index is rebuilt from those rows
    at startup in one cheap local pass — zero OpenAI calls.
  - It does NOT decide similarity with an LLM. Similarity is plain cosine:
    vectors are L2-normalized so FAISS inner product == cosine similarity
    (that's why the index type is IndexFlatIP).

CONCURRENCY MODEL:

  All callers run on the FastAPI event loop. FAISS calls here are
  microsecond-scale (a few hundred vectors, flat index), so nothing is
  offloaded to threads. A single asyncio.Lock serializes MUTATIONS
  (add/remove/load/reset) because eviction and cache-clear span multiple
  awaits; lookups take the same lock only to guarantee they never interleave
  with a partial rebuild. The lock is never held across network I/O.

  Failures here must never break /chat — same philosophy as analytics.py.
  Every public method catches, warns on stderr, and degrades to "cache miss".
"""

import asyncio
import sys

import faiss
import numpy as np

__all__ = ["cache_index", "SemanticCacheIndex"]


def to_bytes(vec: list[float]) -> bytes:
    """float32 little-endian bytes — the storage form used in the DB.
    ~6 KB per text-embedding-3-small vector vs ~30 KB as a JSON list."""
    return np.asarray(vec, dtype="<f4").tobytes()


def from_bytes(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f4")


def _unit_rows(arr: np.ndarray) -> np.ndarray:
    """(n, d) float32, L2-normalized rows. OpenAI embeddings arrive ~unit
    length already, but normalizing again is cheap and makes the
    inner-product==cosine invariant unconditional."""
    arr = np.ascontiguousarray(arr, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    faiss.normalize_L2(arr)
    return arr


class SemanticCacheIndex:
    def __init__(self):
        self._index: faiss.Index | None = None
        self._dim: int | None = None
        self._lock = asyncio.Lock()
        # strong refs, as in analytics
        self._pending: set[asyncio.Task] = set()

    # -- lifecycle ---------------------------------------------------------

    def _new_index(self, dim: int) -> None:
        # Flat IP = exact search, no training, no recall loss. At the scale
        # of SEMANTIC_CACHE_MAX_ENTRIES (hundreds) this is the right tool;
        # IVF/HNSW would be pure overhead. IDMap2 lets us add/remove by
        # CachedAnswer.id instead of positional ids.
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._dim = dim

    async def load(self, entries: list[tuple[int, bytes]]) -> int:
        """Bulk-(re)populate from DB rows at startup: (cache_id, raw bytes).
        Local math only — never calls any embedding API. Rows whose vector
        length disagrees with the first row (e.g. after an embedding-model
        change) are skipped with a warning rather than poisoning the index."""
        async with self._lock:
            self._index, self._dim = None, None
            loaded = 0
            for cache_id, raw in entries:
                try:
                    vec = from_bytes(raw)
                    if self._index is None:
                        self._new_index(vec.shape[0])
                    if vec.shape[0] != self._dim:
                        print(f"[semantic-cache] skip id={cache_id}: dim "
                              f"{vec.shape[0]} != {self._dim}", file=sys.stderr)
                        continue
                    self._index.add_with_ids(
                        _unit_rows(vec), np.asarray([cache_id], dtype="int64"))
                    loaded += 1
                except Exception as e:
                    print(f"[semantic-cache] skip id={cache_id}: {e!r}",
                          file=sys.stderr)
            return loaded

    async def reset(self) -> None:
        """Forget everything — wired into analytics.cache_clear() so vectors
        never outlive their database rows."""
        async with self._lock:
            self._index, self._dim = None, None

    # -- mutation ----------------------------------------------------------

    async def add(self, cache_id: int, vec: list[float] | np.ndarray) -> None:
        try:
            arr = np.asarray(vec, dtype="float32")
            async with self._lock:
                if self._index is None:
                    self._new_index(arr.shape[-1])
                if arr.shape[-1] != self._dim:
                    print(f"[semantic-cache] refuse add id={cache_id}: dim "
                          f"{arr.shape[-1]} != {self._dim}", file=sys.stderr)
                    return
                # IDMap2 would happily store duplicates of the same id;
                # remove first so re-adding an id is idempotent.
                self._index.remove_ids(np.asarray([cache_id], dtype="int64"))
                self._index.add_with_ids(
                    _unit_rows(arr), np.asarray([cache_id], dtype="int64"))
        except Exception as e:
            print(f"[semantic-cache] add failed: {e!r}", file=sys.stderr)

    async def remove(self, cache_id: int) -> None:
        try:
            async with self._lock:
                if self._index is not None:
                    self._index.remove_ids(
                        np.asarray([cache_id], dtype="int64"))
        except Exception as e:
            print(f"[semantic-cache] remove failed: {e!r}", file=sys.stderr)

    async def remove_many(self, cache_ids: list[int]) -> None:
        if not cache_ids:
            return
        try:
            async with self._lock:
                if self._index is not None:
                    self._index.remove_ids(
                        np.asarray(cache_ids, dtype="int64"))
        except Exception as e:
            print(
                f"[semantic-cache] remove_many failed: {e!r}", file=sys.stderr)

    def remove_later(self, cache_id: int) -> None:
        """Fire-and-forget removal, callable from non-async cleanup spots.
        Used to self-heal a vector whose DB row turned out to be gone."""
        try:
            task = asyncio.get_running_loop().create_task(self.remove(cache_id))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        except RuntimeError:
            pass

    # -- the hot path ------------------------------------------------------

    async def lookup(self, vec: list[float] | np.ndarray, threshold: float,
                     min_margin: float = 0.0) -> tuple[int | None, float]:
        """Return (cache_id_or_None, top_cosine_similarity). Two gates, both
        tunable via env (see main.py):

          1. top similarity >= threshold           — the primary safety gate
          2. top - runner_up >= min_margin         — optional ambiguity gate
             (0.0 disables it; useful once the cache is big enough that two
             DIFFERENT cached questions can both score near the top)

        A near-tie between two different cached questions means the query
        sits between them semantically — exactly when reusing either answer
        is risky — so with a margin configured we prefer a miss. False
        negatives cost one extra generation; false positives cost a wrong
        answer on the portfolio. We buy the former.

        WHY THE SIMILARITY COMES BACK ON A MISS TOO: this used to return a
        bare None, so every miss looked identical in the logs — a 0.91
        near-miss that wants a lower threshold was indistinguishable from an
        unrelated question, a failed embed, or an empty index. The caller
        logs this number, which turns "why didn't that hit?" from a code-
        reading exercise into a column you can sort. 0.0 means no comparison
        happened at all (empty index, wrong dim, or an exception)."""
        try:
            async with self._lock:
                if self._index is None or self._index.ntotal == 0:
                    return None, 0.0
                arr = np.asarray(vec, dtype="float32")
                if arr.shape[-1] != self._dim:
                    return None, 0.0
                k = min(2, self._index.ntotal)
                sims, ids = self._index.search(_unit_rows(arr), k)
            top_sim = float(sims[0][0])
            top_id = int(ids[0][0])
            if top_id == -1:
                return None, 0.0
            if top_sim < threshold:
                return None, top_sim          # near-miss: threshold too high?
            if min_margin > 0.0 and k == 2 and int(ids[0][1]) != -1:
                if top_sim - float(sims[0][1]) < min_margin:
                    return None, top_sim      # rejected as ambiguous
            return top_id, top_sim
        except Exception as e:
            print(f"[semantic-cache] lookup failed: {e!r}", file=sys.stderr)
            return None, 0.0

    def size(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0


# The one shared instance the whole app uses (module-as-singleton, matching
# how analytics.py exposes its state).
cache_index = SemanticCacheIndex()
