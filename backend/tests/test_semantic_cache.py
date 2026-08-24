"""
Semantic-cache behavior tests. Each test name states the invariant it locks
in; the docstring states which paid calls the path is ALLOWED to make.

Run from backend/:  pip install -r tests/requirements-test.txt
                    python -m pytest tests/ -q
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update

import analytics
import main as app_main
import semantic_cache
from conftest import (
    Q_BASE, Q_OTHER, Q_PARA, Q_PY, Q_SKILLS, Q_SKILLS_PARA,
    VECTORS, chat, drain_writes,
)


def _reset_counters(env):
    env.embeddings.query_calls = 0
    env.embeddings.queries.clear()
    env.gpt.calls = 0
    env.vdb.by_vector_calls.clear()
    env.vdb.by_text_calls.clear()
    env.bm25_calls.clear()


async def _seed(env, question=Q_BASE) -> str:
    """Cache-miss a question through the full pipeline so its answer (and
    embedding) land in the cache, then zero the counters."""
    answer = await chat(env, question)
    assert env.gpt.calls == 1, "seeding must be a real generation"
    _reset_counters(env)
    return answer


# ---------------------------------------------------------------------------
# Level 1 — exact cache
# ---------------------------------------------------------------------------

async def test_normalization_still_unifies_trivial_variants():
    assert (analytics.normalize_question("  What are his SKILLS??")
            == analytics.normalize_question("what are his skills"))


async def test_exact_hit_costs_absolutely_nothing(env):
    """Exact hit: 0 embeddings, 0 GPT, 0 FAISS, 0 BM25 — and it must not
    consume the daily budget (it runs above the budget check)."""
    seeded = await _seed(env)
    budget_before = app_main._daily_usage["count"]

    answer = await chat(env, "what projects has aditya BUILT??")

    assert answer == seeded
    assert env.embeddings.query_calls == 0
    assert env.gpt.calls == 0
    assert env.vdb.by_vector_calls == [] and env.vdb.by_text_calls == []
    assert env.bm25_calls == []
    assert app_main._daily_usage["count"] == budget_before

    async with analytics._Session() as s:
        row = (await s.execute(
            select(analytics.ChatTurn)
            .order_by(analytics.ChatTurn.id.desc()))).scalars().first()
    assert row.status == "cache_hit"


# ---------------------------------------------------------------------------
# Level 2 — semantic cache
# ---------------------------------------------------------------------------

async def test_semantic_paraphrase_hit_costs_one_embedding_only(env):
    """Paraphrase above threshold (0.95 >= 0.92): 1 embedding, 0 GPT,
    0 document retrieval. Answer is the cached one; the hit is logged as
    semantic_hit with its similarity."""
    seeded = await _seed(env)

    answer = await chat(env, Q_PARA)

    assert answer == seeded
    assert env.embeddings.query_calls == 1
    assert env.gpt.calls == 0
    assert env.vdb.by_vector_calls == [] and env.vdb.by_text_calls == []
    assert env.bm25_calls == []

    async with analytics._Session() as s:
        row = (await s.execute(
            select(analytics.ChatTurn)
            .where(analytics.ChatTurn.status == "semantic_hit"))
        ).scalars().first()
    assert row is not None
    assert abs(row.cache_similarity - 0.95) < 1e-3


async def test_skills_paraphrase_pair_from_spec(env):
    """The spec's second pair: 'main AI skills' vs 'AI and ML technologies'
    (cos 0.94) — reused only because it clears the threshold."""
    seeded = await _seed(env, Q_SKILLS)
    answer = await chat(env, Q_SKILLS_PARA)
    assert answer == seeded
    assert env.gpt.calls == 0


async def test_different_question_is_a_miss(env):
    """Orthogonal question: semantic lookup happens (1 embedding) but the
    request proceeds to real RAG + generation."""
    await _seed(env)

    await chat(env, Q_OTHER)

    assert env.embeddings.query_calls == 1
    assert env.gpt.calls == 1
    assert len(env.vdb.by_vector_calls) == 1


async def test_similar_but_materially_different_is_conservative(env):
    """'...using Python' sits at cos 0.90 — BELOW the 0.92 threshold, so no
    reuse: the qualifier changes what a correct answer contains."""
    await _seed(env)

    await chat(env, Q_PY)

    assert env.gpt.calls == 1, "must regenerate, not reuse"


async def test_miss_generates_exactly_one_embedding_and_reuses_it(env):
    """THE performance invariant: a cache-miss request embeds the query
    ONCE, and document FAISS receives that IDENTICAL vector (by-vector API).
    The embed-inside-search text API is never touched."""
    _reset_counters(env)

    await chat(env, Q_OTHER)

    assert env.embeddings.query_calls == 1
    assert len(env.vdb.by_vector_calls) == 1
    assert env.vdb.by_vector_calls[0] == VECTORS[Q_OTHER]
    assert env.vdb.by_text_calls == []
    # BM25 keeps getting the raw string, not the vector.
    assert env.bm25_calls == [Q_OTHER]


async def test_embed_lru_dedupes_repeat_paraphrase_embeddings(env):
    """Same normalized question, different raw casing/punctuation -> one
    underlying embedding call thanks to the in-process LRU."""
    v1 = await app_main.embed_query_cached("What are his skills?")
    v2 = await app_main.embed_query_cached("what are his SKILLS??")
    assert v1 == v2
    assert env.embeddings.query_calls == 1


# ---------------------------------------------------------------------------
# Freshness: TTL + knowledge version
# ---------------------------------------------------------------------------

async def test_expired_entry_is_a_miss_and_gets_removed(env):
    """similarity high + expired = MISS. The stale row is lazily deleted and
    its vector leaves the index, so it can't win the next lookup either."""
    await _seed(env)
    async with analytics._Session() as s:   # age the row past the TTL
        await s.execute(update(analytics.CachedAnswer).values(
            created_at=datetime.now(timezone.utc)
            - timedelta(days=analytics.CACHE_TTL_DAYS + 1)))
        await s.commit()

    await chat(env, Q_PARA)                 # cos 0.95 — would hit if fresh

    assert env.gpt.calls == 1, "expired entry must not be served"
    await drain_writes()
    async with analytics._Session() as s:
        stale = (await s.execute(
            select(analytics.CachedAnswer)
            .where(analytics.CachedAnswer.question_norm
                   == analytics.normalize_question(Q_BASE)))
                 ).scalar_one_or_none()
    assert stale is None, "expired row should be lazily deleted"


async def test_knowledge_version_bump_invalidates_both_levels(env):
    """After a rebuild mints a new version, versioned rows from the old
    knowledge state serve NEITHER exact nor semantic hits."""
    await _seed(env)

    analytics.bump_knowledge_version_bg()
    await drain_writes()

    await chat(env, Q_BASE)      # exact wording — still must regenerate
    assert env.gpt.calls == 1
    _reset_counters(env)
    await chat(env, Q_PY)        # and unrelated-enough wording too
    assert env.gpt.calls == 1


async def test_legacy_rows_without_version_still_serve_exact_hits(env):
    """Backward compatibility: rows written BEFORE this feature (embedding
    and knowledge_version both NULL) keep serving exact hits, and their
    absence from the semantic index is silent."""
    async with analytics._Session() as s:
        s.add(analytics.CachedAnswer(
            question_norm=analytics.normalize_question(Q_BASE),
            question_raw=Q_BASE,
            answer="Legacy cached answer from before the semantic cache."))
        await s.commit()
    _reset_counters(env)

    answer = await chat(env, Q_BASE)

    assert answer.startswith("Legacy cached answer")
    assert env.embeddings.query_calls == 0 and env.gpt.calls == 0


# ---------------------------------------------------------------------------
# Invalidation: clear / hide / dangling vectors
# ---------------------------------------------------------------------------

async def test_cache_clear_also_resets_the_index(env):
    await _seed(env)
    assert semantic_cache.cache_index.size() == 1

    cleared = await analytics.cache_clear()

    assert cleared == 1
    assert semantic_cache.cache_index.size() == 0
    await chat(env, Q_PARA)
    assert env.gpt.calls == 1, "nothing cached => must regenerate"


async def test_hidden_project_cannot_resurface_via_semantic_hit(env):
    """The privacy sequence /admin/hide queues: rebuild (-> version bump) +
    full cache clear. Afterwards a semantically identical question must be
    answered FRESH — the old answer (which could mention the now-hidden
    repo) must be unreachable through either cache level."""
    old_answer = await _seed(env)

    # What admin_hide()'s background tasks do to the caches (the GitHub
    # fetch + re-embedding parts of _rebuild_index are out of scope here):
    analytics.bump_knowledge_version_bg()
    await analytics.cache_clear()
    await drain_writes()
    assert semantic_cache.cache_index.size() == 0

    env.gpt.answer = "Fresh answer without the hidden project."
    answer = await chat(env, Q_PARA)

    assert env.gpt.calls == 1
    assert answer != old_answer


async def test_dangling_vector_self_heals(env):
    """A vector whose DB row vanished (e.g. clear raced a write) nominates a
    dead id: the request falls through to real RAG, and the vector is
    removed so it can't keep winning lookups."""
    await semantic_cache.cache_index.add(999, VECTORS[Q_BASE])
    _reset_counters(env)

    await chat(env, Q_PARA)

    assert env.gpt.calls == 1
    await drain_writes()
    assert semantic_cache.cache_index.size() == 1, \
        "dead vector removed; this request's own write re-added one"


# ---------------------------------------------------------------------------
# Ambiguity margin (optional gate)
# ---------------------------------------------------------------------------

async def test_min_margin_rejects_near_ties(env, monkeypatch):
    """Two cached questions score 0.95 and ~0.93 for the same query: with
    MIN_MARGIN=0.05 the 0.02 gap is ambiguous -> miss; with the default 0.0
    the same lookup hits."""
    import math
    q2 = "What has Aditya been building lately?"
    VECTORS[q2] = [math.cos(0.6934), math.sin(0.6934),
                   0.0]  # ~cos 0.93 to Q_PARA
    await _seed(env)                      # Q_BASE: cos 0.95 to Q_PARA
    await _seed(env, q2)                  # runner-up

    monkeypatch.setattr(app_main, "SEMANTIC_CACHE_MIN_MARGIN", 0.05)
    await chat(env, Q_PARA)
    assert env.gpt.calls == 1, "ambiguous near-tie must miss"

    _reset_counters(env)
    monkeypatch.setattr(app_main, "SEMANTIC_CACHE_MIN_MARGIN", 0.0)
    await chat(env, Q_PARA)
    assert env.gpt.calls == 0, "margin off -> plain threshold hit"


# ---------------------------------------------------------------------------
# Size cap / eviction
# ---------------------------------------------------------------------------

async def test_eviction_drops_lowest_value_entries_first(env, monkeypatch):
    monkeypatch.setattr(analytics, "SEMANTIC_CACHE_MAX_ENTRIES", 2)
    await _seed(env, Q_BASE)
    await _seed(env, Q_SKILLS)
    # Make Q_BASE clearly valuable; Q_SKILLS stays at 0 hits.
    for _ in range(3):
        await chat(env, Q_BASE)

    await _seed(env, Q_OTHER)             # 3rd embedded row -> cap enforced
    await drain_writes()

    async with analytics._Session() as s:
        kept = {r for (r,) in (await s.execute(
            select(analytics.CachedAnswer.question_raw)
            .where(analytics.CachedAnswer.embedding.is_not(None)))).all()}
    assert kept == {Q_BASE, Q_OTHER}, "least-hit row (Q_SKILLS) evicted"
    assert semantic_cache.cache_index.size() == 2


# ---------------------------------------------------------------------------
# Feature flag rollback
# ---------------------------------------------------------------------------

async def test_disabled_flag_restores_legacy_pipeline(env, monkeypatch):
    """SEMANTIC_CACHE_ENABLED=0: no query embedding, dense retrieval embeds
    internally (text API), rows are written WITHOUT vectors, index unused."""
    monkeypatch.setattr(app_main, "SEMANTIC_CACHE_ENABLED", False)

    await chat(env, Q_BASE)

    assert env.embeddings.query_calls == 0
    assert env.vdb.by_text_calls == [Q_BASE]
    assert env.vdb.by_vector_calls == []
    assert semantic_cache.cache_index.size() == 0
    async with analytics._Session() as s:
        row = (await s.execute(select(analytics.CachedAnswer))).scalars().first()
    assert row is not None and row.embedding is None


# ---------------------------------------------------------------------------
# Startup rehydration
# ---------------------------------------------------------------------------

async def test_startup_load_restores_index_without_api_calls(env):
    """Simulated restart: wipe the in-memory index, reload from DB rows —
    vector count restored, zero embedding calls, and lookups hit again."""
    await _seed(env)
    await semantic_cache.cache_index.reset()
    assert semantic_cache.cache_index.size() == 0

    entries = await analytics.load_semantic_entries(app_main.EMBEDDING_MODEL)
    n = await semantic_cache.cache_index.load(entries)

    assert n == 1
    assert env.embeddings.query_calls == 0, "rehydration is DB-only"
    await chat(env, Q_PARA)
    assert env.gpt.calls == 0, "reloaded vector serves the paraphrase"


async def test_old_schema_database_migrates_in_place(tmp_path):
    """A pre-feature analytics.db (no embedding columns, no app_meta) must
    open, gain the new nullable columns, and keep serving its legacy rows —
    the 'existing analytics.db must continue opening' guarantee."""
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE answer_cache (id INTEGER PRIMARY KEY,
      question_norm VARCHAR(500) UNIQUE, question_raw TEXT, answer TEXT,
      created_at DATETIME, hits INTEGER, last_hit_at DATETIME);
    INSERT INTO answer_cache
      (question_norm, question_raw, answer, created_at, hits)
    VALUES ('what are his skills', 'What are his skills?',
            'A legacy answer, long enough to be a realistic cache row.',
            datetime('now'), 3);
    """)
    con.commit()
    con.close()

    old_url = analytics.DATABASE_URL
    analytics.DATABASE_URL = f"sqlite+aiosqlite:///{path}"
    try:
        await analytics.startup()
        assert analytics._Session is not None, "startup must not disable"
        got = await analytics.cache_get("What are his SKILLS??")
        assert got and got.startswith("A legacy answer")
        assert await analytics.load_semantic_entries(
            app_main.EMBEDDING_MODEL) == []
        cols = {c[1] for c in sqlite3.connect(path).execute(
            "PRAGMA table_info(answer_cache)").fetchall()}
        assert {"embedding", "embedding_model", "embedding_dim",
                "knowledge_version"} <= cols
    finally:
        await analytics.shutdown()
        analytics._engine, analytics._Session = None, None
        analytics.DATABASE_URL = old_url


async def test_startup_load_skips_other_models_vectors(env):
    await _seed(env)
    async with analytics._Session() as s:   # pretend an old model wrote it
        await s.execute(update(analytics.CachedAnswer)
                        .values(embedding_model="text-embedding-ancient"))
        await s.commit()

    entries = await analytics.load_semantic_entries(app_main.EMBEDDING_MODEL)
    assert entries == [], "cross-model vectors must never enter the index"


async def test_near_miss_similarity_is_logged_on_the_generated_row(env):
    """OBSERVABILITY INVARIANT: a below-threshold miss must record HOW CLOSE
    it came, not just that it missed.

    Q_PY sits at cos 0.90 to the seeded question — under the 0.92 gate, so
    the request pays for a generation. Before this was logged, that row and
    a row for a totally unrelated question looked identical (both NULL), so
    the analytics table could not answer "is my threshold too high?". The
    0.90 here is what lets you sort misses by proximity and retune with
    evidence instead of guessing."""
    await _seed(env)

    await chat(env, Q_PY)

    async with analytics._Session() as s:
        row = (await s.execute(
            select(analytics.ChatTurn)
            .where(analytics.ChatTurn.question == Q_PY))
        ).scalars().first()

    assert row is not None
    assert row.status == "ok", "0.90 is under the gate, so this must regenerate"
    assert row.cache_similarity is not None, "a miss must still record the top score"
    assert abs(row.cache_similarity - 0.90) < 1e-3


async def test_unrelated_question_logs_a_low_similarity_not_null(env):
    """The other half of the same invariant: an orthogonal question logs a
    NEAR-ZERO similarity, which is what distinguishes 'compared and nowhere
    close' from 'never compared at all' (feature off / embed failed -> NULL)."""
    await _seed(env)

    await chat(env, Q_OTHER)

    async with analytics._Session() as s:
        row = (await s.execute(
            select(analytics.ChatTurn)
            .where(analytics.ChatTurn.question == Q_OTHER))
        ).scalars().first()

    assert row is not None
    assert row.cache_similarity is not None
    assert row.cache_similarity < 0.5
