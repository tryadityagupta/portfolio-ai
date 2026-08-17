"""
analytics.py — conversation log + answer cache for the portfolio chatbot.

WHAT THIS FILE GIVES YOU (plain English):

  1. A LOG. Every question and every answer becomes one row in a table
     called `chat_turns`. That's your record of who asked what.

  2. A CACHE. Every successful answer is also saved in a table called
     `answer_cache`, keyed by the question. If the same question comes in
     again (ignoring case/punctuation/extra spaces), the saved answer is
     replayed and OpenAI is never called. Faster for the visitor, free for
     you, and it doesn't spend your daily request budget.

WHERE THE DATA LIVES:

  If the environment variable DATABASE_URL is not set (your laptop), both
  tables live in ONE FILE named `analytics.db`, created in the folder you
  start uvicorn from — for you that's `backend/analytics.db`. It's just a
  file; you can open it, copy it, or delete it to start fresh.

  When you later deploy, set DATABASE_URL to a hosted Postgres URL and this
  module switches automatically. Nothing else changes.

DESIGN RULES this module follows:

  - Logging must NEVER break /chat. Every database call is wrapped; on any
    failure it prints a warning and the chatbot carries on.
  - Writes are fire-and-forget (asyncio.create_task) from the END of the
    SSE stream, so they add zero latency to the user's answer.
  - log_turn_bg() / cache_put_bg() are deliberately SYNCHRONOUS so they can
    be called from a `finally:` block inside an async generator — including
    when the visitor closes the tab mid-answer.
  - The `_pending` set holds strong references to in-flight write tasks.
    asyncio only keeps weak references, so without this the garbage
    collector can kill an INSERT before it lands. Keep it.

Env vars (all optional on your laptop):
    DATABASE_URL            Postgres URL when deployed. Default: local SQLite.
    IP_HASH_SALT            Random text; lets you count repeat visitors
                            without storing real IP addresses.
    CACHE_TTL_DAYS          How long a cached answer stays valid. Default 7.
    ANALYTICS_RETENTION_DAYS  How long log rows are kept by purge_old(). Default 365.
    ANALYTICS_OFF=1         Kill switch: disables all logging and caching.
"""

import asyncio
import hashlib
import os
import re
import secrets
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import (
    JSON, DateTime, Float, Index, Integer, LargeBinary, String, Text,
    delete, desc, func, select,
)
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

import semantic_cache   # in-memory FAISS twin of the embedded cache rows

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./analytics.db")
IP_HASH_SALT = os.getenv("IP_HASH_SALT", "")
ANALYTICS_OFF = os.getenv("ANALYTICS_OFF") == "1"

MAX_QUESTION = 2_000          # hard caps so one abusive request
MAX_ANSWER = 8_000            # can't write a 2 MB row
RETENTION_DAYS = int(os.getenv("ANALYTICS_RETENTION_DAYS", "365"))

CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "7"))
CACHE_MIN_ANSWER = 20         # don't cache empty / one-word junk

# Cap on rows that carry an embedding (== vectors in the in-memory FAISS
# index). When exceeded, the LOWEST-value rows go first: fewest hits, then
# least-recently hit, then oldest — an LFU-then-LRU blend built from columns
# the cache already tracks. Rows WITHOUT embeddings (pre-feature legacy) are
# untouched here; the TTL retires them.
SEMANTIC_CACHE_MAX_ENTRIES = int(
    os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "500"))


def _normalise_url(raw: str) -> tuple[str, dict]:
    """Turn whatever the host hands you into an async SQLAlchemy URL.

    Hosted Postgres (Neon/Render) gives `postgresql://...?sslmode=require`.
    SQLAlchemy needs the async driver (postgresql+asyncpg://) and asyncpg
    does NOT understand sslmode/channel_binding query params — they must be
    stripped and ssl passed via connect_args instead."""
    connect_args: dict = {}

    if raw.startswith("sqlite"):
        if "+aiosqlite" not in raw:
            raw = raw.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return raw, connect_args

    if raw.startswith("postgres://"):            # legacy scheme some hosts use
        raw = raw.replace("postgres://", "postgresql://", 1)
    if raw.startswith("postgresql://"):
        raw = raw.replace("postgresql://", "postgresql+asyncpg://", 1)

    if "+asyncpg" in raw:
        parts = urlsplit(raw)
        kept = [(k, v) for k, v in parse_qsl(parts.query)
                if k not in ("sslmode", "channel_binding")]
        raw = urlunsplit(parts._replace(query=urlencode(kept)))
        connect_args["ssl"] = True

    return raw, connect_args


# --------------------------------------------------------------------------
# SCHEMA — two tables, one database
# --------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class ChatTurn(Base):
    """THE LOG: one question + one answer per row."""

    __tablename__ = "chat_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # --- who (pseudonymous) -------------------------------------------------
    visitor_id: Mapped[str | None] = mapped_column(
        String(64))   # from browser localStorage
    session_id: Mapped[str | None] = mapped_column(
        String(64))   # from browser sessionStorage
    ip_hash: Mapped[str | None] = mapped_column(
        String(64))      # sha256(ip+salt), not the IP
    country: Mapped[str | None] = mapped_column(String(4))
    city: Mapped[str | None] = mapped_column(String(80))
    user_agent: Mapped[str | None] = mapped_column(String(400))

    # --- how they got here --------------------------------------------------
    referrer: Mapped[str | None] = mapped_column(String(400))
    page_url: Mapped[str | None] = mapped_column(String(400))
    utm_source: Mapped[str | None] = mapped_column(String(80))

    # --- what was said -------------------------------------------------------
    question: Mapped[str | None] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text)
    sources: Mapped[list | None] = mapped_column(JSON)   # repos used to answer

    # --- how it went ----------------------------------------------------------
    status: Mapped[str] = mapped_column(String(16), default="ok")
    # "ok" | "cache_hit" | "semantic_hit" | "error" | "aborted"
    # | "over_budget" | "not_ready"
    # ("cache_hit" stays the EXACT-match status so old rows and any dashboards
    #  built on it keep meaning the same thing; "semantic_hit" is new and fits
    #  the existing String(16) column, unlike "cache_semantic_hit".)
    model: Mapped[str | None] = mapped_column(String(60))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    ttft_ms: Mapped[int | None] = mapped_column(
        Integer)   # time to first token
    total_ms: Mapped[int | None] = mapped_column(Integer)
    cache_similarity: Mapped[float | None] = mapped_column(
        Float)   # cosine score, set only on status == "semantic_hit" rows —
    # lets /admin/analytics show the observed similarity distribution, which
    # is the evidence for tuning SEMANTIC_CACHE_THRESHOLD later.

    __table_args__ = (
        Index("ix_turns_created_at", "created_at"),
        Index("ix_turns_visitor", "visitor_id"),
        Index("ix_turns_session", "session_id"),
    )


class CachedAnswer(Base):
    """THE CACHE: one saved answer per distinct (normalized) question.

    Since the semantic-cache feature, a row MAY also carry the embedding of
    its question. Rows written before the feature have embedding = NULL and
    keep working exactly as before (exact-match only); they simply don't
    participate in semantic lookups and retire via the normal TTL. No
    backfill job re-embeds them — that would mean surprise OpenAI spend at
    startup for rows that expire within CACHE_TTL_DAYS anyway."""

    __tablename__ = "answer_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_norm: Mapped[str] = mapped_column(
        String(500), unique=True, index=True)
    question_raw: Mapped[str | None] = mapped_column(
        Text)   # first wording seen
    answer: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    hits: Mapped[int] = mapped_column(Integer, default=0)
    last_hit_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True))

    # --- semantic-cache extension (all nullable => additive migration) ----
    embedding: Mapped[bytes | None] = mapped_column(
        LargeBinary)   # float32 LE bytes (semantic_cache.to_bytes): ~6 KB
    # per 1536-dim vector vs ~30 KB as a JSON float list. BLOB on SQLite,
    # BYTEA on Postgres — both native.
    embedding_model: Mapped[str | None] = mapped_column(String(60))
    embedding_dim: Mapped[int | None] = mapped_column(Integer)
    knowledge_version: Mapped[str | None] = mapped_column(String(64))
    # ^ which knowledge-base build produced this answer. NULL = legacy row
    # from before versioning existed: still valid for EXACT hits (unchanged
    # behavior), excluded from the semantic index (it has no embedding).


class AppMeta(Base):
    """One-row-per-key settings store. Currently holds only the knowledge
    version — a random token regenerated whenever the RAG index is rebuilt,
    so cached answers can be pinned to the knowledge state that produced
    them. Deliberately not a 'versioning system': one value, one writer."""

    __tablename__ = "app_meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str | None] = mapped_column(String(128))


_KNOWLEDGE_VERSION_KEY = "knowledge_version"
_knowledge_version: str | None = None
_loop: asyncio.AbstractEventLoop | None = None   # captured in startup(); lets
# bump_knowledge_version_bg() work from BackgroundTasks worker threads too.


_WORD_STRIP = re.compile(r"[^\w\s]")   # drop punctuation, keep letters/digits
_WS = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """'  What are his SKILLS??' and 'what are his skills' become the same
    cache key. Lowercase, remove punctuation, collapse whitespace. This is
    exact-match after cleanup — it does NOT match paraphrases (on purpose:
    serving a wrong cached answer is worse than paying for a fresh one)."""
    t = _WORD_STRIP.sub(" ", (text or "").lower())
    return _WS.sub(" ", t).strip()[:500]


# --------------------------------------------------------------------------
# ENGINE LIFECYCLE
# --------------------------------------------------------------------------

_engine = None
_Session: async_sessionmaker[AsyncSession] | None = None
_pending: set[asyncio.Task] = set()


async def startup() -> None:
    """Call once from the FastAPI lifespan. If the DB can't be reached it
    disables itself with a printed warning instead of crashing the app."""
    global _engine, _Session, _loop

    if ANALYTICS_OFF:
        print("[analytics] disabled via ANALYTICS_OFF", file=sys.stderr)
        return

    url, connect_args = _normalise_url(DATABASE_URL)
    try:
        _loop = asyncio.get_running_loop()
        _engine = create_async_engine(
            url,
            connect_args=connect_args,
            pool_size=2,
            max_overflow=0,
            pool_pre_ping=True,     # hosted Postgres kills idle connections
            pool_recycle=280,
        )
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # create_all builds MISSING TABLES but never alters existing
            # ones, so an analytics.db from before the semantic cache would
            # lack the new columns and every SELECT would fail. This adds
            # exactly the missing nullable columns — additive, idempotent,
            # works on SQLite and Postgres.
            await conn.run_sync(_migrate_add_missing_columns)
        _Session = async_sessionmaker(_engine, expire_on_commit=False)
        await _load_or_init_knowledge_version()
        print(
            f"[analytics] ready ({url.split('@')[-1][:40]})", file=sys.stderr)
    except Exception as e:
        _engine, _Session = None, None
        print(f"[analytics] DISABLED, init failed: {e!r}", file=sys.stderr)


def _migrate_add_missing_columns(conn) -> None:
    """ALTER TABLE ... ADD COLUMN for any model column absent from the live
    table. Only ever ADDS nullable columns — never drops, renames or
    rewrites — so it is safe to run on every boot against existing data."""
    insp = sa_inspect(conn)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue                      # create_all just made it; complete
        existing = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing:
                continue
            ddl_type = col.type.compile(dialect=conn.dialect)
            conn.exec_driver_sql(
                f"ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl_type}")
            print(f"[analytics] migrated: {table.name}.{col.name} "
                  f"({ddl_type})", file=sys.stderr)


# --------------------------------------------------------------------------
# KNOWLEDGE VERSION — pins cached answers to the knowledge state that
# produced them. main._rebuild_index() bumps it; both cache read paths
# refuse rows stamped with any OTHER (non-NULL) version.
# --------------------------------------------------------------------------

def get_knowledge_version() -> str | None:
    return _knowledge_version


async def _load_or_init_knowledge_version() -> None:
    global _knowledge_version
    async with _Session() as s:
        row = await s.get(AppMeta, _KNOWLEDGE_VERSION_KEY)
        if row is None or not row.value:
            row = AppMeta(key=_KNOWLEDGE_VERSION_KEY,
                          value=secrets.token_hex(8))
            await s.merge(row)   # merge: races with a twin worker are benign
            await s.commit()
        _knowledge_version = row.value


async def _persist_knowledge_version(version: str) -> None:
    try:
        async with _Session() as s:
            await s.merge(AppMeta(key=_KNOWLEDGE_VERSION_KEY, value=version))
            await s.commit()
    except Exception as e:
        print(f"[analytics] version persist failed: {e!r}", file=sys.stderr)


def bump_knowledge_version_bg() -> str | None:
    """Mint a NEW version. The in-memory copy flips immediately — so /chat
    stops trusting old-version cache rows the instant the knowledge base
    changes — and the DB write is fire-and-forget. Callable from the event
    loop OR from a BackgroundTasks worker thread (where _rebuild_index runs):
    the thread case schedules the write onto the captured startup loop."""
    global _knowledge_version
    if _Session is None:
        return None
    new = secrets.token_hex(8)
    _knowledge_version = new
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    try:
        if loop is not None:                      # already on the app loop
            task = loop.create_task(_persist_knowledge_version(new))
            _pending.add(task)
            task.add_done_callback(_pending.discard)
        elif _loop is not None:                   # worker thread → app loop
            asyncio.run_coroutine_threadsafe(
                _persist_knowledge_version(new), _loop)
    except Exception as e:
        print(f"[analytics] version bump schedule failed: {e!r}",
              file=sys.stderr)
    return new


async def shutdown() -> None:
    """Give in-flight writes a moment to land, then close the pool."""
    if _pending:
        await asyncio.wait(set(_pending), timeout=3.0)
    if _engine is not None:
        await _engine.dispose()


def hash_ip(ip: str | None) -> str | None:
    """Store a fingerprint, not an address: unique-visitor counts without
    keeping anything that identifies a person."""
    if not ip:
        return None
    return hashlib.sha256(f"{IP_HASH_SALT}:{ip}".encode()).hexdigest()[:32]


# --------------------------------------------------------------------------
# WRITE PATH — the log
# --------------------------------------------------------------------------

def log_turn_bg(**fields) -> None:
    """Fire-and-forget. Synchronous on purpose — callable from `finally:`
    inside an async generator, including on client disconnect."""
    if _Session is None:
        return
    try:
        task = asyncio.create_task(_write_turn(fields))
        _pending.add(task)                       # strong ref, see docstring
        task.add_done_callback(_pending.discard)
    except RuntimeError:
        pass                                     # no running loop


async def _write_turn(fields: dict) -> None:
    try:
        if fields.get("question"):
            fields["question"] = fields["question"][:MAX_QUESTION]
        if fields.get("answer"):
            fields["answer"] = fields["answer"][:MAX_ANSWER]
        if fields.get("user_agent"):
            fields["user_agent"] = fields["user_agent"][:400]

        async with _Session() as s:
            s.add(ChatTurn(**fields))
            await s.commit()
    except Exception as e:
        print(f"[analytics] write failed: {e!r}", file=sys.stderr)


# --------------------------------------------------------------------------
# THE CACHE
# --------------------------------------------------------------------------

def _row_unusable(row: CachedAnswer) -> bool:
    """A cached row may be served only if it is (a) inside the TTL and
    (b) not stamped with a DIFFERENT knowledge version. Applies identically
    to exact and semantic hits — high cosine similarity never overrides
    freshness. NULL version = legacy row = exempt from (b), which keeps
    pre-feature caches behaving exactly as before this change."""
    created = row.created_at
    if created is not None and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)  # SQLite quirk
    expired = (created is None or
               created < datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS))
    current = _knowledge_version
    stale = (row.knowledge_version is not None and current is not None
             and row.knowledge_version != current)
    return expired or stale


async def _drop_row(s: AsyncSession, row: CachedAnswer) -> None:
    """Lazy deletion of an unusable row + its in-memory vector, so the FAISS
    index never keeps pointing at a database row that no longer exists."""
    rid = row.id
    await s.delete(row)
    await s.commit()
    await semantic_cache.cache_index.remove(rid)


async def cache_get(question: str) -> str | None:
    """Return a saved answer for this question, or None.
    Called INLINE at the top of /chat (we must know the result before
    deciding whether to call OpenAI). One indexed read — sub-millisecond
    on SQLite. Also counts the hit and enforces TTL + knowledge version."""
    if _Session is None:
        return None
    qn = normalize_question(question)
    if not qn:
        return None
    try:
        async with _Session() as s:
            row = (await s.execute(
                select(CachedAnswer).where(CachedAnswer.question_norm == qn)
            )).scalar_one_or_none()
            if row is None:
                return None
            if _row_unusable(row):
                await _drop_row(s, row)
                return None

            row.hits += 1
            row.last_hit_at = datetime.now(timezone.utc)
            await s.commit()
            return row.answer
    except Exception as e:
        print(f"[analytics] cache_get failed: {e!r}", file=sys.stderr)
        return None


async def cache_get_by_id(cache_id: int) -> str | None:
    """The semantic-cache follow-up read: the FAISS index said 'row N looks
    like this question' — this validates row N is still servable (exists,
    unexpired, current knowledge version) and counts the hit. Returning None
    tells the caller to treat it as a miss; the caller also removes the
    dangling vector so the same stale id can't win the next lookup."""
    if _Session is None:
        return None
    try:
        async with _Session() as s:
            row = await s.get(CachedAnswer, cache_id)
            if row is None:
                return None
            if _row_unusable(row):
                await _drop_row(s, row)
                return None

            row.hits += 1
            row.last_hit_at = datetime.now(timezone.utc)
            await s.commit()
            return row.answer
    except Exception as e:
        print(f"[analytics] cache_get_by_id failed: {e!r}", file=sys.stderr)
        return None


def cache_put_bg(question: str, answer: str,
                 embedding: list[float] | None = None,
                 embedding_model: str | None = None,
                 knowledge_version: str | None = None) -> None:
    """Save an answer for next time. Fire-and-forget, called from the
    stream's finally block — only for status == 'ok' answers.

    `embedding` is the SAME vector /chat already generated for this request
    (semantic lookup + FAISS retrieval reused it) — persisting it here is
    free. This function never calls an embedding API; if no vector is passed
    (semantic cache disabled, or the embed call failed), the row is written
    without one and serves exact-match hits only, i.e. the pre-feature
    behavior."""
    if _Session is None:
        return
    if not answer or len(answer) < CACHE_MIN_ANSWER:
        return
    qn = normalize_question(question)
    if not qn:
        return
    try:
        task = asyncio.create_task(
            _cache_write(qn, question[:MAX_QUESTION], answer[:MAX_ANSWER],
                         embedding, embedding_model, knowledge_version))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
    except RuntimeError:
        pass


async def _cache_write(qn: str, qraw: str, answer: str,
                       embedding: list[float] | None,
                       embedding_model: str | None,
                       knowledge_version: str | None) -> None:
    try:
        emb_bytes = (semantic_cache.to_bytes(embedding)
                     if embedding is not None else None)
        emb_dim = len(embedding) if embedding is not None else None

        async with _Session() as s:
            existing = (await s.execute(
                select(CachedAnswer).where(CachedAnswer.question_norm == qn)
            )).scalar_one_or_none()

            if existing is not None:
                # First answer wins until TTL expiry / clear. One exception:
                # two CONCURRENT misses on the same normalized question both
                # generate, the loser lands here — if the winner has no
                # embedding yet, attach ours (same normalized question ⇒
                # same vector semantics). Heals the row into the semantic
                # index instead of discarding a vector we already paid for.
                if (existing.embedding is None and emb_bytes is not None
                        and not _row_unusable(existing)):
                    existing.embedding = emb_bytes
                    existing.embedding_model = embedding_model
                    existing.embedding_dim = emb_dim
                    existing.knowledge_version = (existing.knowledge_version
                                                  or knowledge_version)
                    await s.commit()
                    await semantic_cache.cache_index.add(existing.id, embedding)
                return

            row = CachedAnswer(
                question_norm=qn, question_raw=qraw, answer=answer,
                embedding=emb_bytes, embedding_model=embedding_model,
                embedding_dim=emb_dim, knowledge_version=knowledge_version)
            s.add(row)
            await s.commit()   # commit assigns row.id — needed as FAISS id
            new_id = row.id

        if emb_bytes is not None:
            # A bump between generation and this write means the row is
            # already stale — leave it out of the index (reads would reject
            # it anyway; this just avoids pointless lookup work).
            if knowledge_version == _knowledge_version:
                await semantic_cache.cache_index.add(new_id, embedding)
            await _evict_over_limit()
    except Exception as e:
        # Includes the harmless race where two identical questions arrive at
        # once and both try to insert — the unique index rejects the second.
        print(f"[analytics] cache write failed: {e!r}", file=sys.stderr)


async def _evict_over_limit() -> None:
    """Keep at most SEMANTIC_CACHE_MAX_ENTRIES embedded rows. Runs inside
    the background write task, never on a user's request path. Eviction
    order = lowest value first: fewest hits, then least-recently used
    (never-hit rows first), then oldest."""
    try:
        async with _Session() as s:
            n = (await s.execute(
                select(func.count(CachedAnswer.id))
                .where(CachedAnswer.embedding.is_not(None))
            )).scalar_one()
            overflow = n - SEMANTIC_CACHE_MAX_ENTRIES
            if overflow <= 0:
                return
            victims = (await s.execute(
                select(CachedAnswer.id)
                .where(CachedAnswer.embedding.is_not(None))
                .order_by(CachedAnswer.hits.asc(),
                          CachedAnswer.last_hit_at.asc().nulls_first(),
                          CachedAnswer.created_at.asc())
                .limit(overflow)
            )).scalars().all()
            if not victims:
                return
            await s.execute(
                delete(CachedAnswer).where(CachedAnswer.id.in_(victims)))
            await s.commit()
        await semantic_cache.cache_index.remove_many(list(victims))
        print(f"[analytics] semantic cache evicted {len(victims)} "
              f"row(s) (cap {SEMANTIC_CACHE_MAX_ENTRIES})", file=sys.stderr)
    except Exception as e:
        print(f"[analytics] eviction failed: {e!r}", file=sys.stderr)


async def load_semantic_entries(embedding_model: str) -> list[tuple[int, bytes]]:
    """Startup feed for the in-memory index: (id, embedding bytes) of every
    row that is still servable AND was embedded with the model the app is
    running now. Pure database read — zero OpenAI calls, so boot cost stays
    flat no matter how many rows exist. Rows without embeddings are simply
    not semantic candidates (legacy rows; also rows written while the
    feature was disabled)."""
    if _Session is None:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)
    current = _knowledge_version
    q = (select(CachedAnswer.id, CachedAnswer.embedding)
         .where(CachedAnswer.embedding.is_not(None),
                CachedAnswer.embedding_model == embedding_model,
                CachedAnswer.created_at >= cutoff))
    if current is not None:
        q = q.where((CachedAnswer.knowledge_version.is_(None)) |
                    (CachedAnswer.knowledge_version == current))
    async with _Session() as s:
        rows = (await s.execute(q)).all()
    return [(rid, emb) for rid, emb in rows]


async def cache_clear() -> int:
    """Wipe the whole cache. Wired to /admin/cache-clear and to hide/unhide,
    because a cached answer written before you hid a project could still
    mention it. The in-memory semantic index is reset in the same call —
    THE privacy invariant of this feature: a vector must never outlive its
    row, or a hidden project could resurface through a similarity hit."""
    if _Session is None:
        return 0
    async with _Session() as s:
        res = await s.execute(delete(CachedAnswer))
        await s.commit()
    await semantic_cache.cache_index.reset()
    return res.rowcount or 0


async def cache_list(limit: int = 100) -> list[dict]:
    """What's cached right now, most-reused first. Exposes WHETHER a row has
    an embedding, never the vector itself — 1536 floats are noise in an
    admin table and don't belong in API responses."""
    if _Session is None:
        return []
    async with _Session() as s:
        rows = (await s.execute(
            select(CachedAnswer).order_by(desc(CachedAnswer.hits)).limit(limit)
        )).scalars().all()
    return [{
        "question": r.question_raw or r.question_norm,
        "hits": r.hits,
        "cached_at": r.created_at.isoformat() if r.created_at else None,
        "answer_preview": (r.answer or "")[:120],
        "embedding_available": r.embedding is not None,
        "knowledge_version": r.knowledge_version,
    } for r in rows]


# --------------------------------------------------------------------------
# READ PATH — powers the /admin endpoints
# --------------------------------------------------------------------------

async def summary(days: int = 30) -> dict:
    if _Session is None:
        return {"error": "analytics disabled"}

    since = datetime.now(timezone.utc) - timedelta(days=days)
    async with _Session() as s:
        totals = (await s.execute(
            select(
                func.count(ChatTurn.id),
                func.count(func.distinct(ChatTurn.visitor_id)),
                func.count(func.distinct(ChatTurn.session_id)),
                func.avg(ChatTurn.ttft_ms),
                func.sum(ChatTurn.completion_tokens),
            ).where(ChatTurn.created_at >= since)
        )).one()

        exact_hits = (await s.execute(
            select(func.count(ChatTurn.id))
            .where(ChatTurn.created_at >= since, ChatTurn.status == "cache_hit")
        )).scalar_one()

        semantic_hits, sim_avg, sim_min = (await s.execute(
            select(func.count(ChatTurn.id),
                   func.avg(ChatTurn.cache_similarity),
                   func.min(ChatTurn.cache_similarity))
            .where(ChatTurn.created_at >= since,
                   ChatTurn.status == "semantic_hit")
        )).one()

        # Answers that DID cost a GPT generation in this window — the number
        # the whole cache exists to shrink. "GPT calls avoided" is exactly
        # exact_hits + semantic_hits over the same window.
        generated = (await s.execute(
            select(func.count(ChatTurn.id))
            .where(ChatTurn.created_at >= since, ChatTurn.status == "ok")
        )).scalar_one()

        by_day = (await s.execute(
            select(func.date(ChatTurn.created_at).label("d"),
                   func.count(ChatTurn.id))
            .where(ChatTurn.created_at >= since)
            .group_by("d").order_by(desc("d"))
        )).all()

        by_ref = (await s.execute(
            select(ChatTurn.utm_source, ChatTurn.referrer,
                   func.count(ChatTurn.id).label("n"))
            .where(ChatTurn.created_at >= since)
            .group_by(ChatTurn.utm_source, ChatTurn.referrer)
            .order_by(desc("n")).limit(15)
        )).all()

        by_country = (await s.execute(
            select(ChatTurn.country, func.count(
                func.distinct(ChatTurn.visitor_id)).label("n"))
            .where(ChatTurn.created_at >= since)
            .group_by(ChatTurn.country).order_by(desc("n")).limit(15)
        )).all()

        top_q = (await s.execute(
            select(func.lower(ChatTurn.question),
                   func.count(ChatTurn.id).label("n"))
            .where(ChatTurn.created_at >= since)
            .group_by(func.lower(ChatTurn.question))
            .order_by(desc("n")).limit(20)
        )).all()

    total_cache_hits = exact_hits + semantic_hits
    answerable = total_cache_hits + generated   # excludes errors/over_budget

    return {
        "window_days": days,
        "turns": totals[0],
        # answered_from_cache keeps its old meaning (any cache hit) so
        # existing consumers of this endpoint don't shift under you.
        "answered_from_cache": total_cache_hits,
        "cache_exact_hits": exact_hits,
        "cache_semantic_hits": semantic_hits,
        "cache_hit_rate": (round(total_cache_hits / answerable, 3)
                           if answerable else None),
        "gpt_generations": generated,
        "semantic_similarity_avg": (round(float(sim_avg), 4)
                                    if sim_avg is not None else None),
        "semantic_similarity_min": (round(float(sim_min), 4)
                                    if sim_min is not None else None),
        "unique_visitors": totals[1],
        "sessions": totals[2],
        "avg_ttft_ms": round(totals[3]) if totals[3] else None,
        "completion_tokens": totals[4],
        "by_day": [{"date": str(d), "turns": n} for d, n in by_day],
        "traffic_sources": [
            {"utm_source": u, "referrer": r, "turns": n} for u, r, n in by_ref
        ],
        "by_country": [{"country": c, "visitors": n} for c, n in by_country],
        "top_questions": [{"question": q, "asked": n} for q, n in top_q],
    }


async def recent(limit: int = 50, visitor_id: str | None = None) -> list[dict]:
    """The transcript view — newest first. The single most useful readout:
    the questions people ask are what your portfolio failed to communicate."""
    if _Session is None:
        return []

    q = select(ChatTurn).order_by(
        desc(ChatTurn.created_at)).limit(min(limit, 200))
    if visitor_id:
        q = q.where(ChatTurn.visitor_id == visitor_id)

    async with _Session() as s:
        rows = (await s.execute(q)).scalars().all()

    return [{
        "at": r.created_at.isoformat() if r.created_at else None,
        "visitor_id": r.visitor_id,
        "session_id": r.session_id,
        "country": r.country,
        "city": r.city,
        "utm_source": r.utm_source,
        "referrer": r.referrer,
        "question": r.question,
        "answer": r.answer,
        "sources": r.sources,
        "status": r.status,
        "ttft_ms": r.ttft_ms,
        "total_ms": r.total_ms,
        "tokens": (r.prompt_tokens or 0) + (r.completion_tokens or 0),
    } for r in rows]


async def purge_old() -> int:
    """Retention: delete log rows older than ANALYTICS_RETENTION_DAYS."""
    if _Session is None:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    async with _Session() as s:
        res = await s.execute(delete(ChatTurn).where(ChatTurn.created_at < cutoff))
        await s.commit()
    return res.rowcount or 0
