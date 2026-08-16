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
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import (
    JSON, DateTime, Index, Integer, String, Text, delete, desc, func, select,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

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
    # "ok" | "cache_hit" | "error" | "aborted" | "over_budget" | "not_ready"
    model: Mapped[str | None] = mapped_column(String(60))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    ttft_ms: Mapped[int | None] = mapped_column(
        Integer)   # time to first token
    total_ms: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_turns_created_at", "created_at"),
        Index("ix_turns_visitor", "visitor_id"),
        Index("ix_turns_session", "session_id"),
    )


class CachedAnswer(Base):
    """THE CACHE: one saved answer per distinct (normalized) question."""

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
    global _engine, _Session

    if ANALYTICS_OFF:
        print("[analytics] disabled via ANALYTICS_OFF", file=sys.stderr)
        return

    url, connect_args = _normalise_url(DATABASE_URL)
    try:
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
        _Session = async_sessionmaker(_engine, expire_on_commit=False)
        print(
            f"[analytics] ready ({url.split('@')[-1][:40]})", file=sys.stderr)
    except Exception as e:
        _engine, _Session = None, None
        print(f"[analytics] DISABLED, init failed: {e!r}", file=sys.stderr)


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

async def cache_get(question: str) -> str | None:
    """Return a saved answer for this question, or None.
    Called INLINE at the top of /chat (we must know the result before
    deciding whether to call OpenAI). One indexed read — sub-millisecond
    on SQLite. Also counts the hit and enforces the TTL."""
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

            created = row.created_at
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)  # SQLite quirk
            expired = (created is None or
                       created < datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS))
            if expired:
                await s.delete(row)
                await s.commit()
                return None

            row.hits += 1
            row.last_hit_at = datetime.now(timezone.utc)
            await s.commit()
            return row.answer
    except Exception as e:
        print(f"[analytics] cache_get failed: {e!r}", file=sys.stderr)
        return None


def cache_put_bg(question: str, answer: str) -> None:
    """Save an answer for next time. Fire-and-forget, called from the
    stream's finally block — only for status == 'ok' answers."""
    if _Session is None:
        return
    if not answer or len(answer) < CACHE_MIN_ANSWER:
        return
    qn = normalize_question(question)
    if not qn:
        return
    try:
        task = asyncio.create_task(
            _cache_write(qn, question[:MAX_QUESTION], answer[:MAX_ANSWER]))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
    except RuntimeError:
        pass


async def _cache_write(qn: str, qraw: str, answer: str) -> None:
    try:
        async with _Session() as s:
            exists = (await s.execute(
                select(CachedAnswer.id).where(CachedAnswer.question_norm == qn)
            )).scalar_one_or_none()
            if exists is not None:
                return          # first answer wins until TTL expiry / clear
            s.add(CachedAnswer(question_norm=qn, question_raw=qraw, answer=answer))
            await s.commit()
    except Exception as e:
        # Includes the harmless race where two identical questions arrive at
        # once and both try to insert — the unique index rejects the second.
        print(f"[analytics] cache write failed: {e!r}", file=sys.stderr)


async def cache_clear() -> int:
    """Wipe the whole cache. Wired to /admin/cache-clear and to hide/unhide,
    because a cached answer written before you hid a project could still
    mention it."""
    if _Session is None:
        return 0
    async with _Session() as s:
        res = await s.execute(delete(CachedAnswer))
        await s.commit()
    return res.rowcount or 0


async def cache_list(limit: int = 100) -> list[dict]:
    """What's cached right now, most-reused first."""
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

        cache_hits = (await s.execute(
            select(func.count(ChatTurn.id))
            .where(ChatTurn.created_at >= since, ChatTurn.status == "cache_hit")
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

    return {
        "window_days": days,
        "turns": totals[0],
        "answered_from_cache": cache_hits,
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
