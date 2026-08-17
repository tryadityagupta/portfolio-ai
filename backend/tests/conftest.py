"""
Test fixtures for the semantic cache.

Everything network-shaped is replaced with counting fakes, because the tests'
whole job is to assert HOW MANY paid calls each cache path makes:

    exact hit      -> 0 embeddings, 0 GPT, 0 retrieval
    semantic hit   -> 1 embedding,  0 GPT, 0 retrieval
    miss           -> 1 embedding (reused by FAISS), 1 GPT

FakeEmbeddings maps known question strings to hand-placed unit vectors whose
pairwise cosines are EXACT by construction (2-D rotations), so threshold
tests don't depend on a real embedding model's behavior:

    Q_BASE  [1, 0, 0]
    Q_PARA  cos 0.95 to Q_BASE   (paraphrase        -> above 0.92 threshold)
    Q_PY    cos 0.90 to Q_BASE   ("...using Python" -> below threshold)
    Q_OTHER [0, 0, 1]            (unrelated         -> orthogonal)

Each test gets its own temp SQLite file and a fully reset module state
(analytics engine, semantic index, embed LRU, BM25, daily budget), so tests
can't order-couple.
"""

import asyncio
import math
import os
import sys
import uuid
from types import SimpleNamespace

import httpx
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

# Must exist before app modules import; values are inert (fakes do the work).
os.environ.setdefault("OPENAI_API_KEY", "test-key-never-used")
os.environ.setdefault("SEMANTIC_CACHE_ENABLED", "1")

import analytics            # noqa: E402
import main as app_main    # noqa: E402
import semantic_cache      # noqa: E402
from langchain_core.documents import Document  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical test questions + hand-placed vectors (cosines are exact).
# ---------------------------------------------------------------------------

Q_BASE = "What projects has Aditya built?"
Q_PARA = "Which projects has Aditya worked on?"
Q_PY = "Which projects has Aditya built using Python?"
Q_OTHER = "Where did Aditya study?"
Q_SKILLS = "What are Aditya's main AI skills?"
Q_SKILLS_PARA = "Which AI and ML technologies does Aditya know?"


def _vec_at(cos_to_base: float) -> list[float]:
    """Unit vector in the xy-plane at the angle whose cosine to [1,0,0] is
    exactly `cos_to_base`."""
    return [cos_to_base, math.sqrt(1.0 - cos_to_base * cos_to_base), 0.0]


VECTORS = {
    Q_BASE: [1.0, 0.0, 0.0],
    Q_PARA: _vec_at(0.95),
    Q_PY: _vec_at(0.90),
    Q_OTHER: [0.0, 0.0, 1.0],
    Q_SKILLS: [0.0, 1.0, 0.0],
    Q_SKILLS_PARA: [math.sqrt(1 - 0.94 ** 2), 0.94, 0.0],  # cos 0.94 to SKILLS
}


class FakeEmbeddings:
    """Stands in for rag.get_embeddings(). Counts every aembed_query call —
    the single most important number in this test suite."""

    def __init__(self):
        self.query_calls = 0
        self.queries: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        self.queries.append(text)
        if text in VECTORS:
            return list(VECTORS[text])
        # Unknown strings get a deterministic unit vector far from the
        # hand-placed plane (z-heavy), so they never accidentally cache-hit.
        h = (hash(text) % 997) / 997.0
        v = [0.1 * h, 0.1 * (1 - h), 1.0]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]


class FakeVectorDB:
    """Stands in for the LangChain FAISS store. Records which retrieval API
    was used and with what argument, so tests can prove (a) the by-vector
    path received the EXACT reused embedding and (b) the embed-inside-search
    path was never touched on that request."""

    def __init__(self):
        self._docs = [
            Document(page_content="[Project: portfolio-ai]\nRAG chatbot "
                                  "with FastAPI, FAISS and BM25.",
                     metadata={"source": "project", "repo": "portfolio-ai"}),
            Document(page_content="education: B.Tech CSE, NIT Silchar, "
                                  "CGPA 8.54",
                     metadata={"source": "profile"}),
        ]
        self.docstore = SimpleNamespace(
            _dict={i: d for i, d in enumerate(self._docs)})
        self.by_vector_calls: list[list[float]] = []
        self.by_text_calls: list[str] = []

    async def asimilarity_search_by_vector(self, embedding, k=4, **kw):
        self.by_vector_calls.append(list(embedding))
        return self._docs[:k]

    async def asimilarity_search(self, query, k=4, **kw):
        self.by_text_calls.append(query)
        return self._docs[:k]


class FakeGPT:
    """Stands in for main.client (AsyncOpenAI). Streams a canned answer in
    chunks shaped like the real SDK objects (delta/choices/usage)."""

    def __init__(self):
        self.calls = 0
        self.answer = ("Aditya has built portfolio-ai, an AI-powered "
                       "portfolio chatbot with hybrid retrieval and "
                       "streaming answers over SSE.")
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        text = self.answer

        async def stream():
            for i in range(0, len(text), 16):
                yield SimpleNamespace(
                    usage=None,
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content=text[i:i + 16]))])
            yield SimpleNamespace(usage=SimpleNamespace(   # usage-only chunk
                prompt_tokens=50, completion_tokens=30), choices=[])
        return stream()


async def drain_writes():
    """Wait for every fire-and-forget task (log rows, cache writes, index
    adds, lazy removals) to land, so assertions see the final state."""
    for _ in range(6):
        pending = (set(analytics._pending)
                   | set(semantic_cache.cache_index._pending))
        if not pending:
            break
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)


@pytest.fixture
async def env(tmp_path, monkeypatch):
    """One fully wired, fully isolated app per test."""
    # Fresh DB file per test; analytics reads this attribute at startup().
    monkeypatch.setattr(
        analytics, "DATABASE_URL",
        f"sqlite+aiosqlite:///{tmp_path}/analytics-{uuid.uuid4().hex}.db")
    await analytics.startup()

    fakes = SimpleNamespace(embeddings=FakeEmbeddings(),
                            vdb=FakeVectorDB(), gpt=FakeGPT())

    monkeypatch.setattr(app_main, "get_embeddings", lambda: fakes.embeddings)
    monkeypatch.setattr(app_main, "vector_db", fakes.vdb)
    monkeypatch.setattr(app_main, "client", fakes.gpt)
    monkeypatch.setattr(app_main, "SEMANTIC_CACHE_ENABLED", True)
    monkeypatch.setattr(app_main, "SEMANTIC_CACHE_THRESHOLD", 0.92)
    monkeypatch.setattr(app_main, "SEMANTIC_CACHE_MIN_MARGIN", 0.0)

    # Reset every piece of module-level state previous tests could have bent.
    app_main._embed_lru.clear()
    app_main._daily_usage.update(count=0)
    app_main.build_bm25()                 # over the fake docstore
    await semantic_cache.cache_index.reset()

    bm25_calls = []
    real_bm25 = app_main._bm25_search

    def counting_bm25(query, k):
        bm25_calls.append(query)
        return real_bm25(query, k)
    monkeypatch.setattr(app_main, "_bm25_search", counting_bm25)
    fakes.bm25_calls = bm25_calls

    transport = httpx.ASGITransport(app=app_main.app)
    ip = f"10.0.{uuid.uuid4().int % 250}.{uuid.uuid4().int % 250}"
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://testserver",
                                 headers={"x-forwarded-for": ip}) as client:
        fakes.client = client
        yield fakes

    await drain_writes()
    await analytics.shutdown()
    analytics._engine, analytics._Session = None, None
    await semantic_cache.cache_index.reset()


async def chat(env, message: str) -> str:
    """POST /chat and reassemble the streamed SSE tokens into the answer."""
    r = await env.client.post("/chat", json={"message": message})
    assert r.status_code == 200
    parts = []
    for line in r.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            import json as _json
            parts.append(_json.loads(line[6:])["token"])
    await drain_writes()
    return "".join(parts)
