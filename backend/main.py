from fastapi import FastAPI, Header, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from datetime import date
import json
import os
import re
import time

from collections import OrderedDict

from rank_bm25 import BM25Okapi

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag import load_vector_store, build_vector_store, get_embeddings, EMBEDDING_MODEL
import projects
import analytics   # conversation log + answer cache (backend/analytics.py)
import semantic_cache   # in-memory FAISS index over cached-question vectors


vector_db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE, right after the server starts listening on its port.
    # We load (or build) the FAISS index here so /health can answer first and
    # Render doesn't mark the deploy as failed during a cold start.
    global vector_db
    vector_db = load_vector_store()
    build_bm25()   # keyword index over the same chunks FAISS just loaded
    await analytics.startup()   # opens the DB; on failure it disables itself
    if SEMANTIC_CACHE_ENABLED:
        # Rehydrate the in-memory question index from vectors ALREADY stored
        # in answer_cache. Pure local work — zero OpenAI calls at boot, no
        # matter how many rows exist. Rows without embeddings (written before
        # this feature, or while it was disabled) are skipped, not re-embedded:
        # they still serve exact hits and the TTL retires them within days.
        try:
            entries = await analytics.load_semantic_entries(EMBEDDING_MODEL)
            n = await semantic_cache.cache_index.load(entries)
            print(f"[semantic-cache] ready ({n} question vectors, "
                  f"threshold {SEMANTIC_CACHE_THRESHOLD})")
        except Exception as e:   # cache starts empty; /chat is unaffected
            print(f"[semantic-cache] startup load failed: {e!r}")
    yield
    await analytics.shutdown()  # let in-flight log writes land


app = FastAPI(lifespan=lifespan)

# ------------------------------------------------------------------------------
# CORS - only these frontends may call this API from a browser.
# "*" + credentials is an invalid combo per the CORS spec, and an open list
# would let any website embed a widget that burns our OpenAI quota.
# ------------------------------------------------------------------------------

ALLOWED_ORIGINS = [
    "https://ysadityagupta.co.in",
    "https://www.ysadityagupta.co.in",
    "https://portfolio-ai-iota-one.vercel.app/",  # Vercel prod URL
    "http://localhost:3000",  # Local frontend dev
    "http://127.0.0.1:5500",
    "http://localhost:5500",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)

# --------------------------------------------------------------------------------
# RATE LIMITING - two layers:
#   1) per-IP: 20 chat requests/minute (stops one person hammering the bot)
#   2) global daily budget: DAILY_CHAT_BUDGET requests/day across everyone
#       (caps the worst-case OpenAI bill even if many IPs attack at once)
# The daily counter is in-memory, so it resets on restart - fine for a spend
# cap; it doesn't need to be exact, it needs to bound the damage.
# ---------------------------------------------------------------------------------


def client_ip(request: Request) -> str:
    # On Render we sit behind a proxy, so the real visitor IP arrives in the
    # X-Forwarded-For header; request.client.host would be the proxy itself
    # and every visitor would share one rate-limit bucket.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return get_remote_address(request)


def geo_country(request: Request) -> str | None:
    """Some edge proxies (e.g. Cloudflare) stamp the visitor's country onto
    the request as a header — free, no IP lookup. Locally and on plain
    Render this returns None, which is fine."""
    for h in ("cf-ipcountry", "x-vercel-ip-country", "x-geo-country"):
        v = request.headers.get(h)
        if v and v != "XX":
            return v[:4]
    return None


limiter = Limiter(key_func=client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

DAILY_CHAT_BUDGET = int(os.getenv("DAILY_CHAT_BUDGET", "300"))
_daily_usage = {"day": date.today().isoformat(), "count": 0}

# ------------------------------------------------------------------------------
# SEMANTIC CACHE config. All optional; unset = sensible defaults; the ENABLED
# flag is the rollback switch — 0 restores the exact-cache → RAG → GPT
# pipeline byte-for-byte (retrieve_docs embeds internally again, cache rows
# are written without vectors).
#
# THRESHOLD is the primary safety gate: reuse a cached answer only if the
# new question's cosine similarity to a cached question clears it. 0.92 is a
# deliberately conservative starting point for text-embedding-3-small — a
# wrong-but-confident cached answer on a portfolio costs more than one extra
# gpt-4o-mini generation ever will. Tune it with evidence, not vibes:
# evals/eval_semantic_threshold.py sweeps candidate values over golden
# questions + labeled paraphrase pairs, and /admin/analytics reports the
# similarity distribution of real hits.
#
# MIN_MARGIN is an optional second gate: top hit must beat the runner-up by
# this much, else the query "sits between" two cached questions and we
# prefer a miss. 0 disables it — the right default while the cache is small
# enough that near-ties are rare.
# ------------------------------------------------------------------------------

SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "1") == "1"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_MIN_MARGIN = float(os.getenv("SEMANTIC_CACHE_MIN_MARGIN", "0"))

# Tiny in-process LRU: normalized question -> query embedding. Catches
# re-punctuated / re-cased repeats of RECENT questions ("skills?" then
# "skills!!") without an OpenAI call, and also makes retries free. Bounded
# and in-memory only — deliberately NOT a second persistent store.
_EMBED_LRU_MAX = 128
_embed_lru: "OrderedDict[str, list[float]]" = OrderedDict()


async def embed_query_cached(message: str) -> list[float] | None:
    """The ONE place /chat gets a query embedding. Returns None on failure
    so callers can fall back to the legacy path (retrieve_docs embedding
    internally) — an OpenAI embedding hiccup must degrade the cache, never
    break the chat."""
    key = analytics.normalize_question(message) or message[:500]
    hit = _embed_lru.get(key)
    if hit is not None:
        _embed_lru.move_to_end(key)
        return hit
    try:
        vec = await get_embeddings().aembed_query(message)
    except Exception as e:
        print(f"[semantic-cache] embed_query failed: {e!r}")
        return None
    _embed_lru[key] = vec
    if len(_embed_lru) > _EMBED_LRU_MAX:
        _embed_lru.popitem(last=False)
    return vec


def daily_budget_spent() -> bool:
    """Count this request against today's budget. True = budget exhausted."""
    today = date.today().isoformat()
    if _daily_usage["day"] != today:          # first request of a new day
        _daily_usage["day"] = today
        _daily_usage["count"] = 0
    if _daily_usage["count"] >= DAILY_CHAT_BUDGET:
        return True
    _daily_usage["count"] += 1
    return False


client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# A shared secret only YOU know. Set it in .env (ADMIN_TOKEN=something-long) for
# local testing, and in Render's Environment tab for production — the two are
# separate copies. The public /projects and /chat endpoints don't need it; only
# the hide/unhide controls do. Without it set, the admin endpoints refuse to run.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

# Path resolved relative to this file, not the working directory.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_PATH = os.path.join(_BASE_DIR, "data", "project_overrides.json")


class ChatRequest(BaseModel):
    message: str
    # Optional extras the frontend MAY send for analytics. All default to
    # None, so the current frontend keeps working without any change.
    visitor_id: str | None = None
    session_id: str | None = None
    referrer: str | None = None
    page_url: str | None = None
    utm_source: str | None = None


class RepoRequest(BaseModel):
    repo: str


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# PUBLIC: the frontend calls this to draw the project cards.
# Hidden repos are already removed inside get_visible_projects(), so they
# never reach the browser.
# ---------------------------------------------------------------------------
@app.get("/projects")
async def list_projects():
    return {"projects": projects.get_visible_projects()}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — split into a public base (safe to live in this repo) and
# private rules loaded at runtime from a gitignored file / env var, so
# personal guidance never appears in public source.
#
# Load order: PRIVATE_PROMPT_RULES env var wins if set (with "\n" expanded);
# otherwise backend/data/private_rules.txt is read if it exists. Locally you
# keep that file on disk (gitignored); on Render you add it as a Secret File
# at the same path. Missing both -> the bot simply runs on the base rules.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """
You are an AI assistant on Aditya Gupta's personal portfolio website.
Your job is to answer questions about Aditya in a professional, friendly, and confident tone.
Answer only about Aditya. If asked anything completely unrelated to him, politely redirect.
 
--- STRICT RULES (always follow these, they override the context) ---
 
RULE 1 — NEVER reveal Aditya's mobile number under any circumstance.
If someone asks for his phone number or contact number, say:
"I'm not able to share Aditya's phone number here. You can reach him at adityagupta.nits2@gmail.com or connect on LinkedIn."
 
RULE 2 — CTC / salary questions:
If someone asks about Aditya's current CTC, expected CTC, or typical market rates, say:
"That's something best discussed directly with Aditya. Feel free to reach out to him at adityagupta.nits2@gmail.com — he'd be happy to connect."

RULE 3 — Date of joining / notice period questions:
If someone asks when Aditya can join or what his notice period is, say:
"For specific availability and joining timelines, it's best to connect directly with Aditya at adityagupta.nits2@gmail.com — he'll be happy to discuss."

--- GENERAL TONE ---
- Be warm, professional, and concise (2-5 sentences unless more detail is clearly needed).
- If a recruiter is asking, sound like Aditya's advocate — highlight his strengths naturally.
- Never make up facts. If something isn't in the context, say you don't have that detail and suggest they email Aditya.
"""

_PRIVATE_RULES_PATH = os.path.join(_BASE_DIR, "data", "private_rules.txt")


def _load_private_rules() -> str:
    env_val = os.getenv("PRIVATE_PROMPT_RULES")
    if env_val:
        # Env vars are single-line; allow literal "\n" to mean a newline.
        return env_val.replace("\\n", "\n")
    try:
        with open(_PRIVATE_RULES_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


_private_rules = _load_private_rules()
SYSTEM_PROMPT = (
    SYSTEM_PROMPT_BASE + "\n--- ADDITIONAL RULES ---\n" + _private_rules
    if _private_rules else SYSTEM_PROMPT_BASE
)


def build_prompt(context: str, question: str) -> str:
    return f"""
You are an AI assistant answering questions about Aditya Gupta.

Context about Aditya Gupta:
{context}

Question:
{question}

Answer based on the context and the rules in your system instructions:
"""


# ------------------------------------------------------------------------------
# SSE (Server-Sent Events) helpers.
#
# Plain text/plain chunked responses get COALESCED by the proxy layers between
# Uvicorn and the browser (Render's proxy + Cloudflare buffer/compress generic
# text responses). text/event-stream is the one content type every CDN treats
# as "pass each chunk through immediately", so we frame every token as an SSE
# event instead. The headers matter too:
#   - no-transform  -> tells intermediaries not to compress/re-encode (gzip
#                      requires buffering, which is exactly what killed us)
#   - X-Accel-Buffering: no -> disables buffering in nginx-style proxies
# ------------------------------------------------------------------------------

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_token(text: str) -> str:
    # JSON-encode so tokens containing newlines can't break the frame format
    # (a raw "\n" inside an SSE data line would terminate the event early).
    return f"data: {json.dumps({'token': text})}\n\n"


SSE_DONE = "data: [DONE]\n\n"


def sse_response(gen) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)


# ---------------------------------------------------------------------------
# RETRIEVAL, factored out of /chat so it has exactly ONE implementation.
# Both the live endpoint and evals/run_ragas_eval.py call this function, so
# the eval scores describe the real production pipeline — not a copy of it
# that could silently drift out of sync.
#
# HYBRID = dense (FAISS) + keyword (BM25), merged with Reciprocal Rank
# Fusion. Dense-only retrieval kept missing keyword-style questions like
# "has he done fine-tuning?": the answer sat inside one big project doc
# whose embedding averaged over its whole README, so it never cracked the
# top-3. BM25 scores exact terms, so chunks that literally say "fine-tuned"
# now surface; dense search still catches paraphrases. Same pattern as the
# AI-Codebase-Tutor retriever.
# ---------------------------------------------------------------------------

_SUFFIXES = ("ing", "ed", "es", "s")


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens; each token is ALSO emitted with a crude suffix
    strip, so morphological cousins meet somewhere: 'fine-tuning' becomes
    [fine, tuning, tun] and a doc saying 'fine-tuned' becomes
    [fine, tuned, tun] — they overlap on 'tun'. Applied identically to
    documents and queries, so the expansion is symmetric."""
    tokens = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        tokens.append(w)
        for suf in _SUFFIXES:
            if len(w) > len(suf) + 2 and w.endswith(suf):
                tokens.append(w[: -len(suf)])
                break
    return tokens


_bm25 = None
_bm25_docs: list = []


def build_bm25():
    """(Re)build the keyword index over the SAME chunks FAISS holds, so both
    retrievers always describe the same corpus (incl. after hide/unhide)."""
    global _bm25, _bm25_docs
    _bm25_docs = list(vector_db.docstore._dict.values())
    _bm25 = BM25Okapi([_tokenize(d.page_content) for d in _bm25_docs])


def _bm25_search(query: str, k: int):
    if _bm25 is None:
        if vector_db is None:
            return []
        build_bm25()   # lazy fallback: covers the eval script, which sets
        # vector_db directly and never runs the server lifespan
    scores = _bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [_bm25_docs[i] for i in order if scores[i] > 0]


def _rrf_merge(ranked_lists, c: int = 60):
    """Reciprocal Rank Fusion: fuse by RANK, not score, so FAISS distances
    and BM25 scores never need to share a scale. A doc found by both
    retrievers gets both contributions and floats to the top."""
    fused: dict[str, list] = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst):
            entry = fused.setdefault(doc.page_content, [0.0, doc])
            entry[0] += 1.0 / (c + rank + 1)
    return [doc for _, doc in sorted(fused.values(), key=lambda e: -e[0])]


async def retrieve_docs(message: str, k: int = 8, keep: int = 5,
                        query_embedding: list[float] | None = None):
    """Pull k candidates from EACH retriever, fuse, drop currently-hidden
    repos, keep the best `keep`. Repo names compared case-insensitively.
    Chunks are ~700 chars now (see rag.load_project_documents), so 5 chunks
    cost FEWER tokens than the old 3 whole-README docs. Requires `vector_db`
    to be loaded (lifespan does this for the server; the eval script sets it
    explicitly).

    `query_embedding`: /chat already embedded the question once for the
    semantic-cache lookup, so dense retrieval REUSES that exact vector via
    the by-vector FAISS API instead of paying for a second embedding of the
    same string (asimilarity_search embeds internally). One cache-miss
    request = one embedding call, total. BM25 always gets the raw STRING —
    keyword scoring has no use for a vector. When no vector is supplied
    (eval script, debug endpoint, embed failure, feature off) the original
    embed-inside-search path runs unchanged."""
    hidden = projects.get_hidden_set()  # lowercased
    if query_embedding is not None:
        dense = await vector_db.asimilarity_search_by_vector(
            query_embedding, k=k)
    else:
        dense = await vector_db.asimilarity_search(message, k=k)
    sparse = _bm25_search(message, k=k)
    merged = _rrf_merge([dense, sparse])
    return [
        d for d in merged
        if (d.metadata.get("repo") or "").lower() not in hidden
    ][:keep]


@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest):
    # NOTE: the parameter MUST be named `request` for slowapi to find the IP.
    t0 = time.perf_counter()

    # Everything we know about the caller, resolved once and attached to
    # every log row this request produces.
    who = dict(
        visitor_id=req.visitor_id,
        session_id=req.session_id,
        ip_hash=analytics.hash_ip(client_ip(request)),
        country=geo_country(request),
        user_agent=request.headers.get("user-agent"),
        referrer=req.referrer,
        page_url=req.page_url,
        utm_source=req.utm_source,
        question=req.message,
    )

    # 0) CACHE. If this exact question (ignoring case/punctuation) was
    #    answered in the last CACHE_TTL_DAYS, replay the saved answer:
    #    no OpenAI call, no daily-budget spend, near-zero latency. Checked
    #    even before the vector_db readiness gate — a cached answer doesn't
    #    need the index, so it works during cold starts too.
    cached = await analytics.cache_get(req.message)
    if cached is not None:
        ms = int((time.perf_counter() - t0) * 1000)
        analytics.log_turn_bg(**who, answer=cached, status="cache_hit",
                              model="cache", ttft_ms=ms, total_ms=ms)

        async def replay():
            yield ": ok\n\n"
            yield sse_token(cached)   # one big token; the frontend's
            yield SSE_DONE            # typewriter still animates it nicely
        return sse_response(replay())

    if vector_db is None:
        analytics.log_turn_bg(**who, status="not_ready")

        async def not_ready():
            yield sse_token("Service is still starting up. Please try again in a moment.")
            yield SSE_DONE
        return sse_response(not_ready())

    # Spend cap: past the daily budget we answer politely WITHOUT calling
    # OpenAI, so the worst-case daily bill is bounded no matter the traffic.
    # Logged too — these rows tell you real traffic is being turned away.
    if daily_budget_spent():
        analytics.log_turn_bg(**who, status="over_budget")

        async def over_budget():
            yield sse_token("The chatbot has hit its daily usage limit. "
                            "Please try again tomorrow, or email Aditya at "
                            "adityagupta.nits2@gmail.com.")
            yield SSE_DONE
        return sse_response(over_budget())

    # 0.5) SEMANTIC CACHE. Placed AFTER the daily-budget check on purpose:
    #    this step spends an OpenAI embedding call, and the budget's job is
    #    to bound the worst-case OpenAI bill — so it must gate EVERY paid
    #    call, embeddings included. The trade: a semantic hit consumes one
    #    budget slot (like any answered request) but turns that slot's cost
    #    from a generation into an embedding, roughly two orders of
    #    magnitude cheaper. Exact hits stay ABOVE the budget check and stay
    #    completely free. Worst-case daily spend can only go DOWN.
    #
    #    The embedding generated here is reused for document retrieval below
    #    on a miss — one embedding per request, used twice, never generated
    #    twice.
    query_vec: list[float] | None = None
    if SEMANTIC_CACHE_ENABLED:
        query_vec = await embed_query_cached(req.message)
        if query_vec is not None:
            found = await semantic_cache.cache_index.lookup(
                query_vec, SEMANTIC_CACHE_THRESHOLD, SEMANTIC_CACHE_MIN_MARGIN)
            if found is not None:
                cache_id, sim = found
                # The index only NOMINATES; the DB row is the authority on
                # TTL and knowledge version. None => expired/stale/deleted:
                # drop the dangling vector and fall through to real RAG.
                cached_sem = await analytics.cache_get_by_id(cache_id)
                if cached_sem is None:
                    semantic_cache.cache_index.remove_later(cache_id)
                else:
                    ms = int((time.perf_counter() - t0) * 1000)
                    analytics.log_turn_bg(
                        **who, answer=cached_sem, status="semantic_hit",
                        model="semantic-cache", cache_similarity=round(sim, 4),
                        ttft_ms=ms, total_ms=ms)

                    async def replay_semantic():
                        yield ": ok\n\n"
                        yield sse_token(cached_sem)
                        yield SSE_DONE
                    return sse_response(replay_semantic())

    # 1) RETRIEVAL. We pull a few EXTRA chunks (k=6) then drop any that belong
    #    to a currently-hidden repo, and keep the best 3. This is a live safety
    #    net: even if the index was built before you hid something, the hidden
    #    project's text can't reach the model on this request. The logic lives
    #    in retrieve_docs() above so the RAGAS eval exercises this same path.
    #    query_vec (if we have one) is REUSED here — see retrieve_docs().
    visible_docs = await retrieve_docs(req.message, query_embedding=query_vec)
    context = "\n".join(d.page_content for d in visible_docs)
    sources = sorted({d.metadata.get("repo") for d in visible_docs
                      if d.metadata.get("repo")})

    # 2) AUGMENT — stuff the retrieved context into the prompt (the "A" in RAG).
    prompt = build_prompt(context, req.message)

    # 3) GENERATION — stream tokens back to the browser as they're produced.
    #    While streaming we also ACCUMULATE the answer, because the log row
    #    and the cache entry can only be written once the full text exists.
    async def token_stream():
        parts: list[str] = []
        ttft_ms = None
        usage = None
        status = "aborted"   # flips to "ok" only if the stream completes

        try:
            # SSE comment frame, sent before the first OpenAI token exists.
            # Invisible to the client but pushes bytes down the wire now, so
            # proxies that wait for "first body bytes" open the pipe early.
            yield ": ok\n\n"
            try:
                stream = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=250,
                    stream=True,
                    # Without this a streamed response reports NO token
                    # counts. It adds one final chunk whose choices == [].
                    stream_options={"include_usage": True},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if not chunk.choices:
                        continue          # the usage-only final chunk
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = int((time.perf_counter() - t0) * 1000)
                        parts.append(delta)
                        yield sse_token(delta)
            except Exception:
                status = "error"
                yield sse_token("Sorry, I couldn't process that right now. "
                                "Please try again later.")
            # Explicit end-of-stream sentinel (same convention OpenAI's own
            # API uses) so the client can tell "finished" from "died".
            yield SSE_DONE
            if status != "error":
                status = "ok"
        finally:
            # Runs on success, on error, AND when the visitor closes the tab
            # mid-answer (GeneratorExit) — partial answers are logged with
            # status "aborted". Both calls are fire-and-forget: they add no
            # latency and can never break the stream.
            answer = "".join(parts)
            analytics.log_turn_bg(
                **who,
                answer=answer,
                sources=sources,
                status=status,
                model="gpt-4o-mini",
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                ttft_ms=ttft_ms,
                total_ms=int((time.perf_counter() - t0) * 1000),
            )
            if status == "ok":
                # The embedding stored with the row is the SAME vector this
                # request already generated (and used for retrieval) — the
                # background write re-embeds nothing. If query_vec is None
                # (feature off / embed failed) the row is exact-match only.
                analytics.cache_put_bg(
                    req.message, answer,
                    embedding=query_vec,
                    embedding_model=(EMBEDDING_MODEL
                                     if query_vec is not None else None),
                    knowledge_version=analytics.get_knowledge_version())

    return sse_response(token_stream())


# ---------------------------------------------------------------------------
# ADMIN (token-protected): the hide/unhide controls used by admin.html.
# ---------------------------------------------------------------------------

def _check_admin(token: str | None):
    if not ADMIN_TOKEN:
        raise HTTPException(
            503, "Admin controls are disabled (ADMIN_TOKEN not set).")
    if token != ADMIN_TOKEN:
        raise HTTPException(401, "Invalid admin token.")


def _read_overrides() -> dict:
    with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_overrides(data: dict):
    with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _rebuild_index():
    """Re-embed from the current visible list, then swap the loaded index."""
    global vector_db
    projects.invalidate_cache()
    build_vector_store()
    vector_db = load_vector_store()
    build_bm25()   # keyword index must track the new chunk set too
    # The knowledge base just changed, so every answer cached before this
    # point describes a world that no longer exists. Minting a new version
    # makes BOTH cache levels refuse those rows immediately (lazy-deleted on
    # touch). hide/unhide additionally queue a full cache_clear — belt and
    # braces — but the version bump also protects any FUTURE caller of this
    # function that forgets to clear. Safe from this worker thread: the DB
    # write is scheduled onto the app's event loop.
    analytics.bump_knowledge_version_bg()


@app.get("/admin/projects")
async def admin_list(x_admin_token: str | None = Header(default=None)):
    """Every owned repo plus whether it's currently hidden — powers the toggles."""
    _check_admin(x_admin_token)
    import github_sync
    hidden = projects.get_hidden_set()  # lowercased
    repos = github_sync.fetch_repos()
    return {
        "repos": [
            {"repo": r["name"], "hidden": r["name"].lower() in hidden,
             "description": r.get("description")}
            for r in repos
        ],
        "hidden": sorted(hidden),
    }


@app.get("/admin/debug-retrieval")
async def admin_debug_retrieval(q: str,
                                x_admin_token: str | None = Header(default=None)):
    """First stop whenever an answer looks wrong: shows the exact chunks the
    model would receive for `q`, in fused order. If the right chunk isn't in
    this list, it's a retrieval problem; if it IS here but the answer is
    still bad, it's a prompting problem. Token-gated (costs one embedding
    call per hit). Try:
      curl -H "X-Admin-Token: $TOKEN" "$API/admin/debug-retrieval?q=fine+tuning"
    """
    _check_admin(x_admin_token)
    docs = await retrieve_docs(q)
    return {
        "query": q,
        "kept": [
            {"source": d.metadata.get("source"),
             "repo": d.metadata.get("repo"),
             "preview": d.page_content[:300]}
            for d in docs
        ],
    }


@app.get("/admin/analytics")
async def admin_analytics(days: int = 30,
                          x_admin_token: str | None = Header(default=None)):
    """Totals, cache-hit count, questions per day, traffic sources,
    top questions. Try: curl -H "X-Admin-Token: $TOKEN" "$API/admin/analytics?days=7" """
    _check_admin(x_admin_token)
    return await analytics.summary(days)


@app.get("/admin/transcripts")
async def admin_transcripts(limit: int = 50, visitor_id: str | None = None,
                            x_admin_token: str | None = Header(default=None)):
    """Full Q&A log, newest first. Pass ?visitor_id=... to read one
    person's entire history across visits."""
    _check_admin(x_admin_token)
    return {"turns": await analytics.recent(limit, visitor_id)}


@app.get("/admin/cache")
async def admin_cache(x_admin_token: str | None = Header(default=None)):
    """What's cached right now, most-reused first — plus semantic-cache
    state (config + live vector count). Raw vectors are never exposed."""
    _check_admin(x_admin_token)
    return {
        "cached": await analytics.cache_list(),
        "semantic_enabled": SEMANTIC_CACHE_ENABLED,
        "semantic_index_size": semantic_cache.cache_index.size(),
        "semantic_threshold": SEMANTIC_CACHE_THRESHOLD,
        "semantic_min_margin": SEMANTIC_CACHE_MIN_MARGIN,
        "knowledge_version": analytics.get_knowledge_version(),
    }


@app.post("/admin/cache-clear")
async def admin_cache_clear(x_admin_token: str | None = Header(default=None)):
    """Manual flush — e.g. after you update project READMEs and want fresh
    answers immediately instead of waiting out the TTL."""
    _check_admin(x_admin_token)
    return {"cleared": await analytics.cache_clear()}


@app.post("/admin/hide")
async def admin_hide(req: RepoRequest, bg: BackgroundTasks,
                     x_admin_token: str | None = Header(default=None)):
    _check_admin(x_admin_token)
    data = _read_overrides()
    hidden = set(data.get("hidden", []))
    hidden.add(req.repo)
    data["hidden"] = sorted(hidden)
    _write_overrides(data)
    projects.invalidate_cache()
    # Rebuild the chatbot's memory in the background so the response is instant.
    bg.add_task(_rebuild_index)
    # Cached answers were written BEFORE the hide — they may mention the
    # hidden repo, so the whole cache goes too.
    bg.add_task(analytics.cache_clear)
    return {"ok": True, "hidden": data["hidden"],
            "note": "Frontend updates now. Chatbot forgets it within a few seconds. "
                    "Commit project_overrides.json to make this permanent across redeploys."}


@app.post("/admin/unhide")
async def admin_unhide(req: RepoRequest, bg: BackgroundTasks,
                       x_admin_token: str | None = Header(default=None)):
    _check_admin(x_admin_token)
    data = _read_overrides()
    hidden = set(data.get("hidden", []))
    # Remove case-insensitively so a differently-cased entry still clears.
    data["hidden"] = sorted(h for h in hidden if h.lower() != req.repo.lower())
    _write_overrides(data)
    projects.invalidate_cache()
    bg.add_task(_rebuild_index)
    bg.add_task(analytics.cache_clear)   # stale answers omit this repo
    return {"ok": True, "hidden": data["hidden"],
            "note": "Project is public again. Commit project_overrides.json to persist."}
