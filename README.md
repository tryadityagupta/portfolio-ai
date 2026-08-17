# Portfolio AI — RAG-Powered Personal Chatbot with GitHub-Synced Projects

A production-deployed AI assistant on top of a personal portfolio website. Visitors chat with an
assistant that answers questions about Aditya Gupta's skills, experience, and projects — powered by a
FastAPI backend using Retrieval-Augmented Generation (RAG) with OpenAI embeddings and `gpt-4o-mini`.
Answers are **streamed token-by-token** over Server-Sent Events, so replies start appearing almost
instantly.

The project list is **synced live from GitHub**. New repositories appear on the site (and in the
chatbot's knowledge) automatically, and any project can be **hidden** with one toggle — which removes it
from *both* the website and the chatbot in the same action.

- **Live site:** [https://ysadityagupta.co.in](https://ysadityagupta.co.in) (static frontend on Vercel)
- **API:** [https://portfolio-ai-qoer.onrender.com](https://portfolio-ai-qoer.onrender.com) (FastAPI backend on Render)

---

## Project Structure

```
PORTFOLIO-AI/
├── backend/
│   ├── data/
│   │   ├── profile.json              # Structured profile data (skills, experience, achievements)
│   │   └── project_overrides.json    # Visibility control: hidden / pinned / per-repo overrides
│   ├── vector_store/                 # Document FAISS index — BUILT AT RUNTIME, NOT committed (gitignored)
│   ├── evals/
│   │   ├── golden_dataset.json       # Questions with known-true reference answers
│   │   ├── run_ragas_eval.py         # RAGAS scoring of the REAL retrieve→generate path
│   │   ├── semantic_pairs.json       # Labeled paraphrase pairs (reuse-safe vs must-miss)
│   │   └── eval_semantic_threshold.py# Sweeps SEMANTIC_CACHE_THRESHOLD candidates over real embeddings
│   ├── tests/
│   │   ├── conftest.py               # Counting fakes: embeddings, vector store, GPT stream
│   │   └── test_semantic_cache.py    # Locks in the paid-call budget of every cache path
│   ├── .env                          # Local secrets (not committed)
│   ├── Aditya_Gupta_AI_ML.pdf        # Optional resume source for RAG (not committed)
│   ├── github_sync.py                # Fetches repos + READMEs from the GitHub REST API
│   ├── projects.py                   # Single source of truth: merges GitHub + overrides, applies hiding
│   ├── analytics.py                  # Conversation log + two-level answer cache (SQLite/Postgres)
│   ├── semantic_cache.py             # In-memory FAISS index over cached-QUESTION embeddings
│   ├── main.py                       # FastAPI app: /chat, /projects, /admin/*, SSE streaming, lifespan
│   ├── rag.py                        # Shared embeddings client, vector store build/load, doc loaders
│   └── requirements.txt              # Python dependencies
├── frontend/
│   ├── index.html                    # Single-page portfolio; renders project cards from /projects
│   └── admin.html                    # Private, token-gated page to show/hide projects
└── .gitignore
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Static HTML, CSS, vanilla JS (portfolio + SSE chat widget + admin panel) |
| Backend | FastAPI (Python), async endpoints |
| Streaming | Server-Sent Events (`text/event-stream`) over FastAPI `StreamingResponse` |
| Project sync | GitHub REST API (via `urllib`, standard library) |
| Embeddings | OpenAI `text-embedding-3-small` (one shared client for documents *and* query embeddings) |
| Vector Store | FAISS (CPU) — two separate indexes: document chunks + cached-question vectors |
| LLM | OpenAI `gpt-4o-mini` (streaming) |
| RAG Framework | LangChain |
| Answer Cache | Two-level: exact normalized match → semantic (cosine ≥ threshold), SQLAlchemy-persisted |
| Persistence | SQLite locally, Postgres in production (same code path, `DATABASE_URL` switches it) |
| Deployment | Backend on Render · Frontend on Vercel (custom domain via GoDaddy DNS) |

---

## How It Works

### One source of truth for projects

`projects.py` exposes `get_visible_projects()`. It:

1. Fetches every owned repo from GitHub (`github_sync.py`), cached in memory for 10 minutes.
2. Removes any repo listed in `hidden` (see below) — **this is the privacy step**.
3. Merges each remaining repo with your polished copy from `project_overrides.json` (your fields win;
   anything you didn't specify falls back to the live GitHub description / language / topics).
4. Orders them: `pinned` repos first (in your order), then the rest by most-recent activity.

Both the **website** (`GET /projects`) and the **chatbot** (`rag.py`, when building the index) call this
same function. Because they share one filtered list, they can never disagree about what's visible.
Matching against `hidden` / `pinned` / `overrides` is **case-insensitive**, so a repo GitHub returns as
`AI-Codebase-Tutor` matches a lowercase key `ai-codebase-tutor`.

### The chat request — two cache levels in front of RAG

```
User query
   ↓
Exact cache  (normalized string match, DB lookup)
   ├─ HIT  → replay saved answer          — 0 OpenAI calls
   └─ MISS
        ↓  generate ONE query embedding
Semantic cache  (cosine vs. previously cached questions, in-memory FAISS)
   ├─ HIT  → replay saved answer          — 1 embedding call, 0 GPT calls
   └─ MISS
        ↓  REUSE the same embedding
Hybrid retrieval:  FAISS (by-vector) + BM25 → RRF → hidden-repo filter
        ↓
gpt-4o-mini → streamed over SSE → background cache write (answer + embedding)
```

1. **Exact cache** — `"What are his SKILLS??"` and `"what are his skills"` normalize to the same key.
   A hit replays the stored answer with **zero** OpenAI calls, zero retrieval, and no daily-budget
   spend. Checked before anything else, so it even works during cold starts.
2. **Semantic cache** — on an exact miss, the question is embedded **once** and compared (plain cosine
   similarity, FAISS `IndexFlatIP` over normalized vectors) against the embeddings of previously
   cached questions. `"Which projects has Aditya worked on?"` can reuse the answer cached for
   `"What projects has Aditya built?"`. A hit costs one embedding call (~100× cheaper than a
   generation) and skips retrieval + GPT entirely.
3. **Retrieval on a miss** — the **same embedding is reused** for dense FAISS search
   (`asimilarity_search_by_vector`), so a cache-miss request never embeds the query twice. BM25 gets
   the raw string, results are fused with Reciprocal Rank Fusion, and chunks from currently-hidden
   repos are dropped — the same live safety net as before.
4. **Generation** — `gpt-4o-mini` answers, streamed as **Server-Sent Events** exactly as before: the
   cache layers changed *whether* generation happens, never *how* streaming works.
5. **Background cache write** — after a successful stream, the answer is saved along with the query
   embedding **already generated in step 2** (no extra API call), fire-and-forget: cache writes can
   never delay or break a response.

**What each request path costs:**

| Path | Embedding calls | GPT calls | Retrieval |
|---|---|---|---|
| Exact cache hit | 0 | 0 | none |
| Semantic cache hit | 1 (0 if the in-process LRU has it) | 0 | none |
| Cache miss (full RAG) | **1, reused for FAISS** | 1 | FAISS + BM25 |

Semantic caching does **not** eliminate all OpenAI calls — a semantic hit still pays for one
embedding, unless the tiny in-process LRU of recent query embeddings already holds that (normalized)
question. What it eliminates is *generations*, which is where the money and latency are: the metric
that matters is **GPT calls avoided while answers stay correct**, not raw hit rate.

**Why similarity is cosine math, not an LLM judge:** asking a model "are these questions the same?"
would cost latency and money on *every* request — including the misses — which defeats the point of a
cache, and it would make cache behavior non-deterministic and untestable. A vector comparison is
microseconds, free, and reproducible in tests.

**Why the threshold is conservative (default `0.92`):** serving a *wrong* cached answer on a
portfolio is strictly worse than paying for one more `gpt-4o-mini` generation, so false negatives are
preferred over false positives. `"Which projects has Aditya built using Python?"` should *not* reuse
the generic projects answer even though the wording is close. The threshold is an env var, tuned with
evidence: `evals/eval_semantic_threshold.py` sweeps candidates over labeled paraphrase pairs, and
`/admin/analytics` reports the similarity distribution of real hits. An optional margin gate
(`SEMANTIC_CACHE_MIN_MARGIN`) can additionally reject queries that sit ambiguously between two
different cached questions.

**How cached answers stay fresh and private:**

- **TTL** — entries older than `CACHE_TTL_DAYS` (default 7) are never served, no matter how similar;
  they're lazily deleted on touch.
- **Knowledge versioning** — every cached answer is stamped with the knowledge-base version that
  produced it. Rebuilding the index (e.g. hide/unhide) mints a new version, instantly invalidating
  answers generated from the old knowledge state.
- **Hide/unhide clears everything** — hiding a project still wipes the *entire* answer cache **and**
  resets the in-memory question index in the same action, so a hidden project can never resurface
  through a similarity hit. Vectors never outlive their database rows.
- **Kill switch** — `SEMANTIC_CACHE_ENABLED=0` reverts to the plain exact-cache → RAG → GPT pipeline
  (retrieval embeds internally again, rows are written without vectors): a one-variable rollback.

The non-blocking `AsyncOpenAI` client is used throughout, so a slow OpenAI call for one visitor doesn't
freeze the server for others (concurrency). The semantic index itself is process-local and rebuilt at
startup from vectors already stored in the database — **zero** embedding API calls at boot, however
many rows exist. Rows written before this feature simply lack vectors: they keep serving exact hits
and age out via the TTL, no backfill job required.

### Server startup

A FastAPI `lifespan` event loads the FAISS index into memory after the app binds to its port (so the
`/health` check passes first). If no index exists on disk — the normal case now, since the index is no
longer committed — it is **built on the spot** from the resume PDF (if present) + `profile.json` +
the live, hidden-filtered GitHub project list.

---

## Project Visibility System

Everything is controlled by `backend/data/project_overrides.json`:

| Key | Purpose |
|---|---|
| `hidden` | Repos to hide **everywhere** — removed from the site *and* the chatbot's knowledge. |
| `pinned` | Repos to show first, in this exact order. |
| `overrides` | Your polished display name, category, description, and tech pills per repo. |

A repo **not** mentioned anywhere still appears automatically, using its GitHub info. Default is "show";
hiding is the only manual action — so a brand-new repo shows up on its own.

### Two ways to hide/show a project

- **Admin panel (`frontend/admin.html`)** — enter your `ADMIN_TOKEN`, then click a project's toggle.
  The site updates immediately and the chatbot rebuilds its memory within a few seconds.
- **Edit the file** — change the `hidden` list in `project_overrides.json` and push.

Because Render's disk is wiped on redeploy, a toggle flipped in the panel is **not permanent** — the
panel shows you the exact JSON snippet to paste into `project_overrides.json` and commit to make it
stick. Think of the panel as the light switch and the committed file as the fuse box.

---

## Local Setup

### Backend

```bash
cd backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create `backend/.env`:

```
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=github_pat_...     # optional but recommended (avoids GitHub rate limits)
ADMIN_TOKEN=some-long-random-string   # required only to use the admin panel
# GITHUB_USERNAME=tryadityagupta      # optional; this is the default

# Answer cache (all optional — these are the defaults):
# CACHE_TTL_DAYS=7                    # how long any cached answer stays valid
# SEMANTIC_CACHE_ENABLED=1            # 0 = rollback to exact-cache → RAG → GPT
# SEMANTIC_CACHE_THRESHOLD=0.92      # min cosine similarity to reuse an answer
# SEMANTIC_CACHE_MIN_MARGIN=0        # >0 also requires top hit to beat runner-up by this
# SEMANTIC_CACHE_MAX_ENTRIES=500     # cap on embedded rows; least-hit/oldest evicted first
# DAILY_CHAT_BUDGET=300              # max answered requests/day (spend cap)
```

Nothing new is *required* to boot — every semantic-cache variable has a working default, and an
existing `analytics.db` from before the semantic cache opens as-is (the new nullable columns are
added automatically on startup).

Run the server (the index builds itself on first boot):

```bash
uvicorn main:app --reload
```

The API runs at `http://localhost:8000`. Test the stream with `--no-buffer`:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/projects
curl --no-buffer -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What projects has Aditya built?"}'
```

The response is a Server-Sent Events stream, so you'll see frames rather than plain prose:

```
: ok

data: {"token": "Aditya"}

data: {"token": " has"}

data: {"token": " built"}

data: [DONE]

```

`: ok` is an SSE comment frame sent before the first model token — it's ignored by the client and exists
only to push bytes down the wire immediately. `--no-buffer` is what lets you *see* frames arrive
incrementally; without it, curl buffers and the output looks like one burst even when the server is
streaming correctly. To timestamp each network read (useful for proving the stream isn't being coalesced
by a proxy), add `--trace-time`.

### Frontend

```bash
cd frontend
python -m http.server 5500
# or use VS Code Live Server
```

Open `http://localhost:5500`. Both `index.html` and `admin.html` **auto-detect** their backend: on
`localhost` they call `http://localhost:8000`; anywhere else they call the deployed Render API. Serve the
pages over `localhost` (not `file://`) so this detection works. The admin panel is at
`http://localhost:5500/admin.html`.

> Note: `ADMIN_TOKEN` is read from your **local** `.env` for local testing and from **Render's**
> Environment tab in production. They are separate copies — setting one does not set the other. If the
> admin panel says "ADMIN_TOKEN isn't set on the server," add it to the `.env` of whichever backend the
> page is talking to and restart that server.

> Note: the origin you serve the frontend from must be in `ALLOWED_ORIGINS` in `main.py`.
> `http://localhost:5500` and `http://127.0.0.1:5500` are **different origins** to the CORS spec — a
> chat request that fails with "connection issue" while the backend logs show no request at all is
> almost always this.

---

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Health check — returns `{"status": "ok"}`. Also used by the keep-warm pinger. |
| GET | `/projects` | — | The visible (hidden-filtered) project list the frontend renders. |
| POST | `/chat` | — | Accepts `{"message": "..."}`. Returns a **Server-Sent Events** stream (`text/event-stream`): each token arrives as a `data: {"token": "..."}\n\n` frame, and the stream ends with `data: [DONE]\n\n`. Rate-limited to 20/min per IP; on 429 the body is plain JSON, not a stream. |
| GET | `/admin/projects` | `X-Admin-Token` | Every owned repo + whether it's currently hidden (powers the toggles). |
| POST | `/admin/hide` | `X-Admin-Token` | Body `{"repo": "..."}` — hides a repo, rebuilds the index, bumps the knowledge version, clears **both** cache levels. |
| POST | `/admin/unhide` | `X-Admin-Token` | Body `{"repo": "..."}` — un-hides a repo; same rebuild + cache-clear sequence. |
| GET | `/admin/analytics` | `X-Admin-Token` | Traffic totals plus cache economics: exact vs. semantic hits, total hit rate, GPT generations, and the similarity distribution (avg/min) of semantic hits. |
| GET | `/admin/transcripts` | `X-Admin-Token` | Full Q&A log, newest first; `?visitor_id=` filters one visitor. |
| GET | `/admin/cache` | `X-Admin-Token` | Cached answers (question, hits, `embedding_available`, knowledge version, answer preview) + live semantic-cache state: enabled flag, in-memory vector count, threshold, margin. Raw vectors are never exposed. |
| POST | `/admin/cache-clear` | `X-Admin-Token` | Wipes the answer cache **and** resets the in-memory semantic index in the same call. |
| GET | `/admin/debug-retrieval` | `X-Admin-Token` | `?q=...` — shows the exact fused chunks retrieval would feed the model for `q`. |

### `/chat` response format

Every `/chat` response — including the startup and daily-budget guard messages — uses the same SSE
frame format, so the client has exactly one code path to maintain:

| Frame | Meaning |
|---|---|
| `: ok\n\n` | Comment frame. Ignored by clients; flushes first bytes before the model responds. |
| `data: {"token": "..."}\n\n` | One delta of generated text. Token text is JSON-encoded, so a token containing a newline can't break the frame. |
| `data: [DONE]\n\n` | End of stream. Lets the client tell "finished" apart from "connection dropped". |

Note that one network read may contain several frames or only *part* of one — SSE guarantees frame
*format*, not frame-per-packet delivery. Any client must buffer reads and only process frames whose
`\n\n` terminator has arrived.

---

## The Vector Store

The FAISS index is **built at runtime and not committed to git**. To rebuild it manually (e.g. after
editing `profile.json`, the resume PDF, or overrides):

```bash
cd backend
python rag.py     # writes backend/vector_store/ locally
```

You do not commit the result — it rebuilds automatically on the server's next boot, and the admin panel
rebuilds it whenever you hide/unhide a project.

**Why not commit it?** The FAISS `index.pkl` stores the raw text of everything embedded — including
resume content. Committing it to a public repo would expose that text (recoverable via unpickling) even
though the chatbot is instructed not to reveal it. Building at runtime keeps that text out of the repo,
and guarantees a hidden project is never embedded in the first place.

---

## Evaluation & Tests

Three harnesses, three different questions:

```bash
cd backend

# 1) Does the RAG pipeline still answer well? (RAGAS: faithfulness, relevancy,
#    context precision/recall — runs the REAL retrieve→generate path with the
#    semantic cache forced OFF, so similarity hits can't inflate the scores)
python evals/run_ragas_eval.py

# 2) Where should the semantic threshold sit? Sweeps candidate thresholds over
#    labeled paraphrase pairs (evals/semantic_pairs.json) using real embeddings,
#    and reports hits / false positives / false negatives per threshold.
python evals/eval_semantic_threshold.py

# 3) Does the cache behave? Offline unit + API tests with counting fakes — no
#    network, no API key spend. Locks in the paid-call budget of every path
#    (exact hit = 0 calls, semantic hit = 1 embedding, miss = 1 embedding + 1
#    generation with the embedding reused), plus TTL expiry, knowledge-version
#    invalidation, hide/clear privacy, eviction order, and the rollback flag.
pip install -r tests/requirements-test.txt
python -m pytest tests/ -q
```

---

## Deployment

### Backend (Render)

- **Service type**: Web Service (Python)
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment variables** (Render dashboard → **Environment** tab):

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Embeddings + chat completions |
| `GITHUB_TOKEN` | Recommended | Fine-grained token, public-repo read is enough. Without it, GitHub rate-limits Render's shared IP and the project list can come back empty. |
| `ADMIN_TOKEN` | For admin panel | Any long random string; must match what you enter in `admin.html`. |
| `GITHUB_USERNAME` | Optional | Defaults to `tryadityagupta`. |

- **Optional resume PDF**: the PDF is gitignored and therefore not on Render, so the chatbot rebuilds
  from `profile.json` + GitHub only. To include the resume text as well, add it privately via Render's
  **Environment** tab → **Secret Files** (not the Settings page). Because Secret Files hold text, add it
  as a base64 string and decode on boot, or accept that the polished overrides + `profile.json` already
  cover project descriptions and skills.

### Keeping the service warm (cold-start fix)

Render's free tier spins the service **down** after ~15 minutes of no traffic; the next request pays a
cold start while the container boots and the index loads/rebuilds. Fix: point a free uptime monitor at
`/health` every ~10–14 minutes. `/health` makes no OpenAI call, so this costs zero tokens.

Active keep-warm pinger: cron-job.org job #7917947, every 14 min from 7 am to 10 pm on `/health`. This
consumes most of the free tier's ~750 instance-hours/month, so disable it before deploying other free
Render services. Its side benefit: because the instance stays warm, the runtime index rebuild only
happens on real redeploys, not on every cold start.

### Frontend (Vercel + custom domain)

- **Framework Preset**: Other (static — no build step)
- **Root Directory**: `frontend`
- Pushing to the connected Git repo triggers an automatic deploy. `admin.html` deploys alongside
  `index.html` and is reachable at `yourdomain/admin.html`.
- **Custom domain** (`ysadityagupta.co.in`) registered at GoDaddy, DNS pointing at Vercel:
  - `A` record on `@` → `76.76.21.21`
  - `CNAME` on `www` → `cname.vercel-dns.com`

---

## Key Engineering Decisions

### Why sync projects from GitHub instead of hardcoding cards?

The old site duplicated every project in two places — the frontend HTML and the chatbot's data — so
adding or removing one meant editing both by hand. Syncing from GitHub makes "show" the default:
new repos appear on their own, and a single `hidden` list is the only thing you maintain.

### Why does hiding remove a project from the chatbot too?

Hiding is a privacy feature, not just a layout toggle. Because both surfaces read the same filtered list,
a hidden repo is never embedded into the FAISS index, and a live query-time filter drops its chunks even
in the seconds before a rebuild finishes. So a hidden project can't be seen *or* asked about.

### Why not commit the vector store?

See "The Vector Store" above — the committed `index.pkl` would expose the raw embedded text (including
resume PII) in a public repo. Building at runtime avoids that and keeps hidden projects out entirely.

### Why stream the response?

Streaming sends tokens as they're produced, so the first words appear in well under a second. This cuts
**time-to-first-token** dramatically even when total generation time is unchanged.

### Why SSE instead of plain `text/plain` chunks?

The first streaming implementation returned a `StreamingResponse` with `media_type="text/plain"`. The
backend was genuinely yielding per-token, but in production the browser still received the answer in one
burst. `curl --no-buffer --trace-time` against the deployed API confirmed it: the headers said
`Transfer-Encoding: chunked`, yet ~1.1 KB — nearly the whole answer — landed in a single receive event
after ~6.5 seconds. The app code was innocent; the chunks were being **coalesced by the hops between
Uvicorn and the browser** (Render's proxy, Cloudflare), which treat generic text as buffer-and-compress
material.

`text/event-stream` is the content type every proxy and CDN understands as "pass each chunk through
immediately, don't compress, don't buffer". Three headers reinforce it:

| Header | Why |
|---|---|
| `Cache-Control: no-cache, no-transform` | `no-transform` forbids intermediaries from re-encoding or gzipping — gzip requires buffering, which is the whole problem. |
| `Connection: keep-alive` | Keeps the socket open for the life of the stream. |
| `X-Accel-Buffering: no` | Disables response buffering in nginx-style proxies. |

WebSockets would also have solved it, but that's a bidirectional protocol with connection state to manage
for what is a strictly one-way stream — SSE is the smaller tool that fits the actual shape of the problem.

### Why a client-side typewriter queue if the server already streams?

Because `reader.read()` resolving does **not** mean "one token arrived" — it means "one network chunk
arrived", and that chunk may legally contain many tokens. SSE fixes systematic buffering, but no HTTP
layer promises one-frame-per-packet, so appending each chunk directly to the DOM still looks lumpy under
real network conditions.

The frontend therefore pushes received text into a character queue and drains it on a timer (~1 char per
14 ms), decoupling render cadence from network cadence. The drain rate adapts to backlog — 4 chars/tick
past 120 queued, 10 past 400 — so a long answer catches up instead of still typing seconds after the
stream closed. The bot bubble is also created on the *first token* rather than on response headers, so
the "Thinking…" indicator covers retrieval and model latency instead of vanishing into an empty box.

### Why `gpt-4o-mini` instead of a larger / reasoning model?

The task is grounded, single-pass Q&A over a small retrieved context. A small fast model answers this
just as well, far cheaper, and with lower latency; a reasoning model would only add latency.

### Why `AsyncOpenAI` and async similarity search?

The endpoint is `async`; using the async client and async similarity search
(`asimilarity_search_by_vector` when the query vector already exists, `asimilarity_search` otherwise)
lets the server handle concurrent visitors without serializing them behind one another.

### Why does the semantic cache reuse the retrieval embedding?

A cache-miss request needs the query embedded twice conceptually — once to compare against cached
questions, once for dense document retrieval — but both consumers want the *same vector for the same
string*. Generating it once and passing it into `retrieve_docs(query_embedding=...)` halves the
embedding spend and latency of every miss, and makes "exactly one embedding call per request" a
testable invariant (`tests/test_semantic_cache.py` asserts it, down to vector identity).

### Why two FAISS indexes instead of one?

They index different things for different questions: the document index maps *content chunks* to
"what's relevant to this query"; the semantic-cache index maps *past questions* to "have we answered
this before". Mixing them would let cached questions compete with real documents during retrieval.
The cache index is tiny (capped by `SEMANTIC_CACHE_MAX_ENTRIES`), flat (`IndexFlatIP` — exact search,
no training), in-memory, and rebuilt at boot from vectors already stored in the database.

### Why OpenAI embeddings instead of HuggingFace?

`sentence-transformers/all-MiniLM-L6-v2` plus `torch` pushed the install over Render's free-tier limits
and timed out deploys. `text-embedding-3-small` removed all heavyweight dependencies in favor of a light
API call.

### Why lifespan-based vector store loading?

Loading the index at module import (before the app bound to its port) made Render's health check fail
during startup. A FastAPI `lifespan` event lets the app accept requests first, then load or build.

### Why a static frontend?

A single hand-written `index.html` (inline CSS/JS) has no framework runtime to bundle, so it serves
statically with no build step and deploys instantly. The project cards are hydrated at runtime from
`/projects`.

---

## License

MIT