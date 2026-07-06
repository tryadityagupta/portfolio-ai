# Portfolio AI — RAG-Powered Personal Chatbot with GitHub-Synced Projects

A production-deployed AI assistant on top of a personal portfolio website. Visitors chat with an
assistant that answers questions about Aditya Gupta's skills, experience, and projects — powered by a
FastAPI backend using Retrieval-Augmented Generation (RAG) with OpenAI embeddings and `gpt-4o-mini`.
Answers are **streamed token-by-token**, so replies start appearing almost instantly.

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
│   ├── vector_store/                 # FAISS index — BUILT AT RUNTIME, NOT committed (gitignored)
│   ├── .env                          # Local secrets (not committed)
│   ├── Aditya_Gupta_AI_ML.pdf        # Optional resume source for RAG (not committed)
│   ├── github_sync.py                # Fetches repos + READMEs from the GitHub REST API
│   ├── projects.py                   # Single source of truth: merges GitHub + overrides, applies hiding
│   ├── main.py                       # FastAPI app: /chat, /projects, /admin/* , streaming, lifespan
│   ├── rag.py                        # Embedding, vector store build/load, profile + project doc loaders
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
| Frontend | Static HTML, CSS, vanilla JS (portfolio + streaming chat widget + admin panel) |
| Backend | FastAPI (Python), async endpoints |
| Project sync | GitHub REST API (via `urllib`, standard library) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (CPU) |
| LLM | OpenAI `gpt-4o-mini` (streaming) |
| RAG Framework | LangChain |
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

### The chat request

1. **Retrieval** — the user's message is embedded and the top matches are pulled from FAISS
   (`asimilarity_search`, the async variant). We fetch a few extra and drop any chunk whose `repo`
   metadata is currently hidden, then keep the best 3 — a live safety net so a just-hidden project can't
   surface even before the index rebuilds.
2. **Augment** — those chunks are injected as context into the prompt (the "A" in RAG).
3. **Generation** — `gpt-4o-mini` answers, streamed back to the browser via a `StreamingResponse`.

The non-blocking `AsyncOpenAI` client is used throughout, so a slow OpenAI call for one visitor doesn't
freeze the server for others (concurrency).

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
```

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

---

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Health check — returns `{"status": "ok"}`. Also used by the keep-warm pinger. |
| GET | `/projects` | — | The visible (hidden-filtered) project list the frontend renders. |
| POST | `/chat` | — | Accepts `{"message": "..."}`, returns a **streaming plain-text** response. |
| GET | `/admin/projects` | `X-Admin-Token` | Every owned repo + whether it's currently hidden (powers the toggles). |
| POST | `/admin/hide` | `X-Admin-Token` | Body `{"repo": "..."}` — hides a repo, rebuilds the index. |
| POST | `/admin/unhide` | `X-Admin-Token` | Body `{"repo": "..."}` — un-hides a repo, rebuilds the index. |

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

### Why `gpt-4o-mini` instead of a larger / reasoning model?

The task is grounded, single-pass Q&A over a small retrieved context. A small fast model answers this
just as well, far cheaper, and with lower latency; a reasoning model would only add latency.

### Why `AsyncOpenAI` and `asimilarity_search`?

The endpoint is `async`; using the async client and async similarity search lets the server handle
concurrent visitors without serializing them behind one another.

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