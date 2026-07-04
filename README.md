# Portfolio AI — RAG-Powered Personal Chatbot

A production-deployed AI assistant on top of a personal portfolio website. Visitors chat with an
assistant that answers questions about Aditya Gupta's skills, experience, and projects — powered by a
FastAPI backend using Retrieval-Augmented Generation (RAG) with OpenAI embeddings and `gpt-4o-mini`.
Answers are **streamed token-by-token**, so replies start appearing almost instantly.

- **Live site:** [https://ysadityagupta.co.in](https://ysadityagupta.co.in) (static frontend on Vercel)
- **API:** [https://portfolio-ai-qoer.onrender.com](https://portfolio-ai-qoer.onrender.com) (FastAPI backend on Render)

---

## Project Structure

```
PORTFOLIO-AI/
├── backend/
│   ├── data/
│   │   └── profile.json          # Structured profile data for RAG
│   ├── vector_store/
│   │   ├── index.faiss           # FAISS vector index (pre-built, committed to repo)
│   │   └── index.pkl             # FAISS metadata
│   ├── .env                      # Local environment variables (not committed)
│   ├── Aditya_Gupta_AI_ML.pdf    # Source resume/document for RAG
│   ├── main.py                   # FastAPI app: lifespan loading, async + streaming /chat
│   ├── rag.py                    # Embedding, vector store build/load, profile doc loader
│   └── requirements.txt          # Python dependencies
├── frontend/
│   └── index.html                # Single-page portfolio + streaming chat widget (static)
└── .gitignore
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Static HTML, CSS, vanilla JS (single-page portfolio + streaming chat widget) |
| Backend | FastAPI (Python), async endpoints |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (CPU) |
| LLM | OpenAI `gpt-4o-mini` (streaming) |
| RAG Framework | LangChain |
| Deployment | Backend on Render · Frontend on Vercel (custom domain via GoDaddy DNS) |

---

## How It Works

1. **At build time** — `rag.py` loads the resume PDF and `profile.json`, splits them into chunks, embeds
   each chunk using OpenAI's `text-embedding-3-small`, and saves the resulting FAISS index locally.

2. **At server startup** — a FastAPI `lifespan` event loads the pre-built FAISS index into memory. This
   runs *after* the app binds to its port, so the `/health` check passes before this heavy I/O begins.

3. **At query time:**
   - **Retrieval** — the user's message is embedded and the top-3 most relevant chunks are pulled from
     FAISS (`asimilarity_search`, the async variant).
   - **Augment** — those chunks are injected as context into the prompt (the "A" in RAG).
   - **Generation** — `gpt-4o-mini` answers, and tokens are **streamed** back to the browser via a
     `StreamingResponse`. The chat widget appends each piece to the bubble as it arrives.

The non-blocking `AsyncOpenAI` client is used throughout, so a slow OpenAI call for one visitor doesn't
freeze the server for others (concurrency).

---

## Local Setup

### Backend

```bash
cd backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

echo "OPENAI_API_KEY=sk-..." > .env

# Build the vector store (once, and after any change to the PDF or profile.json)
python rag.py

# Run the server
uvicorn main:app --reload
```

The API runs at `http://localhost:8000`. The `/chat` endpoint returns a **plain-text stream**, so test
it with `--no-buffer` to watch it arrive live:

```bash
curl http://localhost:8000/health

curl --no-buffer -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What projects has Aditya built?"}'
```

### Frontend

The frontend is a single static `index.html` — no build step:

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500`. By default the chat widget calls the deployed Render API. To test against a
**local** backend, temporarily change the `fetch(...)` URL in `index.html` from the `onrender.com` address
to `http://localhost:8000/chat`, then change it back before deploying.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check — returns `{"status": "ok"}`. Also used by the keep-warm pinger. |
| POST | `/chat` | Accepts `{"message": "..."}`, returns a **streaming plain-text** response. |

---

## Rebuilding the Vector Store

If you update `Aditya_Gupta_AI_ML.pdf` or `data/profile.json`, rebuild and re-commit the index — the
chatbot reads the FAISS index, **not** the JSON/PDF directly:

```bash
cd backend
python rag.py
git add vector_store/
git commit -m "Rebuild vector store"
git push
```

---

## Deployment

### Backend (Render)

- **Service type**: Web Service (Python)
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment variable**: `OPENAI_API_KEY` set in Render dashboard → Environment tab

### Keeping the service warm (cold-start fix)

Render's free tier spins the service **down** after ~15 minutes of no traffic. The next request then pays
a **cold start** (~15–20s) while the container boots and the FAISS index reloads into memory. This delay
is from Render, not the model — the LLM choice does not affect it.

Fix: point a free uptime monitor (e.g. UptimeRobot or cron-job.org) at `/health` every ~10 minutes so the
service never goes idle. `/health` makes **no OpenAI call**, so this costs **zero tokens** — it only uses
Render compute hours. (The free tier's ~750 instance-hours/month is roughly one always-on service, so
there's little headroom left for other free services.) The paid Starter tier never spins down if the
keep-warm approach isn't enough.

Active keep-warm pinger: cron-job.org job #7917947, every 14 min from 7 am to 10 pm on /health — this consumes most of the render free tier workspace's 750 free instance-hours, so disable it before deploying other free Render services

### Frontend (Vercel + custom domain)

- **Framework Preset**: Other (static — no build step)
- **Root Directory**: `frontend`
- Pushing to the connected Git repo triggers an automatic deploy.
- **Custom domain** (`ysadityagupta.co.in`) registered at GoDaddy, DNS pointing at Vercel:
  - `A` record on `@` → `76.76.21.21`
  - `CNAME` on `www` → `cname.vercel-dns.com`

---

## Key Engineering Decisions

### Why stream the response?

Previously the endpoint waited for the *entire* completion and returned one JSON blob, so the user stared
at a "Thinking…" placeholder for the full generation. Streaming sends tokens as they're produced, so the
first words appear in well under a second. This cuts **perceived latency** dramatically even when total
generation time is unchanged — the metric that improves is **time-to-first-token (TTFT)**.

### Why `gpt-4o-mini` instead of a larger / reasoning model?

The task is grounded, single-pass Q&A over a small retrieved context — not open-ended reasoning. A small
fast model answers this just as well, far cheaper, and with lower latency. A larger reasoning model would
*add* latency (it thinks before answering) for no quality gain here.

### Why `AsyncOpenAI` and `asimilarity_search`?

The endpoint is `async`, but the old code called the **synchronous** OpenAI client, which blocked the
event loop and forced requests to queue one behind another. The async client and async similarity search
let the server handle concurrent visitors without serializing them. (This improves **concurrency**, not
the latency of a single request.)

### Why OpenAI embeddings instead of HuggingFace?

The original used `sentence-transformers/all-MiniLM-L6-v2`, whose model download (~400 MB) plus `torch`/
`transformers`/`accelerate` pushed the install over Render's free-tier limits and timed out deploys.
Switching to `text-embedding-3-small` removed all heavyweight dependencies in favor of a light API call.

### Why lifespan-based vector store loading?

Loading the index at module import (before the app bound to its port) made Render's health check fail
during startup. Moving it into a FastAPI `lifespan` event lets the app accept requests first, then load.

### Why commit the vector store to git?

The FAISS index is pre-built locally and committed with the code, so Render just loads it from disk —
no embeddings at deploy time, no extra API calls, faster startup.

### Why a static frontend?

A single hand-written `index.html` (inline CSS/JS) has no framework runtime to bundle, so it serves
statically with no build step and deploys instantly.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for both embeddings and chat completions |

---

## License

MIT