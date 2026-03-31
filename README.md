# Portfolio AI — RAG-Powered Personal Chatbot

A production-deployed AI assistant built on top of a personal portfolio website. Visitors can chat with an AI that answers questions about Aditya Gupta's skills, experience, and projects — powered by a FastAPI backend using Retrieval-Augmented Generation (RAG) with OpenAI embeddings and GPT-4o-mini.

Live demo: [https://portfolio-ai-qoer.onrender.com](https://portfolio-ai-qoer.onrender.com)

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
│   ├── main.py                   # FastAPI app with lifespan-based vector store loading
│   ├── rag.py                    # Embedding, vector store build/load, profile doc loader
│   ├── requirements.txt          # Python dependencies
│   └── test_openai.py            # Quick OpenAI connectivity test
├── frontend/
│   ├── src/
│   │   ├── assets/               # Static assets
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── index.html                # Main portfolio page with embedded chat widget
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
└── .gitignore
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, Vanilla JS (chat widget) + React + TypeScript + Vite |
| Backend | FastAPI (Python) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (CPU) |
| LLM | GPT-4o-mini |
| RAG Framework | LangChain |
| Deployment | Render (free tier) |

---

## How It Works

1. **At build time** — `rag.py` loads the resume PDF and `profile.json`, splits them into chunks, embeds each chunk using OpenAI's `text-embedding-3-small` model, and saves the resulting FAISS index locally.

2. **At server startup** — the FastAPI lifespan event loads the pre-built FAISS vector store into memory. This happens after the app binds to its port, so it does not block Render's health check.

3. **At query time** — the user's message is embedded, the top-3 most relevant chunks are retrieved from FAISS, and these are injected as context into a GPT-4o-mini prompt. The model answers only about Aditya.

---

## Local Setup

### Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Build the vector store (only needed once, or after updating the PDF/profile)
python rag.py

# Run the server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Test OpenAI connectivity

```bash
python test_openai.py
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| POST | `/chat` | Chat endpoint — accepts `{"message": "..."}`, returns `{"response": "..."}` |

### Example

```bash
curl -X POST https://portfolio-ai-qoer.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are Aditya's skills?"}'
```

---

## Rebuilding the Vector Store

If you update `Aditya_Gupta_AI_ML.pdf` or `data/profile.json`, you need to rebuild and re-commit the vector store:

```bash
cd backend
python rag.py
git add vector_store/
git commit -m "Rebuild vector store"
git push
```

---

## Deployment (Render)

- **Service type**: Web Service (Python)
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment variable**: `OPENAI_API_KEY` set in Render dashboard → Environment tab

> **Note**: The free tier spins down after inactivity, which can delay the first request by ~50 seconds. The `/health` endpoint helps Render detect when the service is ready.

---

## Key Engineering Decisions

### Why OpenAI Embeddings instead of HuggingFace?

The original implementation used `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace). This caused Render deployments to time out because the model download (~400 MB) plus `transformers`, `accelerate`, and `torch` pushed the total install size over 1 GB — well beyond Render's free tier memory and build time limits. Switching to `OpenAIEmbeddings(model="text-embedding-3-small")` eliminated all heavyweight dependencies and replaced the local model download with a lightweight API call.

### Why lifespan-based vector store loading?

The original code ran `vector_db = load_vector_store()` at module import time, before the app bound to its port. On Render, this meant the health check failed during startup because the app wasn't listening yet. Moving it into a FastAPI `@asynccontextmanager lifespan` event ensures the app starts accepting requests first, then loads the vector store.

### Why commit the vector store to git?

The FAISS index is pre-built locally and committed alongside the code. This means Render never needs to run embeddings at deploy time — it just loads the existing index from disk. Avoids unnecessary OpenAI API calls on every deploy and keeps startup fast.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for both embeddings and chat completions |

---

## License

MIT
