from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
import json
import os

from rag import load_vector_store, build_vector_store
import projects


vector_db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE, right after the server starts listening on its port.
    # We load (or build) the FAISS index here so /health can answer first and
    # Render doesn't mark the deploy as failed during a cold start.
    global vector_db
    vector_db = load_vector_store()
    yield


app = FastAPI(lifespan=lifespan)

# CORS: lets the browser frontend (a different domain) call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


SYSTEM_PROMPT = """
You are an AI assistant on Aditya Gupta's personal portfolio website.
Your job is to answer questions about Aditya in a professional, friendly, and confident tone.
Answer only about Aditya. If asked anything completely unrelated to him, politely redirect.
 
--- STRICT RULES (always follow these, they override the context) ---
 
RULE 1 — NEVER reveal Aditya's mobile number under any circumstance.
If someone asks for his phone number or contact number, say:
"I'm not able to share Aditya's phone number here. You can reach him at adityagupta.nits2@gmail.com or connect on LinkedIn."
 
RULE 2 — CTC / salary questions:
- If someone asks about Aditya's current CTC or expected CTC, say:
  "That's something best discussed directly with Aditya. Feel free to reach out to him at adityagupta.nits2@gmail.com — he'd be happy to connect."
- If someone asks what the industry standard CTC is for someone like Aditya, say:
  "For an AI/ML Engineer with Aditya's background and skills, the industry standard in India is typically around a competitive range, depending on the company and role."
 
RULE 3 — Date of joining / notice period questions:
If someone asks when Aditya can join or what his notice period is, say:
"For specific availability and joining timelines, it's best to connect directly with Aditya at adityagupta.nits2@gmail.com — he'll be happy to discuss."
 
RULE 4 — Aditya's experience at Optum:

If asked about his work at Optum or his day-to-day, always frame it like this:
"At Optum, Aditya worked on building Python-based automation frameworks and data pipelines that helped teams validate large-scale healthcare data workflows. His day-to-day involved writing Python scripts, designing reusable automation utilities, and integrating them with backend services and databases to ensure data consistency across systems. Over time he became more interested in building intelligent systems, which is why he started focusing on machine learning and LLM-based applications — including a RAG-powered portfolio chatbot he built and deployed using FastAPI, LangChain, and FAISS."
 



 
--- GENERAL TONE ---
- Be warm, professional, and concise (2–5 sentences unless more detail is clearly needed).
- If a recruiter is asking, sound like Aditya's advocate — highlight his strengths naturally.
- Never make up facts. If something isn't in the context, say you don't have that detail and suggest they email Aditya.
"""


def build_prompt(context: str, question: str) -> str:
    return f"""
You are an AI assistant answering questions about Aditya Gupta.

Context about Aditya Gupta:
{context}

Question:
{question}

Answer based on the context and the rules in your system instructions:
"""


@app.post("/chat")
async def chat(req: ChatRequest):

    if vector_db is None:
        async def not_ready():
            yield "Service is still starting up. Please try again in a moment."
        return StreamingResponse(not_ready(), media_type="text/plain")

    # 1) RETRIEVAL. We pull a few EXTRA chunks (k=6) then drop any that belong
    #    to a currently-hidden repo, and keep the best 3. This is a live safety
    #    net: even if the index was built before you hid something, the hidden
    #    project's text can't reach the model on this request. Compared
    #    case-insensitively because repo names on GitHub can be any case.
    hidden = projects.get_hidden_set()  # lowercased
    docs = await vector_db.asimilarity_search(req.message, k=6)
    visible_docs = [
        d for d in docs
        if (d.metadata.get("repo") or "").lower() not in hidden
    ][:3]
    context = "\n".join(d.page_content for d in visible_docs)

    # 2) AUGMENT — stuff the retrieved context into the prompt (the "A" in RAG).
    prompt = build_prompt(context, req.message)

    # 3) GENERATION — stream tokens back to the browser as they're produced.
    async def token_stream():
        try:
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=250,
                stream=True,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception:
            yield "Sorry, I couldn't process that right now. Please try again later."

    return StreamingResponse(token_stream(), media_type="text/plain")


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
    return {"ok": True, "hidden": data["hidden"],
            "note": "Project is public again. Commit project_overrides.json to persist."}
