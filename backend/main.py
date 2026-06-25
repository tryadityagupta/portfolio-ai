from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from rag import load_vector_store
import os


vector_db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE, right after the server starts listening on its port.
    # We load the FAISS index into memory here (not at import time) so the
    # /health check can answer first and Render doesn't mark the deploy as failed.
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

# AsyncOpenAI = the non-blocking client. We can "await" its calls, so while we
# wait on OpenAI's network response the server is free to handle other users.
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health():
    return {"status": "ok"}


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

    # If a user arrives during the few seconds of cold start, the index isn't
    # loaded yet. We still answer as a text stream so the frontend reads every
    # reply the same way.
    if vector_db is None:
        async def not_ready():
            yield "Service is still starting up. Please try again in a moment."
        return StreamingResponse(not_ready(), media_type="text/plain")

    # 1) RETRIEVAL — embed the question and pull the top-3 most relevant chunks.
    #    asimilarity_search is the async version: the query-embedding call to
    #    OpenAI happens without blocking the event loop.
    docs = await vector_db.asimilarity_search(req.message, k=3)
    context = "\n".join(d.page_content for d in docs)

    # 2) AUGMENT — stuff that retrieved context into the prompt (the "A" in RAG).
    prompt = build_prompt(context, req.message)

    # 3) GENERATION — stream tokens back to the browser as the model produces
    #    them, instead of waiting for the whole answer. This is what makes the
    #    reply start appearing almost immediately (low time-to-first-token).
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
