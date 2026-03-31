from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag import load_vector_store
from openai import OpenAI
from contextlib import asynccontextmanager
import os


vector_db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db
    vector_db = load_vector_store()
    yield

app = FastAPI(lifespan=lifespan)

# allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
  "For an AI/ML Engineer with Aditya's background and skills, the industry standard in India is typically around ₹26–30 LPA, depending on the company and role."
 
RULE 3 — Date of joining / notice period questions:
If someone asks when Aditya can join or what his notice period is, say:
"For specific availability and joining timelines, it's best to connect directly with Aditya at adityagupta.nits2@gmail.com — he'll be happy to discuss."
 
RULE 4 — Aditya's experience at Optum:
Never describe Aditya as a QA engineer or say he did manual testing, test cases, or regression testing.
If asked about his work at Optum or his day-to-day, always frame it like this:
"At Optum, Aditya worked on building Python-based automation frameworks and data pipelines that helped teams validate large-scale healthcare data workflows. His day-to-day involved writing Python scripts, designing reusable automation utilities, and integrating them with backend services and databases to ensure data consistency across systems. Over time he became more interested in building intelligent systems, which is why he started focusing on machine learning and LLM-based applications — including a RAG-powered portfolio chatbot he built and deployed using FastAPI, LangChain, and FAISS."
 
Always use these words when talking about Optum work:
✅ automation frameworks  ✅ Python tooling  ✅ validation systems  ✅ data workflows  ✅ backend integration
Never use: ❌ testing  ❌ test cases  ❌ regression testing  ❌ QA engineer  ❌ manual testing
 
--- GENERAL TONE ---
- Be warm, professional, and concise (2–5 sentences unless more detail is clearly needed).
- If a recruiter is asking, sound like Aditya's advocate — highlight his strengths naturally.
- Never make up facts. If something isn't in the context, say you don't have that detail and suggest they email Aditya.
"""


@app.post("/chat")
async def chat(req: ChatRequest):

    if vector_db is None:
        return {"response": "Service is still starting up. Please try again in a moment."}

    query = req.message

    docs = vector_db.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are an AI assistant answering questions about Aditya Gupta.

Context about Aditya Gupta:
{context}

Question:
{query}

Answer based on the context and the rules in your system instructions:
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=250,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        response = completion.choices[0].message.content
        return {"response": response}
    except Exception as e:
        return {"response": "Sorry, I couldn't process that right now. Please try again later."}
