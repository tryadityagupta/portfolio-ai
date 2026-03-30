from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag import load_vector_store
from openai import OpenAI
import os

app = FastAPI()

# allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_db = load_vector_store()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):

    query = req.message

    docs = vector_db.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are an AI assistant answering questions about Aditya Gupta.

Context:
{context}

Question:
{query}

Answer concisely:
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer questions about Aditya Gupta."},
            {"role": "user", "content": prompt}
        ]
    )

    response = completion.choices[0].message.content

    return {"response": response}
