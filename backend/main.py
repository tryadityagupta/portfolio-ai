from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag import load_vector_store

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


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):

    query = req.message

    docs = vector_db.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])

    response = f"Based on Aditya's resume:\n{context}"

    return {"response": response}
