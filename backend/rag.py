from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import json
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()


VECTOR_PATH = "vector_store"
PDF_PATH = "Aditya_Gupta_AI_ML.pdf"


def build_vector_store():

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    profile_docs = load_profile_documents()  # ADD THIS
    chunks.extend(profile_docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    import time

    vectorstore = None

    for attempt in range(5):
        try:
            print(f"Embedding attempt {attempt+1}...")

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            break

        except Exception as e:
            print("Embedding failed:", e)
            time.sleep(5)

    if vectorstore is None:
        raise RuntimeError(
            "Failed to generate embeddings after multiple attempts")

    vectorstore.save_local(VECTOR_PATH)

    print("Vector DB built successfully")


def load_vector_store():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    return FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def load_profile_documents():
    with open("data/profile.json", "r", encoding="utf-8") as f:
        profile = json.load(f)

    docs = []

    for key, value in profile.items():
        docs.append(Document(page_content=f"{key}: {value}"))

    return docs


if __name__ == "__main__":
    build_vector_store()
