from langchain_core.documents import Document
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTOR_PATH)

    print("Vector DB built successfully")


def load_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
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
