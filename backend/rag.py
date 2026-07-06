from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import json
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import time

import projects  # our single source of truth (GitHub + hide/override rules)

load_dotenv()


# Resolve every path relative to THIS file, not the working directory, so it
# behaves the same locally and on Render regardless of how uvicorn is launched.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_PATH = os.path.join(_BASE_DIR, "vector_store")
PDF_PATH = os.path.join(_BASE_DIR, "Aditya_Gupta_AI_ML.pdf")
PROFILE_PATH = os.path.join(_BASE_DIR, "data", "profile.json")


def build_vector_store():

    chunks = []

    # 1) Resume PDF (optional). If it's missing we just skip it — the site
    #    still works from GitHub + profile data. NOTE: the PDF is its own
    #    source of truth. If you describe a project in the resume, the chatbot
    #    can mention it from here even if the repo is hidden. Keep the resume
    #    limited to work you're happy to be public.
    if os.path.exists(PDF_PATH):
        pdf_docs = PyPDFLoader(PDF_PATH).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50)
        for c in splitter.split_documents(pdf_docs):
            c.metadata = {**(c.metadata or {}), "source": "resume"}
            chunks.append(c)

    # 2) Profile facts (skills, education, etc.) — but NOT the old static
    #    'projects' list. Projects now come live from GitHub in step 3.
    chunks.extend(load_profile_documents())

    # 3) Projects — built from the SAME filtered list the frontend uses.
    #    Hidden repos are already gone by the time we get here, so they are
    #    never embedded and the chatbot literally has no memory of them.
    chunks.extend(load_project_documents())

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = None

    for attempt in range(5):
        try:
            print(f"Embedding attempt {attempt+1}...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            break
        except Exception as e:
            print("Embedding failed:", e)
            time.sleep(5)

    if vectorstore is None:
        raise RuntimeError(
            "Failed to generate embeddings after multiple attempts")

    vectorstore.save_local(VECTOR_PATH)
    print(f"Vector DB built successfully ({len(chunks)} chunks)")


def load_vector_store():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # If no prebuilt index exists (e.g. first boot on a fresh server because we
    # no longer commit the index), build it now from live data.
    if not os.path.exists(os.path.join(VECTOR_PATH, "index.faiss")):
        print("No vector store found — building one from GitHub + profile...")
        build_vector_store()

    return FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def load_profile_documents():
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        profile = json.load(f)

    docs = []
    for key, value in profile.items():
        # Skip the legacy static projects list — GitHub is the source now.
        if key == "projects":
            continue
        if isinstance(value, list):
            text = f"{key}: " + ", ".join(value)
        else:
            text = f"{key}: {value}"
        docs.append(Document(page_content=text,
                    metadata={"source": "profile"}))

    return docs


def load_project_documents():
    """
    Turn each VISIBLE GitHub project into a document the chatbot can retrieve.
    Every chunk is tagged with metadata['repo'] so it can also be filtered at
    query time (see main.py) — a second safety net on top of not embedding
    hidden repos in the first place.
    """
    docs = []
    for p in projects.get_visible_projects(force_refresh=True, include_readme=True):
        parts = [
            f"Project: {p['name']}",
            f"Category: {p['type']}",
            f"Tech: {', '.join(p['tech'])}" if p.get("tech") else "",
            f"Description: {p['description']}" if p.get("description") else "",
            f"GitHub: {p['url']}" if p.get("url") else "",
            f"Live: {p['homepage']}" if p.get("homepage") else "",
        ]
        readme = p.get("readme") or ""
        if readme:
            parts.append(f"README excerpt:\n{readme}")

        text = "\n".join(x for x in parts if x)
        docs.append(Document(
            page_content=text,
            metadata={"source": "project", "repo": p["repo"]},
        ))
    return docs


if __name__ == "__main__":
    build_vector_store()
