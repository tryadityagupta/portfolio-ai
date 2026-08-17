from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import json
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import time
import base64
import tempfile

import projects  # our single source of truth (GitHub + hide/override rules)

load_dotenv()


# Resolve every path relative to THIS file, not the working directory, so it
# behaves the same locally and on Render regardless of how uvicorn is launched.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_PATH = os.path.join(_BASE_DIR, "vector_store")
PDF_PATH = os.path.join(_BASE_DIR, "Aditya_Gupta_AI_ML.pdf")
PROFILE_PATH = os.path.join(_BASE_DIR, "data", "profile.json")

# Render secret file (base64 text)
RESUME_B64_PATH = "/etc/secrets/resume_b64.txt"

# ONE embedding model name for the whole app. The semantic cache stamps each
# stored vector with this string and refuses vectors from any other model at
# load time — cosine similarity between vectors from different models is
# meaningless, so a model swap must quietly restart the cache, not corrupt it.
EMBEDDING_MODEL = "text-embedding-3-small"

_embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    """The shared OpenAIEmbeddings client. Built once, reused everywhere:
    document store build/load AND per-request query embeddings (main.py).
    One object = one underlying HTTP client with connection pooling, instead
    of a new client per call — and one place where the model is chosen."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    return _embeddings


def _resolve_resume_pdf():
    """Return the path to a readable resume PDF, or None.

    Local dev: the real PDF sits next to this file (never committed).
    Render: Secret Files are plaintext-only — a raw PDF uploaded there
    gets corrupted — so the resume travels as base64 text and is decoded
    back into a real PDF here at startup.
    """
    if os.path.exists(PDF_PATH):
        return PDF_PATH

    if os.path.exists(RESUME_B64_PATH):
        decoded_path = os.path.join(tempfile.gettempdir(), "resume.pdf")
        with open(RESUME_B64_PATH, "r", encoding="utf-8") as f:
            pdf_bytes = base64.b64decode(f.read())
        with open(decoded_path, "wb") as out:
            out.write(pdf_bytes)
        return decoded_path

    return None


def build_vector_store():

    chunks = []

    # 1) Resume (optional). Missing OR unreadable → skip it — the site still
    #    works from GitHub + profile data, and a bad resume file must never
    #    take the whole service down again. NOTE: the PDF is its own source
    #    of truth; keep it limited to work you're happy to be public.
    resume_path = _resolve_resume_pdf()
    if resume_path:
        try:
            pdf_docs = PyPDFLoader(resume_path).load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50)
            resume_chunks = splitter.split_documents(pdf_docs)
            for c in resume_chunks:
                c.metadata = {**(c.metadata or {}), "source": "resume"}
                chunks.append(c)
            print(f"Resume indexed ({len(resume_chunks)} chunks)")
        except Exception as e:
            print(f"WARNING: could not parse resume PDF — skipping it. {e}")

    # 2) Profile facts (skills, education, etc.) — but NOT the old static
    #    'projects' list. Projects now come live from GitHub in step 3.
    chunks.extend(load_profile_documents())

    # 3) Projects — built from the SAME filtered list the frontend uses.
    #    Hidden repos are already gone by the time we get here, so they are
    #    never embedded and the chatbot literally has no memory of them.
    chunks.extend(load_project_documents())

    embeddings = get_embeddings()   # shared client — see top of file

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

    embeddings = get_embeddings()   # shared client — see top of file

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
    Turn each VISIBLE GitHub project into documents the chatbot can retrieve.

    Two kinds of docs per project:
      1) one small OVERVIEW doc (name, category, tech, description, links) —
         serves "what has he built?" style questions, and
      2) the README split into ~700-char CHUNKS, each prefixed with a
         [Project: name] tag so a chunk from deep inside a README still
         self-identifies which project it came from.

    Why chunk at all: one embedding per whole README is an AVERAGE of
    everything in it (setup steps, endpoints, architecture tables...), so a
    pointed question like "has he done fine-tuning?" barely moves the needle
    against it, and the doc loses to short ML-flavoured profile snippets.
    Small chunks give the fine-tuning paragraph its own sharp embedding.

    Every chunk is tagged with metadata['repo'] so it can also be filtered at
    query time (see main.py) — a second safety net on top of not embedding
    hidden repos in the first place.
    """
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)

    for p in projects.get_visible_projects(force_refresh=True, include_readme=True):
        meta = {"source": "project", "repo": p["repo"]}

        overview = "\n".join(x for x in [
            f"Project: {p['name']}",
            f"Category: {p['type']}",
            f"Tech: {', '.join(p['tech'])}" if p.get("tech") else "",
            f"Description: {p['description']}" if p.get("description") else "",
            f"GitHub: {p['url']}" if p.get("url") else "",
            f"Live: {p['homepage']}" if p.get("homepage") else "",
        ] if x)
        docs.append(Document(page_content=overview, metadata=meta))

        readme = p.get("readme") or ""
        if readme:
            tag = f"[Project: {p['name']}]"
            for piece in splitter.split_text(readme):
                docs.append(Document(page_content=f"{tag}\n{piece}",
                                     metadata=meta))
    return docs


if __name__ == "__main__":
    build_vector_store()
