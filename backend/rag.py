"""
rag.py
------
Responsible for:
- Loading documents
- Chunking text
- Creating embeddings
- Vector search (FAISS)
"""

import os
import faiss
import numpy as np
import pickle
from pypdf import PdfReader
from openai import OpenAI

# ----------------------------------
# OpenAI client (embeddings only)
# ----------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------
# Config
# ----------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500  # characters

# ----------------------------------
# In-memory stores
# ----------------------------------
documents: list[dict] = []
index = None

# ----------------------------------
# Persistence paths
# ----------------------------------
RAG_STORE_PATH = "rag_store"
INDEX_PATH = f"{RAG_STORE_PATH}/index.faiss"
DOCS_PATH = f"{RAG_STORE_PATH}/documents.pkl"

# ----------------------------------
# Text chunking
# ----------------------------------
def chunk_text(text: str) -> list[str]:
    chunks = []
    current = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if len(current) + len(line) > CHUNK_SIZE:
            chunks.append(current.strip())
            current = line
        else:
            current += " " + line

    if current.strip():
        chunks.append(current.strip())

    return chunks

# ----------------------------------
# Loaders
# ----------------------------------
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(path: str):
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append((i + 1, text))

    return pages

# ----------------------------------
# Document ingestion
# ----------------------------------
def load_documents(folder_path="documents"):
    global documents
    documents = []

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)

        if filename.endswith(".txt"):
            text = load_txt(path)
            for chunk in chunk_text(text):
                documents.append({
                    "text": chunk,
                    "source": filename,
                    "page": None
                })

        elif filename.endswith(".pdf"):
            for page_num, page_text in load_pdf(path):
                for chunk in chunk_text(page_text):
                    documents.append({
                        "text": chunk,
                        "source": filename,
                        "page": page_num
                    })

    print(f"📄 Loaded {len(documents)} clean document chunks")

# ----------------------------------
# Embeddings + FAISS (FIXED)
# ----------------------------------
def create_index():
    global index, documents

    # ✅ sanitize inputs
    clean_docs = []
    texts = []

    for doc in documents:
        text = doc["text"].strip()
        if text:
            texts.append(text)
            clean_docs.append(doc)

    if not texts:
        raise RuntimeError("❌ No valid text chunks to embed")

    documents = clean_docs

    print(f"🧠 Creating embeddings for {len(texts)} chunks")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    vectors = np.array(
        [item.embedding for item in response.data],
        dtype="float32"
    )

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    print("✅ FAISS index built")

# ----------------------------------
# Persistence
# ----------------------------------
def save_index():
    os.makedirs(RAG_STORE_PATH, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("💾 RAG index saved")

# ----------------------------------
# Retrieval
# ----------------------------------
def retrieve_context(query: str, k: int = 3):
    query_vec = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    ).data[0].embedding

    distances, indices = index.search(
        np.array([query_vec], dtype="float32"),
        k
    )

    chunks = []
    sources = []

    for idx in indices[0]:
        doc = documents[idx]
        chunks.append(doc["text"])
        sources.append({
            "file": doc["source"],
            "page": doc["page"]
        })

    print("🔎 Retrieved:", {s["file"] for s in sources})

    return chunks, sources

# ----------------------------------
# Initialization (ALWAYS REBUILD)
# ----------------------------------
def initialize_rag():
    """
    WHY:
    - Guarantees new uploads are searchable
    - Prevents stale FAISS state
    """
    load_documents()
    create_index()
    save_index()
