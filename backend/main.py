"""
main.py
-------
This file is responsible for:
- API endpoints
- Session handling
- Chat memory
- Calling OpenAI chat
- Injecting RAG context
- Streaming responses
"""

# --------------------------------------------------
# 1️⃣ ENVIRONMENT LOADING (MUST BE FIRST)
# --------------------------------------------------
# WHY:
# - Python executes imports top-down
# - rag.py creates OpenAI client at import time
# - So the API key MUST exist before importing rag
import os
from dotenv import load_dotenv

load_dotenv()  # Loads OPENAI_API_KEY into process env

# Hard fail early if missing (prevents silent bugs)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env file.")

# --------------------------------------------------
# 2️⃣ STANDARD IMPORTS
# --------------------------------------------------
import asyncio
import shutil
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

# --------------------------------------------------
# 3️⃣ RAG IMPORTS (SAFE NOW)
# --------------------------------------------------
# WHY:
# - Env is loaded
# - OpenAI client creation inside rag.py is now safe
from rag import initialize_rag, retrieve_context

# --------------------------------------------------
# 4️⃣ FASTAPI APP
# --------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # WHY: allow frontend (localhost:5500)
    allow_credentials=True,
    allow_methods=["*"],  # WHY: allow OPTIONS, POST, GET
    allow_headers=["*"],  # WHY: allow Content-Type, Authorization
)
# --------------------------------------------------
# 5️⃣ OPENAI CLIENT
# --------------------------------------------------
# WHY:
# - Used for chat completions + streaming
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------
# 6️⃣ SESSION STORE
# --------------------------------------------------
# WHY:
# - session_id → list of messages
# - prevents different users from mixing conversations
sessions: Dict[str, List[Dict[str, str]]] = {}

# --------------------------------------------------
# 7️⃣ REQUEST / RESPONSE MODELS
# --------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str


class Source(BaseModel):
    file: str
    page: int | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: List[Source]

# --------------------------------------------------
# 8️⃣ APP STARTUP (LOAD RAG ONCE)
# --------------------------------------------------
@app.on_event("startup")
def startup():
    """
    WHY:
    - Load documents
    - Create embeddings
    - Build FAISS index
    - Only once (not per request)
    """
    initialize_rag()

# --------------------------------------------------
# 9️⃣ HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# 🔟 STREAMING HELPER
# --------------------------------------------------
async def stream_openai_response(prompt: str):
    """
    WHY:
    - Streams tokens as they arrive
    - Keeps frontend responsive
    """

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content
            await asyncio.sleep(0)

# --------------------------------------------------
# 1️⃣1️⃣ NORMAL CHAT ENDPOINT
# --------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    WHY:
    - Non-streaming chat
    - Returns full answer + citations
    """

    context_chunks, sources = retrieve_context(request.message)
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context_text}

Question:
{request.message}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "reply": completion.choices[0].message.content,
        "sources": sources
    }

# --------------------------------------------------
# 1️⃣2️⃣ STREAMING CHAT ENDPOINT
# --------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    WHY:
    - Used by frontend
    - Supports token-by-token rendering
    """

    context_chunks, _ = retrieve_context(request.message)
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context_text}

Question:
{request.message}
"""

    return StreamingResponse(
        stream_openai_response(prompt),
        media_type="text/plain"
    )

@app.options("/chat/stream")
def chat_stream_options():
    """
    WHY THIS EXISTS:
    
    - Handles browser preflight requests
    - Prevents 405 Method Not Allowed
    - Required for streaming from frontend
    """
    return {}
# --------------------------------------------------
# 1️⃣3️⃣ DOCUMENT UPLOAD
# --------------------------------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    WHY:
    - Allows dynamic knowledge ingestion
    - Rebuilds RAG index after upload
    """

    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF and TXT supported")

    os.makedirs("documents", exist_ok=True)
    path = os.path.join("documents", file.filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    initialize_rag()

    return {"message": "Uploaded and indexed", "file": file.filename}
