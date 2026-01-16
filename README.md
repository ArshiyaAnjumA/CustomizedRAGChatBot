
# 🧠 DocDive Chat  
### A Step-by-Step RAG Chatbot That Reads Your PDFs (with Streaming + Citations)

DocDive Chat is a **Retrieval-Augmented Generation (RAG) chatbot** built to show **how RAG actually works**, end to end — not just how to demo it.

Instead of answering questions directly, the chatbot:
1. retrieves relevant context from your documents  
2. grounds the response in that context  
3. streams the answer live  
4. shows where the answer came from  

This project is designed to be **beginner-friendly**, **transparent**, and **production-minded**.

---

## ✨ What This Project Demonstrates

- ✅ Real RAG pipeline (not prompt-only)
- ✅ PDF + TXT document ingestion
- ✅ Clean text chunking and embeddings
- ✅ FAISS vector search
- ✅ Streaming chat responses
- ✅ Short citations for trust and clarity
- ✅ Conversation memory (per session)
- ✅ Simple, presentable UI
- ✅ Clear separation of frontend / backend / retrieval logic

If you want to understand **why RAG works** — and what usually breaks — this repo is for you.

---

## 🏗️ Architecture Overview

```

User Question
↓
Frontend UI (streaming chat)
↓
FastAPI Backend
↓
Conversation Memory
↓
RAG Retrieval (FAISS)
↓
Relevant Document Chunks
↓
LLM (Answer Generation)
↓
Answer + Citations
↓
Back to User

```

The chatbot never answers directly.  
Every response is grounded in retrieved document context.

---

## 📁 Project Structure

```

docdive-chat/
backend/
main.py          # FastAPI server, streaming, uploads
rag.py           # RAG engine (chunking, embeddings, FAISS)
requirements.txt
.env             # OpenAI API key
documents/       # Your PDFs / TXT files (source of truth)
rag_store/       # Persisted FAISS index (auto-created)
frontend/
index.html       # Streaming chat UI

````

---

## ⚙️ Prerequisites

- Python **3.10 – 3.12** recommended
- OpenAI API key
- macOS / Linux / Windows (WSL works)

---

## 🔑 Setup

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/docdive-chat.git
cd docdive-chat
````

### 2️⃣ Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/binactivate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

Add at least one document to:

```
backend/documents/
```

---

## ▶️ Run the Application

### Start the backend

```bash
uvicorn main:app --reload
```

You should see logs indicating:

* documents loaded
* embeddings created
* FAISS index built

### Start the frontend

```bash
cd ../frontend
python3 -m http.server 5500
```

Open in browser:

```
http://localhost:5500
```

---

## 📄 Upload New Documents

You can upload new PDFs or TXT files at runtime.

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@/full/path/to/your_document.pdf"
```

The backend **automatically rebuilds the RAG index**, so new documents are searchable immediately.

---

## 💡 How RAG Works (In This Project)

RAG is a simple pipeline:

1. Take your domain knowledge (PDFs, docs)
2. Split it into clean chunks
3. Convert chunks into embeddings
4. Store them in a vector index (FAISS)
5. For each question:

   * retrieve the most relevant chunks
   * generate an answer **only using that context**

This approach produces answers that feel:

* helpful
* reliable
* explainable
* ready for real users

---

## 🧯 Common Issues (And Why You Won’t Hit Them Here)

| Problem                         | How This Repo Avoids It           |
| ------------------------------- | --------------------------------- |
| New PDFs not showing up         | Index rebuild on upload           |
| `$.input is invalid` errors     | Chunk sanitation before embedding |
| Chat works but UI shows nothing | Proper CORS + streaming setup     |
| Hallucinated answers            | Hard prompt grounding             |
| Confusing architecture          | Clear file separation             |

---

## 🧪 Who This Is For

* Engineers learning RAG for the first time
* Developers tired of “toy chatbot” tutorials
* Anyone who wants to understand **why RAG works**
* Teams prototyping internal search or knowledge bots

---

## 🚀 Possible Next Improvements

* Incremental indexing (no full rebuild)
* Hybrid search (keyword + vector)
* Smarter chunking with overlap
* Highlighting retrieved chunks in UI
* Human handoff for complex queries

---

## 🧡 A Note From the Author

This project came from breaking things, fixing them, and understanding *why* they broke.

RAG systems don’t fail loudly — they fail confidently.
That’s what makes building them interesting.

If you find this useful, feel free to star the repo or share feedback.


