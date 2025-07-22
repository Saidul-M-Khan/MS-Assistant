# MS Assistant AI

## Overview

**MS Assistant AI** is an AI-powered Q&A system for medical and research documents (PDF/TXT), with a focus on Multiple Sclerosis (MS) and mycotoxin testing. It allows users to upload documents, processes them into searchable vector embeddings using SentenceTransformers, and provides document-grounded answers using OpenAI's GPT models. All document search and chat context is managed with ChromaDB for fast, persistent vector search.

---

## Features

- **Document Upload**: Upload PDF or TXT files for ingestion into the knowledge base.
- **Semantic Search**: Uses SentenceTransformers to embed and search document chunks.
- **Vector Database**: ChromaDB for persistent, high-performance vector search.
- **AI Chat**: Ask questions and get answers strictly based on uploaded documents (no hallucination).
- **Session Management**: Maintains chat sessions and history per user.
- **Mycotoxin & MS Knowledge**: Built-in medical context for specialized queries.
- **Admin/Role Support**: Public/private document access and admin features.
- **REST API**: FastAPI backend with clear endpoints for integration.

---

## Architecture

```mermaid
flowchart TD
    A[User] -->|Uploads PDF/TXT| B[FastAPI Backend]
    B --> C[Document Processing]
    C --> D[Text Chunking & Embedding]
    D --> E[ChromaDB (Vector DB)]
    D --> F[SQL Database (Metadata)]
    A -->|Asks Question| B
    B -->|Query Embedding| E
    E -->|Relevant Chunks| B
    B -->|Context + Chat| G[OpenAI GPT]
    G -->|Response| A
```

---

## Project Structure

```
MS-Assistant/
├── app/
│   ├── api.py            # FastAPI endpoints
│   ├── ms_assistant.py   # Core AI logic (embedding, ChromaDB, OpenAI)
│   ├── models.py         # SQLAlchemy models
│   ├── database.py       # DB session/config
│   └── ...
├── uploads/              # Uploaded PDF/TXT files
├── chroma_db/            # ChromaDB vector store (auto-created)
├── venv/                 # Python virtual environment
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
└── main.py               # App entry point
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd MS-Assistant
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=sk-...
```

### 5. Run the Application
```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

---

## API Usage

### **Upload Documents**
`POST /ai/upload`
- **Form Data:**
  - `files`: One or more PDF/TXT files
  - `email`: User email
  - `author`, `description`, `category` (optional)
- **Response:**
```json
{
  "uploaded": [
    {"filename": "doc1.pdf", "document_id": "...", "status": "success"}
  ],
  "errors": [],
  "total_files": 1,
  "successful": 1,
  "failed": 0
}
```

### **Chat with AI**
`POST /chat`
- **Body:**
```json
{
  "session_id": "...",  // Optional for new session
  "message": "What are the main findings?",
  "email": "user@example.com"
}
```
- **Response:**
```json
{
  "session_id": "...",
  "message": "What are the main findings?",
  "response": "...AI answer strictly from docs...",
  "timestamp": "..."
}
```

### **Get Session Messages**
`GET /session/{session_id}/chats`
- **Response:**
```json
[
  {"id": "...", "session_id": "...", "message": "...", "response": "...", "timestamp": "..."}
]
```

### **Delete Session**
`DELETE /sessions/{session_id}`
- Deletes the session and all its chat messages.

---

## Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

---

## Troubleshooting
- **No `chroma_db/` folder?**
  - Make sure you are using `chromadb.PersistentClient(path="./chroma_db")` in your code.
  - Upload a document to trigger vector storage.
- **Uploads not working?**
  - Ensure the `uploads/` directory exists and is writable.
- **OpenAI errors?**
  - Check your API key and network connection.
- **Database errors?**
  - Make sure you have the correct DB setup and migrations applied.

---

## Contribution Guidelines
- Fork the repo and create a feature branch.
- Write clear, well-documented code.
- Add tests for new features if possible.
- Open a pull request with a detailed description.

---

## License
MIT License (add your license file if needed) 