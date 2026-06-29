# RAG Agent — PDF Question Answering System

A full-stack Retrieval-Augmented Generation (RAG) application for chatting with PDF documents. Supports single-document and multi-document (Project) modes, hybrid search, cross-encoder reranking, and RAGAS-based pipeline evaluation.

---

## Features

| Category | Details |
|----------|---------|
| **Upload & Ingest** | PDF parsing → chunking → embedding → Qdrant vector store |
| **Hybrid Search** | Dense vector (NVIDIA NIM / Ollama) + BM25 sparse, fused with RRF |
| **Reranking** | Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| **Advanced Retrieval** | HyDE (Hypothetical Document Embedding), query decomposition |
| **Projects** | Group multiple PDFs → multi-document RAG retrieval in one query |
| **Conversation Memory** | Full history passed to LLM; history-aware retrieval for follow-up queries |
| **Context Window Panel** | Inspect system prompt, retrieved chunks, history, and token estimates live |
| **Citations** | Page-level citations with jump-to-page in the built-in PDF viewer |
| **Image Support** | Vision-capable models can receive extracted page images |
| **RAGAS Evaluation** | 5-config ablation suite with HTML report |

---

## Architecture

```
frontend (React + Vite)
    │  SSE stream
    ▼
backend (FastAPI)
    ├── /documents  — upload, ingest, status
    ├── /projects   — multi-doc project CRUD
    ├── /chat       — hybrid search → rerank → LLM stream
    └── /eval       — RAGAS evaluation endpoint

Storage
    ├── Qdrant      — vector store (Docker)
    ├── SQLite      — document & project metadata
    └── storage/    — uploaded PDFs, parsed chunks, images
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)
- NVIDIA NIM API key **or** Ollama running locally

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Backend

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Copy and fill in your credentials
cp .env.example .env
# Edit .env — set NVIDIA_API_KEY and NVIDIA_CHAT_MODEL (or switch LLM_PROVIDER=ollama)

uvicorn app.main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### One-command launch (Windows)

```bat
launch.bat
```

Kills any existing processes on ports 8000 / 5173, then opens two CMD windows (backend + frontend) in parallel.

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and fill in:

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `nvidia` (default) or `ollama` |
| `NVIDIA_API_KEY` | Key from [build.nvidia.com](https://build.nvidia.com) |
| `NVIDIA_CHAT_MODEL` | e.g. `openai/gpt-4o`, `nvidia/llama-3.1-nemotron-ultra-253b-v1` |
| `NVIDIA_EMBED_MODEL` | e.g. `nvidia/nv-embedqa-e5-v5` |
| `OLLAMA_CHAT_MODEL` | e.g. `qwen2.5`, `llama3.2` (when using Ollama) |
| `QDRANT_URL` | Default: `http://localhost:6333` |
| `RERANKER_ENABLED` | `true` / `false` |
| `HYBRID_SEARCH_ENABLED` | `true` / `false` |

> **Security:** Never commit `.env`. Never prefix secrets with `VITE_`.

---

## Memory Architecture

Each chat request assembles a 4-layer context sent to the LLM:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — System Prompt (static)                       │
│  "You are a helpful assistant that answers questions…"  │
├─────────────────────────────────────────────────────────┤
│  Layer 2 — Retrieved Context (working memory)           │
│  Top-K chunks from Qdrant + BM25, reranked by score     │
│  Updated on every new query                             │
├─────────────────────────────────────────────────────────┤
│  Layer 3 — Conversation History (short-term memory)     │
│  All previous user/assistant turns in this session      │
│  Configurable: keep last N turns via the UI slider      │
├─────────────────────────────────────────────────────────┤
│  Layer 4 — Current Query                                │
│  The user's new message                                 │
└─────────────────────────────────────────────────────────┘
```

### History-Aware Retrieval

Short follow-up messages like *"Answer in Vietnamese"* or *"Explain more"* contain no document keywords, causing RAG to retrieve irrelevant chunks. The system detects queries under 8 words and automatically enriches them with the previous user question before running retrieval — while still passing the original message to the LLM:

```
User turn 1: "What is Cosine Similarity?"  →  retrieval query: "What is Cosine Similarity?"
User turn 2: "Answer in Vietnamese"        →  retrieval query: "What is Cosine Similarity? Answer in Vietnamese"
                                               (LLM still sees: "Answer in Vietnamese" + history)
```

### Context Window Panel

The UI exposes all 4 layers in a collapsible **Context Window** panel at the bottom of the chat column, with live token estimates (~4 chars/token) and a slider to cap how many history turns are sent to the LLM.

---

## Project Mode (Multi-Document RAG)

1. Upload multiple PDFs
2. Create a **Project** and add documents to it
3. Select the project in the sidebar — all chat queries retrieve across all documents simultaneously

Retrieval uses Qdrant `Filter(should=[...])` for multi-doc vector search and per-doc BM25 merge.

---

## RAGAS Evaluation

Evaluate pipeline quality with synthetic Q&A pairs generated from your documents.

```bash
cd backend

# List ingested documents
python scripts/eval_suite.py list

# Generate 25 Q&A pairs and run full ablation (5 configs)
python scripts/eval_suite.py prepare --document-id <uuid> --n 25
python scripts/eval_suite.py ablation --document-id <uuid>

# Generate HTML report
python scripts/eval_suite.py report --document-id <uuid>
# → eval_results/<uuid>/report.html
```

### Ablation Configs

| Config | Hybrid | Rerank | HyDE | Decomp |
|--------|--------|--------|------|--------|
| Vector Only | ✗ | ✗ | ✗ | ✗ |
| Hybrid | ✓ | ✗ | ✗ | ✗ |
| Hybrid + Rerank | ✓ | ✓ | ✗ | ✗ |
| Hybrid + Rerank + HyDE | ✓ | ✓ | ✓ | ✗ |
| Full | ✓ | ✓ | ✓ | ✓ |

### Sample Results (25 Q&A, gpt-oss-120b)

| Pipeline | Faithfulness | Answer Relevancy | Ctx Precision | Ctx Recall | Avg |
|----------|-------------|-----------------|---------------|------------|-----|
| Vector Only | 0.898 | 0.807 | 0.872 | 1.000 | 0.894 |
| Hybrid BM25+RRF | 0.929 | 0.813 | 0.910 | 0.958 | 0.903 |
| Hybrid + Rerank | 0.930 | 0.802 | 0.911 | 1.000 | 0.911 |
| **Hybrid+Rerank+HyDE** | **0.933** | **0.968** | **0.935** | **1.000** | **0.959** |
| Full (+Decomp) | 0.914 | 0.845 | 0.881 | 0.955 | 0.898 |

---

## Tech Stack

**Backend:** FastAPI · SQLAlchemy (async) · Qdrant · rank-bm25 · sentence-transformers · PyMuPDF · NLTK

**Frontend:** React 18 · TypeScript · Vite · react-pdf · react-markdown

**Evaluation:** RAGAS 0.1 · LangChain · HuggingFace sentence-transformers

**Infrastructure:** Docker (Qdrant) · SQLite · NVIDIA NIM / Ollama
