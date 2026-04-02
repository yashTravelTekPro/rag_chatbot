# EzeeChatBot - RAG Chatbot API

A minimal RAG (Retrieval-Augmented Generation) chatbot API that lets you upload your own knowledge base and get a chatbot that answers questions grounded only in that content.

## Features

- Upload text content or URLs to create custom knowledge bases
*** Begin README replacement ***

# EzeeChatBot — Minimal RAG Chatbot API

A minimal backend that accepts user knowledge (text, HTML URL, or PDF URL), chunks it, generates embeddings, stores them in a vector store, and serves a streaming chat endpoint that answers using only the uploaded content.

---

## Quick setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the server (preferred):

```bash
python main.py
```

Alternative (explicit uvicorn):

```bash
# optional
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

OpenAPI UI: http://localhost:8000/docs

---

## API — endpoints & examples

### 1) POST /upload

Accepts:
- JSON: `{ "content": "..." }` or `{ "url": "..." }`
- `text/plain` body: raw multi-line text (convenient from curl)

Examples:

Upload text content:
```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991.","url":""
  }'
```

Upload from URL:
```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{"content":"",
    "url": "https://www.mwrlife.com/content/MWRTravelAdvantageIntlT&C.pdf"
  }'
```

Successful response:

```json
{
  "bot_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_created": 10,
  "message": "Knowledge base created successfully with 10 chunks"
}
```

Notes:
- PDFs: the service will try LangChain's `UnstructuredPDFLoader` if present; otherwise it uses `pypdf`.

---

### 2) POST /chat

CURL:

  ```bash
  curl -N -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"bot_id":"<BOT_ID>","user_message":"What is X?","conversation_history": [] }'
  ```


Behavior and response:
- The endpoint streams plain-text chunks from the LLM. The HTTP response is `text/plain` and may arrive incrementally.
- If the requested `bot_id` does not exist the API returns 404.
- If the knowledge base does not contain an answer, the assistant is instructed to respond with a short refusal like:

```
I don't have enough information in my knowledge base to answer that question.
```

Example (streaming):

```text
The cancellation and refund policy states that cancellations made within 24 hours ... [stream continues]
```

Client note: to see streaming in the terminal use `curl -N` (no buffering):

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"bot_id":"<BOT_ID>","user_message":"Tell me about refunds","conversation_history": [] }'
```

---

### 3) GET /stats/{bot_id}

Returns per-bot statistics. Example response:
curl -X 'GET' \
  'http://localhost:8000/stats/0adf2434-5b4f-4cb8-9725-5bdf47fcde76' \
  -H 'accept: application/json'
  
  
```json
{
  "bot_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_messages": 42,
  "average_latency_ms": 1250.5,
  "estimated_cost_usd": 0.12,
  "unanswered_questions": 3
}
```

- `total_messages`: number of chat requests served for the bot
- `average_latency_ms`: average response time (milliseconds)
- `estimated_cost_usd`: running token/LLM cost estimate (USD)
- `unanswered_questions`: count of conversations where the bot explicitly said it couldn't answer

---

## Chunking strategy (short)

- Split by sentence boundaries using a regex: `(?<=[.!?])\\s+`.
- Accumulate sentences until a target character size is reached (default 800 chars).
- When a chunk is emitted, keep an overlap of sentences (up to `overlap` characters) to seed the next chunk.

Rationale: fast, deterministic, sentence-coherent chunks with overlap to preserve cross-boundary context.

---

  ## Project layout

  - `main.py` — FastAPI app entrypoint (registers routes and starts server).
  - `app/api/routes.py` — API routes: `/upload`, `/chat`, `/stats`.
  - `app/models.py` — Pydantic request/response models used by endpoints.
  - `app/services/` — core logic:
    - `upload_service.py` — fetch/parse URL or accept raw text, chunk, embed, store.
    - `chunking_engine.py` — simple sentence-based chunker with overlap.
    - `embedding_service.py` — wrapper over `sentence-transformers` for embeddings.
    - `llm_client.py` — wrapper for streaming LLM calls (project-specific).
    - `chat_service.py` — retrieves relevant chunks, builds prompt, streams LLM response, updates stats.
  - `app/data/` — in-memory/demo stores and data models (vector store, stats store).
  - `tests/` — test suite (unit tests for chunking engine included).

  ## How the API works

  1. `POST /upload`
     - Accepts either:
       - JSON: `{ "content": "..." }` or `{ "url": "..." }`
       - `text/plain` body: raw multi-line text (convenient from curl)
     - If `url` is provided the service fetches it and extracts text:
       - HTML pages: strips scripts/styles and tags to get visible text.
       - PDFs: attempts to use LangChain's `UnstructuredPDFLoader` (if installed) and falls back to `pypdf`.
     - Text is chunked, embeddings are computed, and chunks are stored with a generated `bot_id`.
     - Returns `{ bot_id, chunks_created, message }`.

  2. `POST /chat`
     - Body: `{ "bot_id": "...", "user_message": "...", "conversation_history": [...] }`
     - The service generates an embedding for the query, retrieves top-k similar chunks from the vector store, builds a system prompt that contains only those chunks, and streams the LLM output back to the client.
     - The bot is constrained to use only the provided context; if the answer is not present it should explicitly say so.

  3. `GET /stats/{bot_id}`
     - Returns per-bot metrics: total messages, average latency (ms), estimated token cost, unanswered question count.

  ## Chunking strategy (short)

  - Split by sentence boundaries using a regex: `(?<=[.!?])\\s+`.
  - Accumulate sentences until a target character size is reached (default 800 characters).
  - When a chunk is emitted, keep an overlap of sentences (up to `overlap` characters) to seed the next chunk.

  Why this approach?
  - Simple, deterministic, and fast. No heavy NLP dependencies required.
  - Sentence boundaries keep chunks coherent; overlap helps when answers span boundaries.

  Limitations: naive splitting can mis-handle abbreviations and other edge cases; for production use a robust tokenizer (spaCy/NLTK).



