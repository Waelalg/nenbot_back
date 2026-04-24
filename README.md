# NENBOT Backend

FastAPI backend for NENBOT, a Hunter x Hunter-only RAG chatbot with Groq, ChromaDB, local sentence-transformer embeddings, structured team lookup, streaming responses, and short-term session memory.

## Backend Repo Layout

Keep this structure in the backend GitHub repository:

```text
nenbot-backend/
  app/
  data/
  requirements.txt
  .env.example
  README.md
```

The backend code now uses relative imports, so the same code also works inside the monorepo as `backend/app/...`.

## Environment

Create `.env` from `.env.example` on the backend host. Never commit `.env`.

```text
GROQ_API_KEY=your_groq_api_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant
VOICE_STT_MODEL=whisper-large-v3-turbo
VOICE_TTS_MODEL=canopylabs/orpheus-v1-english
VOICE_TTS_VOICE=troy
VOICE_TTS_FORMAT=wav
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./chroma_db
MEMORY_TURNS=8
MAX_CONTEXT_CHUNKS=5
ALLOWED_SMALLTALK=true
```

## Local Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app/rag/ingest.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```text
http://localhost:8000/health
```

## Deploy Notes

- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Build/install command: `pip install -r requirements.txt`
- Add `GROQ_API_KEY` as a private environment variable in the hosting dashboard.
- Run `python app/rag/ingest.py` during deployment or after the first deploy to build ChromaDB.
- Keep `chroma_db/` out of Git; it is generated data.

## Public API

- `GET /health`
- `POST /chat`
- `POST /chat/stream`
- `POST /voice/transcribe`
- `POST /voice/speak`
- `POST /reset`
- `POST /ingest`
- `GET /team`
- `GET /sources`
