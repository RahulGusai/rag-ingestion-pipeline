# LlamaIndex + Supabase Ingest (FastAPI)

This project is a local FastAPI example that uses **LlamaIndex** primitives to ingest files stored in Supabase Storage, parse -> chunk -> embed -> store, and uses a **custom VectorStore adapter** that writes embeddings into Supabase (pgvector).

**What it contains**
- `app/main.py` — FastAPI server with `/ingest` endpoint (workspace_id, trigger_by, file_ids).
- `app/vectorstores/supabase_vector_store.py` — Custom LlamaIndex VectorStore adapter that writes chunks and embeddings into Supabase tables.
- `requirements.txt` — Python dependencies list.
- `.env.example` — Example environment variables.
- `Dockerfile` — Optional Dockerfile to containerize the service.
- `README.md` — this file.

**Notes**
- This package only creates the code scaffolding locally. It does not run or install dependencies here.
- The code is written to be production-ready and idempotent (deterministic chunk ids).
- After testing locally, you can containerize and deploy to Cloud Run.
