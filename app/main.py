# main.py (only the relevant changed parts shown; keep rest of file as-is)
import os
import io
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, UUID4

# llama-index
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from openai import OpenAI

from app.vectorstores.supabase_vector_store import SupabaseVectorStore
from supabase import create_client

# ---- NEW: import parse helpers from utils
from app.utils.parse_bytes import (
    parse_pdf_bytes,
    parse_docx_bytes,
    parse_text_bytes,
    parse_csv_bytes,
)

# keep the rest of your imports
from pypdf import PdfReader
from dotenv import load_dotenv
import tiktoken

# retries for embeddings
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()

# ---------------------------
# Config & env (unchanged)
# ---------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "files")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "text-embedding-3-small")
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "64"))
AVG_CHARS_PER_TOKEN = float(os.environ.get("AVG_CHARS_PER_TOKEN", "4.0"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))

# Concurrency settings (unchanged)
MAX_CONCURRENT_FILES = int(os.environ.get(
    "MAX_CONCURRENT_FILES", "4"))
THREADPOOL_MAX_WORKERS = int(os.environ.get(
    "THREADPOOL_MAX_WORKERS", str(MAX_CONCURRENT_FILES)))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")

logger = logging.getLogger("ingestion-pipeline")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG ingestion pipeline (parallel-file workers)")

ALLOWED_FILE_TYPES = {
    'pdf': 'application/pdf',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'txt': 'text/plain',
    'csv': 'text/csv'
}

# ThreadPoolExecutor is module-level so it can be reused across requests.
_executor = ThreadPoolExecutor(max_workers=THREADPOOL_MAX_WORKERS)


class IngestRequest(BaseModel):
    workspace_id: UUID4
    trigger_by: UUID4
    file_ids: List[UUID4] = Field(..., min_items=1)
# ---------------------------
# parse_file_bytes (now delegates to utils.parse_bytes)
# ---------------------------


def parse_file_bytes(mime_type: str, filename: str, file_bytes: bytes) -> str:
    """
    Delegates to parse helpers in utils.parse_bytes.
    """
    if mime_type == ALLOWED_FILE_TYPES['pdf'] or filename.lower().endswith(".pdf"):
        return parse_pdf_bytes(file_bytes)
    if mime_type == ALLOWED_FILE_TYPES['docx'] or filename.lower().endswith(".docx"):
        return parse_docx_bytes(file_bytes)
    if mime_type == ALLOWED_FILE_TYPES['txt'] or filename.lower().endswith(".txt"):
        return parse_text_bytes(file_bytes)
    if mime_type == ALLOWED_FILE_TYPES['csv'] or filename.lower().endswith(".csv"):
        return parse_csv_bytes(file_bytes)
    # default to text parsing
    return parse_text_bytes(file_bytes)

# ---------------------------
# Supabase helpers (now accept optional supabase_client so workers can create their own)
# ---------------------------


def fetch_file_row(file_id: str, supabase_client=None) -> Optional[Dict[str, Any]]:
    client = supabase_client or create_client(
        SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    resp = client.table("files").select(
        "*").eq("id", str(file_id)).maybe_single().execute()
    # supabase-py returns .status_code on resp in some versions; robustly check resp.data or resp.error
    if hasattr(resp, "response") and getattr(resp, "response").status_code not in (200, 204):
        raise RuntimeError(
            f"Error fetching file {file_id}: {resp.response.text}")
    return resp.data


def download_file_from_storage(storage_path: str, supabase_client=None) -> bytes:
    client = supabase_client or create_client(
        SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    bucket = SUPABASE_STORAGE_BUCKET
    res = client.storage.from_(bucket).download(storage_path)
    if isinstance(res, (bytes, bytearray)):
        return bytes(res)
    if hasattr(res, "content"):
        return res.content
    if isinstance(res, dict) and res.get("data"):
        return res["data"]
    raise RuntimeError("Failed to download file from storage")


def mark_file_indexed(file_id: str, supabase_client=None):
    client = supabase_client or create_client(
        SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    now = datetime.now(timezone.utc).isoformat()
    resp = client.table("files").update({
        "last_indexed_at": now,
        "indexing_status": "indexed"
    }).eq("id", file_id).execute()

    if hasattr(resp, "response") and getattr(resp, "response").status_code not in (200, 204):
        raise RuntimeError(
            f"Error marking file {file_id} indexed: {resp.response.text}")
    else:
        logger.info("File %s marked as indexed", file_id)


# ---------------------------
# Token helpers using tiktoken (instantiated per-worker)
# ---------------------------
def get_tiktoken_encoder_for_model(model_name: str):
    """
    Use model_name to pick the best encoding; fall back to cl100k_base.
    NOTE: instantiate per worker to avoid any shared-state/thread issues.
    """
    if tiktoken is None:
        raise RuntimeError(
            "tiktoken required. Install with `pip install tiktoken`.")
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            raise RuntimeError("Failed to get tiktoken encoder.")


def encode_tokens(encoder, text: str) -> List[int]:
    return encoder.encode(text)


def decode_tokens(encoder, tokens: List[int]) -> str:
    return encoder.decode(tokens)


def chunk_node_tokens(node_text: str, encoder, max_tokens: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Token-accurate chunking that returns only:
      - text (decoded slice)
      - token_count
    """
    if not node_text:
        return []

    tokens = encode_tokens(encoder, node_text)
    n = len(tokens)
    if n == 0:
        return []

    if overlap >= max_tokens:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_TOKENS")

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0
    start_idx = 0

    while start_idx < n:
        end_idx = min(n, start_idx + max_tokens)
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = decode_tokens(encoder, chunk_tokens).strip()

        chunks.append({
            "text": chunk_text,
            "token_count": len(chunk_tokens),
        })

        chunk_idx += 1
        if end_idx == n:
            break
        start_idx = end_idx - overlap

    return chunks


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception))
def create_embeddings_batch(openai_client: OpenAI, model_name: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = openai_client.embeddings.create(model=model_name, input=texts)
    if not getattr(resp, "data", None):
        raise RuntimeError("OpenAI embeddings API returned no data")
    return [item.embedding for item in resp.data]


# ---------------------------
# Per-file worker function (synchronous) - instantiated per worker
# ---------------------------
def _process_single_file(
    fid: str,
    workspace_id: str,
    trigger_by: str,
) -> Dict[str, Any]:
    """
    Synchronous worker function that handles one file end-to-end.
    It creates local clients/encoder to avoid sharing issues when used with ThreadPoolExecutor.
    Returns dict with either processed_files entry or an error entry.
    """
    # Create per-worker clients/encoder
    local_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    local_openai = OpenAI(api_key=OPENAI_API_KEY)
    encoder = get_tiktoken_encoder_for_model(EMBEDDING_MODEL_NAME)

    results_entry = {"file_id": fid}

    try:
        file_row = fetch_file_row(fid, supabase_client=local_supabase)
        if not file_row:
            raise HTTPException(
                status_code=404, detail=f"file {fid} not found")
        if str(file_row.get("workspace_id")) != workspace_id:
            raise HTTPException(
                status_code=400, detail=f"file {fid} not in workspace {workspace_id}")
        if file_row.get("is_deleted", False):
            raise HTTPException(
                status_code=400, detail=f"file {fid} is deleted")

        filename = file_row.get("filename")
        storage_path = f"{workspace_id}/{filename}"

        file_bytes = download_file_from_storage(
            storage_path, supabase_client=local_supabase)

        text = parse_file_bytes(file_row.get(
            "mime_type") or "", filename or "", file_bytes)

        if not text.strip():
            raise HTTPException(
                status_code=400, detail=f"no text extracted from file {fid}")

        # Document + token count
        doc = Document(text=text, doc_id=fid, extra_info={
            "filename": filename, "storage_path": storage_path, "file_id": fid
        })
        doc_token_count = len(encode_tokens(encoder, text))

        # Parse nodes
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents([doc])

        node_entries = []
        for n_idx, n in enumerate(nodes):
            try:
                node_text = n.get_text()
            except Exception:
                node_text = str(n)
            extra = getattr(n, "extra_info", {}) or {}
            node_entries.append({
                "node_id": getattr(n, "node_id", None) or getattr(n, "doc_id", None) or f"{fid}-node-{n_idx}",
                "text": node_text,
                "extra_info": extra
            })

        all_chunks = []
        per_file_chunk_counter = 0

        vector_store = SupabaseVectorStore(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_SERVICE_ROLE_KEY,
            table_chunks="file_chunks",
            table_embeddings="embeddings",
            embedding_dim=EMBEDDING_DIM,
        )

        for node in node_entries:
            node_text = node["text"] or ""
            node_chunks = chunk_node_tokens(
                node_text, encoder, CHUNK_TOKENS, CHUNK_OVERLAP)
            for nc in node_chunks:
                chunk_id = vector_store.generate_chunk_id(
                    fid, per_file_chunk_counter, nc["text"])
                chunk_metadata = {
                    "node_id": node["node_id"],
                    "filename": filename,
                    "source": storage_path,
                    "workspace_id": workspace_id,
                    **(node["extra_info"] or {})
                }

                all_chunks.append({
                    "id": chunk_id,
                    "file_id": fid,
                    "text": nc["text"],
                    "token_count": int(nc["token_count"]),
                    "metadata": chunk_metadata,
                })
                per_file_chunk_counter += 1

        if not all_chunks:
            raise HTTPException(
                status_code=400, detail=f"no chunks produced from file {fid}")

        logger.info("File %s: produced %d chunks", fid, len(all_chunks))

        # Prepare texts & chunk_ids for embedding (filter out empty / whitespace-only)
        texts_for_embed = []
        chunk_ids_ordered = []
        chunks_for_db = []
        for c in all_chunks:
            txt = (c["text"] or "").strip()
            if not txt:
                logger.info("Skipping empty chunk %s for file %s",
                            c.get("id"), fid)
                continue
            texts_for_embed.append(txt)
            chunk_ids_ordered.append(c["id"])
            chunks_for_db.append(c)

        if not texts_for_embed:
            raise HTTPException(
                status_code=400, detail=f"all chunks empty for file {fid}")

        # Create embeddings in batches with retry (using per-worker local_openai)
        embeddings: List[List[float]] = []
        models: List[str] = []
        for i in range(0, len(texts_for_embed), EMBED_BATCH):
            batch_texts = texts_for_embed[i:i + EMBED_BATCH]
            batch_embs = create_embeddings_batch(
                local_openai, EMBEDDING_MODEL_NAME, batch_texts)
            for emb in batch_embs:
                if len(emb) != EMBEDDING_DIM:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(emb)}")
            embeddings.extend(batch_embs)
            models.extend([EMBEDDING_MODEL_NAME] * len(batch_embs))

        if len(embeddings) != len(chunk_ids_ordered):
            raise RuntimeError("Mismatch between embeddings and chunk ids")

        # Upsert chunks + embeddings (vector_store will use its own supabase client internally)
        vector_store.upsert_chunks_and_embeddings(
            chunks_for_db, chunk_ids_ordered, embeddings, models=models)
        logger.info("File %s: upserted %d chunks+embeddings",
                    fid, len(chunk_ids_ordered))

        # mark as indexed using local_supabase for consistency
        mark_file_indexed(fid, supabase_client=local_supabase)

        results_entry.update({
            "chunks": len(chunk_ids_ordered),
            "tokens_in_document": doc_token_count,
            "status": "indexed"
        })

    except HTTPException as he:
        # Preserve HTTP status/detail for caller to inspect
        logger.exception("HTTP error processing %s: %s", fid, he)
        results_entry.update({"status": "error", "error": str(
            he.detail if hasattr(he, "detail") else he)})
    except Exception as e:
        logger.exception("error processing %s", fid)
        results_entry.update({"status": "error", "error": str(e)})

    return results_entry


@app.post("/ingest")
async def ingest(req: IngestRequest):
    workspace_id = str(req.workspace_id)
    trigger_by = str(req.trigger_by)
    file_ids = [str(x) for x in req.file_ids]

    # Basic preflight
    if not file_ids:
        raise HTTPException(status_code=400, detail="file_ids required")

    # We'll dispatch file-level workers to the module-level executor
    loop = asyncio.get_running_loop()

    # Create partial worker wrapper with fixed workspace/trigger_by
    worker = partial(_process_single_file,
                     workspace_id=workspace_id, trigger_by=trigger_by)

    # Submit all tasks but bounded by executor size (ThreadPoolExecutor already bounds concurrent threads).
    # Use asyncio.gather on run_in_executor for each file id to await them concurrently.
    tasks = [loop.run_in_executor(_executor, worker, fid) for fid in file_ids]

    # Await all worker results
    results_list = await asyncio.gather(*tasks, return_exceptions=False)

    # Aggregate results into processed_files and errors (consistent with previous API)
    response = {"processed_files": [], "errors": []}
    for r in results_list:
        if r.get("status") == "indexed":
            response["processed_files"].append({
                "file_id": r["file_id"],
                "chunks": r.get("chunks", 0),
                "tokens_in_document": r.get("tokens_in_document", 0),
            })
        else:
            response["errors"].append(
                {"file_id": r["file_id"], "error": r.get("error", "unknown")})

    return response
