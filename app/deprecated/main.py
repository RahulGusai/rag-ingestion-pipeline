# main.py
import os
import io
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, UUID4

# llama-index
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from openai import OpenAI

from app.vectorstores.supabase_vector_store import SupabaseVectorStore
from supabase import create_client

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text
from pypdf import PdfReader
from dotenv import load_dotenv
import tiktoken
import zipfile

# retries for embeddings
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()

# ---------------------------
# Config & env
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

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

logger = logging.getLogger("llamaindex_ingest")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG ingestion pipeline (full)")

ALLOWED_FILE_TYPES = {
    'pdf': 'application/pdf',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'txt': 'text/plain',
    'csv': 'text/csv'
}


class IngestRequest(BaseModel):
    workspace_id: UUID4
    trigger_by: UUID4
    file_ids: List[UUID4] = Field(..., min_items=1)


# ---------------------------
# Parsing helpers (unchanged)
# ---------------------------
def parse_pdf_bytes(file_bytes: bytes) -> str:
    try:
        with io.BytesIO(file_bytes) as b:
            elements = partition_pdf(file=b)
            text = "\n\n".join([e.get_text()
                               for e in elements if hasattr(e, "get_text")])
            if text.strip():
                return text
    except Exception as e:
        logger.info("unstructured partition_pdf failed: %s", e)

    with io.BytesIO(file_bytes) as b:
        reader = PdfReader(b)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n\n".join(pages)


def parse_docx_bytes(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as b:
        elements = partition_docx(filename="tmp.docx", file=b)
        text = "\n\n".join([e.get_text()
                           for e in elements if hasattr(e, "get_text")])
        return text


def parse_text_bytes(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as b:
        elements = partition_text(filename="tmp.txt", file=b)
        text = "\n\n".join([e.get_text()
                           for e in elements if hasattr(e, "get_text")])
        return text


def parse_csv_bytes(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def parse_file_bytes(mime_type: str, filename: str, file_bytes: bytes) -> str:
    if mime_type == ALLOWED_FILE_TYPES['pdf'] or filename.lower().endswith(".pdf"):
        return parse_pdf_bytes(file_bytes)
    if mime_type == ALLOWED_FILE_TYPES['docx'] or filename.lower().endswith(".docx"):
        return parse_docx_bytes(file_bytes)
    if mime_type == ALLOWED_FILE_TYPES['txt'] or filename.lower().endswith(".txt"):
        return parse_text_bytes(file_bytes)
    if mime_type == ALLOWED_FILE_TYPES['csv'] or filename.lower().endswith(".csv"):
        return parse_csv_bytes(file_bytes)
    return parse_text_bytes(file_bytes)


# ---------------------------
# Supabase helpers (unchanged)
# ---------------------------
def fetch_file_row(file_id: str) -> Optional[Dict[str, Any]]:
    resp = supabase.table("files").select(
        "*").eq("id", str(file_id)).maybe_single().execute()
    if hasattr(resp, "response") and getattr(resp, "response").status_code not in (200, 204):
        raise RuntimeError(
            f"Error fetching file {file_id}: {resp.response.text}")
    return resp.data


def download_file_from_storage(storage_path: str) -> bytes:
    bucket = SUPABASE_STORAGE_BUCKET
    res = supabase.storage.from_(bucket).download(storage_path)
    if isinstance(res, (bytes, bytearray)):
        return bytes(res)
    if hasattr(res, "content"):
        return res.content
    if isinstance(res, dict) and res.get("data"):
        return res["data"]
    raise RuntimeError("Failed to download file from storage")


def mark_file_indexed(file_id: str):
    now = datetime.now(timezone.utc).isoformat()
    resp = supabase.table("files").update({
        "last_indexed_at": now,
        "status": "indexed"
    }).eq("id", file_id).execute()

    if hasattr(resp, "response") and getattr(resp, "response").status_code not in (200, 204):
        raise RuntimeError(
            f"Error marking file {file_id} indexed: {resp.response.text}")
    else:
        logger.info("File %s marked as indexed", file_id)


# ---------------------------
# Token helpers using tiktoken (match encoder to embedding model)
# ---------------------------
def get_tiktoken_encoder_for_model(model_name: str):
    """
    Use model_name to pick the best encoding; fall back to cl100k_base.
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
      - chunk_index

    Does NOT return start/end token indices or char offsets (MVP).
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
# Ingest endpoint - full pipeline (uses deterministic chunk ids + client-side transactional upsert)
# ---------------------------
@app.post("/ingest")
async def ingest(req: IngestRequest):
    workspace_id = str(req.workspace_id)
    trigger_by = str(req.trigger_by)
    file_ids = [str(x) for x in req.file_ids]

    if tiktoken is None:
        raise HTTPException(
            status_code=500, detail="tiktoken is required. Install with `pip install tiktoken`.")

    # Choose encoder that matches embedding model tokenizer
    encoder = get_tiktoken_encoder_for_model(EMBEDDING_MODEL_NAME)

    # instantiate vector store
    vector_store = SupabaseVectorStore(
        supabase_url=os.environ.get("SUPABASE_URL"),
        supabase_key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
        table_chunks="file_chunks",
        table_embeddings="embeddings",
        embedding_dim=EMBEDDING_DIM,
    )

    results = {"processed_files": [], "errors": []}

    for fid in file_ids:
        try:
            file_row = fetch_file_row(fid)
            if not file_row:
                raise HTTPException(
                    status_code=404, detail=f"file {fid} not found")
            if str(file_row.get("workspace_id")) != workspace_id:
                raise HTTPException(
                    status_code=400, detail=f"file {fid} not in workspace {workspace_id}")
            if file_row.get("is_deleted", False):
                raise HTTPException(
                    status_code=400, detail=f"file {fid} is deleted")

            uploaded_by = file_row.get("uploaded_by")
            filename = file_row.get("filename")
            storage_path = f"{uploaded_by}/{filename}"
            file_bytes = download_file_from_storage(storage_path)
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

            # Build node entries: node_id, text, extra_info
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
                    logger.info(
                        "Skipping empty chunk %s for file %s", c.get("id"), fid)
                    continue
                texts_for_embed.append(txt)
                chunk_ids_ordered.append(c["id"])
                # these include 'id' so vectorstore can upsert deterministically
                chunks_for_db.append(c)

            if not texts_for_embed:
                raise HTTPException(
                    status_code=400, detail=f"all chunks empty for file {fid}")

            # Create embeddings in batches with retry
            embeddings: List[List[float]] = []
            models: List[str] = []
            for i in range(0, len(texts_for_embed), EMBED_BATCH):
                batch_texts = texts_for_embed[i:i + EMBED_BATCH]
                batch_embs = create_embeddings_batch(
                    client, EMBEDDING_MODEL_NAME, batch_texts)
                for emb in batch_embs:
                    if len(emb) != EMBEDDING_DIM:
                        raise RuntimeError(
                            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(emb)}")
                embeddings.extend(batch_embs)
                models.extend([EMBEDDING_MODEL_NAME] * len(batch_embs))

            if len(embeddings) != len(chunk_ids_ordered):
                raise RuntimeError("Mismatch between embeddings and chunk ids")

            # Upsert chunks + embeddings client-side with best-effort rollback
            vector_store.upsert_chunks_and_embeddings(
                chunks_for_db, chunk_ids_ordered, embeddings, models=models)
            logger.info("File %s: upserted %d chunks+embeddings",
                        fid, len(chunk_ids_ordered))

            # mark as indexed
            mark_file_indexed(fid)
            results["processed_files"].append({
                "file_id": fid,
                "chunks": len(chunk_ids_ordered),
                "tokens_in_document": doc_token_count
            })
        except HTTPException as he:
            logger.exception("HTTP error processing %s: %s", fid, he)
            results["errors"].append({"file_id": fid, "error": str(
                he.detail if hasattr(he, 'detail') else he)})
        except Exception as e:
            logger.exception("error processing %s", fid)
            results["errors"].append({"file_id": fid, "error": str(e)})

    return results
