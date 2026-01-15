# vectorstore.py
import os
import hashlib
import logging
import uuid
from typing import List, Dict, Any, Optional
from app.utils.text import sanitize_text_for_db
from supabase import create_client

logger = logging.getLogger("supabase_vector_store")
logging.basicConfig(level=logging.INFO)


class SupabaseVectorStore:
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        table_chunks: str = "file_chunks",
        table_embeddings: str = "embeddings",
        embedding_dim: int = 1536,
    ):
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get(
            "SUPABASE_SERVICE_ROLE_KEY")
        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        self.client = create_client(self.supabase_url, self.supabase_key)
        self.table_chunks = table_chunks
        self.table_embeddings = table_embeddings
        self.embedding_dim = int(embedding_dim)

        # fixed namespace for deterministic uuid generation (stable)
        self._chunk_ns = uuid.UUID("00000000-0000-0000-0000-000000000000")

    @staticmethod
    def _sha256_hex(s: str) -> str:
        return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

    def generate_chunk_id(self, file_id: str, chunk_index: int, chunk_text: str) -> str:
        """
        Deterministic chunk UUID based on (file_id, chunk_index, sha256(chunk_text)).
        """
        sha_hex = self._sha256_hex(chunk_text or "")
        name = f"{file_id}|{chunk_index}|{sha_hex}"
        return str(uuid.uuid5(self._chunk_ns, name))

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Upsert chunk rows into `file_chunks`. Expects each chunk to either include 'id' or we'll compute one.
        Returns deterministic chunk_ids in the same order as input.
        """
        if not chunks:
            return []

        rows = []
        chunk_ids = []
        for c in chunks:
            file_id = str(c["file_id"])
            text = c.get("text") or ""
            token_count = int(c.get("token_count") or 0)
            metadata = c.get("metadata") or {}

            chunk_id = c.get("id")
            chunk_ids.append(chunk_id)

            safe_text = sanitize_text_for_db(text)

            row = {
                "id": chunk_id,
                "file_id": file_id,
                "text": safe_text,
                "token_count": token_count,
                "metadata": metadata,
            }
            rows.append(row)

        resp = self.client.from_(self.table_chunks).insert(
            rows, upsert=True).execute()
        if hasattr(resp, "status_code") and resp.status_code not in (200, 201, 204):
            logger.error("add_chunks failed: %s %s", getattr(
                resp, "status_code", None), getattr(resp, "text", None))
            raise RuntimeError("Failed to insert chunks into Supabase")
        logger.info("Inserted/upserted %d chunks", len(rows))
        return chunk_ids

    def add_embeddings(self, chunk_ids: List[str], vectors: List[List[float]], models: Optional[List[str]] = None) -> List[str]:
        """
        Upsert embeddings rows corresponding to chunk_ids.
        Each row includes chunk_id, model, dimension, vector, provenance JSONB.
        """
        if not chunk_ids:
            return []

        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors length mismatch")

        rows = []
        for i, cid in enumerate(chunk_ids):
            vec = vectors[i]
            model_name = models[i] if models and i < len(
                models) else os.environ.get("EMBEDDING_MODEL_NAME", "unknown")
            row = {
                "chunk_id": cid,
                "model": model_name,
                "dimension": len(vec),
                "vector": vec,
                "provenance": {"model": model_name},
            }
            rows.append(row)

        resp = self.client.from_(self.table_embeddings).insert(
            rows, upsert=True).execute()
        if hasattr(resp, "status_code") and resp.status_code not in (200, 201, 204):
            logger.error("add_embeddings failed: %s %s", getattr(
                resp, "status_code", None), getattr(resp, "text", None))
            raise RuntimeError("Failed to insert embeddings into Supabase")
        logger.info("Inserted/upserted %d embeddings", len(rows))
        return [r.get("chunk_id") for r in rows]

    def upsert_chunks_and_embeddings(self, chunks: List[Dict[str, Any]], chunk_ids: List[str], vectors: List[List[float]], models: Optional[List[str]] = None) -> None:
        """
        Client-side transactional upsert (best-effort):
         1) upsert chunks (insert ... upsert=True)
         2) upsert embeddings (insert ... upsert=True)
         3) if embeddings upsert fails, attempt to delete embeddings and delete chunks (rollback)
        This is NOT a true DB transaction, but offers a safe rollback attempt.
        """
        if not chunks and not chunk_ids:
            return

        # Validate lengths
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors must have same length")

        # Validate embedding dims before touching DB
        for idx, v in enumerate(vectors):
            if v is None or not isinstance(v, (list, tuple)):
                raise ValueError(
                    f"Embedding at index {idx} is not a list/tuple")
            if len(v) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch at index {idx}: expected {self.embedding_dim} got {len(v)}")

        # Upsert chunks first
        try:
            self.add_chunks(chunks)
        except Exception as e:
            logger.error("Failed to upsert chunks: %s", e)
            raise

        # Then upsert embeddings. If this fails, attempt best-effort rollback by deleting embeddings and chunks.
        try:
            self.add_embeddings(chunk_ids, vectors, models=models)
        except Exception as e:
            logger.error(
                "Failed to upsert embeddings after chunks inserted: %s", e)
            # Best-effort rollback: delete any embeddings (partial) then delete chunks
            try:
                # delete embeddings that may have been partially inserted
                del_resp = self.client.from_(self.table_embeddings).delete().in_(
                    "chunk_id", chunk_ids).execute()
                logger.info("Attempted to delete embeddings for rollback (status=%s)", getattr(
                    del_resp, "status_code", None))
            except Exception as e2:
                logger.warning(
                    "Failed to delete embeddings during rollback attempt: %s", e2)

            try:
                del_chunks_resp = self.client.from_(
                    self.table_chunks).delete().in_("id", chunk_ids).execute()
                logger.info("Attempted to delete chunks for rollback (status=%s)", getattr(
                    del_chunks_resp, "status_code", None))
            except Exception as e3:
                logger.warning(
                    "Failed to delete chunks during rollback attempt: %s", e3)

            # Re-raise original exception to caller
            raise

    # Backwards-compatible alias
    def add_vectors(self, chunk_ids: List[str], vectors: List[List[float]], models: Optional[List[str]] = None):
        return self.add_embeddings(chunk_ids, vectors, models=models)

    def query(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        try:
            resp = self.client.rpc("vector_search_placeholder", {
                                   "query_embedding": vector, "limit": top_k})
            return resp.data or []
        except Exception as e:
            logger.warning(
                "RPC call to vector_search_placeholder failed or not implemented: %s", e)
            return []

    def delete(self, chunk_ids: List[str]):
        if not chunk_ids:
            return
        resp = self.client.from_(self.table_embeddings).delete().in_(
            "chunk_id", chunk_ids).execute()
        if hasattr(resp, "status_code") and resp.status_code not in (200, 204):
            logger.warning(
                "Failed to delete embeddings for chunk_ids: %s", getattr(resp, "text", None))
        resp2 = self.client.from_(self.table_chunks).delete().in_(
            "id", chunk_ids).execute()
        if hasattr(resp2, "status_code") and resp2.status_code not in (200, 204):
            logger.warning(
                "Failed to delete chunks for chunk_ids: %s", getattr(resp2, "text", None))
