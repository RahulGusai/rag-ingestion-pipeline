# utils/parse_bytes.py
import io
import logging
import csv
from typing import List, Dict, Any

# unstructured partitioners
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text

# pypdf fallback
from pypdf import PdfReader

# python-docx fallback for docx
from docx import Document as DocxReader

logger = logging.getLogger("ingestion-pipeline.parse_bytes")


# ---------------------------
# PDF parsing (left essentially unchanged)
# ---------------------------
def parse_pdf_bytes(file_bytes: bytes) -> str:
    """
    Primary: try unstructured.partition_pdf, fallback to pypdf PdfReader.
    (Function copy of previous behavior — kept untouched logically.)
    """
    try:
        with io.BytesIO(file_bytes) as b:
            elements = partition_pdf(file=b)
            text = "\n\n".join([e.get_text()
                               for e in elements if hasattr(e, "get_text")])
            if text and text.strip():
                logger.info(
                    "parse_pdf_bytes: extracted %d chars via unstructured.partition_pdf", len(text))
                return text
    except Exception as e:
        logger.info("parse_pdf_bytes: unstructured.partition_pdf failed: %s", e)

    # fallback to pypdf text extraction
    try:
        with io.BytesIO(file_bytes) as b:
            reader = PdfReader(b)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n\n".join(pages)
            logger.info(
                "parse_pdf_bytes: extracted %d chars via pypdf fallback", len(text))
            return text
    except Exception as e:
        logger.exception("parse_pdf_bytes: pypdf fallback failed")
        return ""


# ---------------------------
# DOCX parsing (partition_docx first, then python-docx fallback)
# ---------------------------
def parse_docx_bytes(file_bytes: bytes) -> str:
    """
    DOCX parsing pipeline (MVP):
      1) Attempt unstructured.partition_docx
      2) If the joined text is empty, fallback to python-docx paragraph & table extraction
    Logs which method succeeded.
    """
    if not isinstance(file_bytes, (bytes, bytearray)):
        logger.error("parse_docx_bytes: expected bytes, got %s",
                     type(file_bytes))
        raise ValueError("parse_docx_bytes expects raw bytes")

    # helper to get fresh BytesIO
    def fresh_bio():
        b = io.BytesIO(file_bytes)
        b.seek(0)
        return b

    # 1) Try unstructured.partition_docx
    try:
        with fresh_bio() as b:
            b.seek(0)
            elements = partition_docx(file=b)
            elem_count = len(elements) if elements is not None else 0
            logger.info(
                "parse_docx_bytes: partition_docx returned %d elements", elem_count)

            texts = []
            for e in elements:
                if hasattr(e, "get_text"):
                    try:
                        t = e.get_text() or ""
                    except Exception:
                        t = ""
                    if t and t.strip():
                        texts.append(t.strip())
            joined = "\n\n".join(texts)
            if joined and joined.strip():
                logger.info(
                    "parse_docx_bytes: returning %d chars from unstructured.partition_docx", len(joined))
                return joined
            else:
                logger.info(
                    "parse_docx_bytes: unstructured.partition_docx returned empty text, will fallback to python-docx")
    except Exception as e:
        logger.exception(
            "parse_docx_bytes: partition_docx invocation failed: %s", e)

    # 2) Fallback: python-docx (paragraphs + table cells)
    try:
        with fresh_bio() as b2:
            b2.seek(0)
            doc = DocxReader(b2)
            paras: List[str] = []
            for p in doc.paragraphs:
                if p.text and p.text.strip():
                    paras.append(p.text.strip())
            # attempt to extract table cell text as well (common in contracts)
            try:
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            if cell.text and cell.text.strip():
                                paras.append(cell.text.strip())
            except Exception:
                # Non-fatal: log and continue
                logger.exception(
                    "parse_docx_bytes: python-docx table extraction encountered an error (continuing)")

            fallback_text = "\n\n".join(paras).strip()
            if fallback_text:
                logger.info(
                    "parse_docx_bytes: returning %d chars from python-docx fallback", len(fallback_text))
                return fallback_text
            else:
                logger.warning(
                    "parse_docx_bytes: python-docx fallback produced empty text")
    except Exception as e:
        logger.exception(
            "parse_docx_bytes: python-docx fallback invocation failed: %s", e)

    # nothing worked
    logger.error("parse_docx_bytes: all methods returned empty text")
    return ""


# ---------------------------
# TEXT parsing (partition_text first, then robust decoding fallback)
# ---------------------------
def parse_text_bytes(file_bytes: bytes) -> str:
    """
    Attempt unstructured.partition_text first (for text-based formats).
    If that yields empty text, attempt decoding bytes using several encodings.
    Logs which path succeeded.
    """
    if not isinstance(file_bytes, (bytes, bytearray)):
        logger.error("parse_text_bytes: expected bytes, got %s",
                     type(file_bytes))
        raise ValueError("parse_text_bytes expects raw bytes")

    # helper: fresh IO
    def fresh_bio():
        b = io.BytesIO(file_bytes)
        b.seek(0)
        return b

    # 1) unstructured partition_text
    try:
        with fresh_bio() as b:
            b.seek(0)
            elements = partition_text(file=b)
            logger.info("parse_text_bytes: partition_text returned %d elements", len(
                elements) if elements is not None else 0)
            texts = []
            for e in elements:
                if hasattr(e, "get_text"):
                    try:
                        t = e.get_text() or ""
                    except Exception:
                        t = ""
                    if t and t.strip():
                        texts.append(t.strip())
            joined = "\n\n".join(texts)
            if joined and joined.strip():
                logger.info(
                    "parse_text_bytes: returning %d chars from unstructured.partition_text", len(joined))
                return joined
            else:
                logger.info(
                    "parse_text_bytes: unstructured.partition_text returned empty text, will try raw decode fallbacks")
    except Exception as e:
        logger.exception(
            "parse_text_bytes: partition_text invocation failed: %s", e)

    # 2) Fallback: attempt common encodings
    encodings_to_try = ["utf-8", "utf-16", "latin-1"]
    for enc in encodings_to_try:
        try:
            decoded = file_bytes.decode(enc)
            if decoded and decoded.strip():
                logger.info(
                    "parse_text_bytes: decoded bytes successfully with encoding=%s (%d chars)", enc, len(decoded))
                return decoded
            else:
                logger.info(
                    "parse_text_bytes: decoded with %s but result empty", enc)
        except Exception:
            logger.info("parse_text_bytes: decoding with %s failed", enc)

    # 3) Last resort: replace-errors decode (preserves content)
    try:
        decoded = file_bytes.decode("utf-8", errors="replace")
        if decoded and decoded.strip():
            logger.info(
                "parse_text_bytes: returning text via utf-8 replace fallback (%d chars)", len(decoded))
            return decoded
    except Exception:
        logger.exception("parse_text_bytes: utf-8 replace fallback failed")

    logger.error(
        "parse_text_bytes: all extraction/decoding methods returned empty text")
    return ""


# ---------------------------
# CSV parsing (improved by adding encoding fallback)
# ---------------------------
def parse_csv_bytes(file_bytes: bytes) -> str:
    """
    Decode CSV bytes using a set of encodings, then parse using csv module
    and return a newline-joined string representation. Logs which encoding succeeded.
    """
    if not isinstance(file_bytes, (bytes, bytearray)):
        logger.error("parse_csv_bytes: expected bytes, got %s",
                     type(file_bytes))
        raise ValueError("parse_csv_bytes expects raw bytes")

    encodings_to_try = ["utf-8", "utf-16", "latin-1"]
    text = None
    used_enc = None
    for enc in encodings_to_try:
        try:
            t = file_bytes.decode(enc)
            if t and t.strip():
                text = t
                used_enc = enc
                logger.info("parse_csv_bytes: decoded using encoding=%s", enc)
                break
        except Exception:
            logger.info("parse_csv_bytes: decode with %s failed", enc)

    # fallback to utf-8 replace if nothing worked
    if text is None:
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            used_enc = "utf-8-replace"
            logger.info("parse_csv_bytes: used utf-8 replace fallback")
        except Exception:
            logger.exception(
                "parse_csv_bytes: all decodings failed; returning empty")
            return ""

    # parse CSV and return a readable text representation
    try:
        reader = csv.reader(io.StringIO(text))
        rows = []
        for row in reader:
            # join by comma, but keep minimal escaping by wrapping fields if they contain comma
            safe_row = []
            for fld in row:
                if "," in fld:
                    safe_row.append(f'"{fld}"')
                else:
                    safe_row.append(fld)
            rows.append(",".join(safe_row))
        out = "\n".join(rows)
        logger.info(
            "parse_csv_bytes: parsed %d CSV rows (encoding=%s)", len(rows), used_enc)
        return out
    except Exception:
        logger.exception(
            "parse_csv_bytes: csv parsing failed; returning raw decoded text")
        return text or ""
