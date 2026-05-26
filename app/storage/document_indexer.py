import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from docx import Document
from pypdf import PdfReader

from app.config import Settings
from app.connectors.bedrock import BedrockConnector
from app.logging_setup import get_logger
from app.storage.sqlite_store import SQLiteStore

logger = get_logger(__name__)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_markdown(index=False)
    if suffix in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        blocks: List[str] = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            blocks.append(f"Sheet: {sheet}\n{df.head(200).to_markdown(index=False)}")
        return "\n\n".join(blocks)
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


async def index_document(
    *,
    store: SQLiteStore,
    bedrock: BedrockConnector,
    settings: Settings,
    file_id: str,
    file_name: str,
    local_path: Path,
    job_id: str,
    run_id: str,
) -> Dict[str, Any]:
    text = extract_text(local_path)
    pieces = chunk_text(text, settings.max_chunk_chars, settings.chunk_overlap_chars)
    index_id = f"kb-{job_id}-{run_id}"
    chunk_rows: List[Dict[str, Any]] = []
    for idx, piece in enumerate(pieces):
        embedding = await bedrock.generate_embedding(piece)
        chunk_rows.append(
            {
                "chunk_index": idx,
                "chunk_text": piece,
                "embedding": embedding,
                "metadata": {
                    "file_name": file_name,
                    "row_count": None,
                },
            }
        )
    store.replace_chunks_for_document(
        index_id=index_id,
        document_id=file_id,
        file_name=file_name,
        chunks=chunk_rows,
    )
    logger.info("Indexed file_id=%s chunks=%d index_id=%s", file_id, len(chunk_rows), index_id)
    return {
        "index_id": index_id,
        "chunk_count": len(chunk_rows),
        "text_size": len(text),
    }


def ensure_supported_file(path: Path) -> None:
    allowed = {".txt", ".md", ".csv", ".xlsx", ".xls", ".pdf", ".docx"}
    if path.suffix.lower() not in allowed:
        raise ValueError(
            f"Unsupported file type {path.suffix}. Allowed: {', '.join(sorted(allowed))}"
        )
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Empty file: {path.name}")

