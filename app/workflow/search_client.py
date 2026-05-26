import json
import math
from typing import Any, Dict, List, Optional

from app.connectors.bedrock import BedrockConnector
from app.storage.sqlite_store import SQLiteStore


def build_index_id(job_id: str, run_id: str) -> str:
    return f"kb-{job_id}-{run_id}"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def search_knowledge_base(
    *,
    store: SQLiteStore,
    bedrock: BedrockConnector,
    index_id: str,
    query: str,
    top_k: int = 5,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query_embedding = await bedrock.generate_embedding(query)
    candidates = (
        store.get_chunks_for_documents(index_id, [document_id])
        if document_id
        else store.get_chunks_for_index(index_id)
    )
    scored: List[Dict[str, Any]] = []
    for c in candidates:
        emb = json.loads(c["embedding_json"])
        score = _cosine_similarity(query_embedding, emb)
        scored.append(
            {
                "score": score,
                "document_id": c["document_id"],
                "chunk_id": f'{c["document_id"]}-{c["chunk_index"]}',
                "file_name": c["file_name"],
                "chunk_text": c["chunk_text"],
                "chunk_index": c["chunk_index"],
                "metadata": json.loads(c["metadata_json"]),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


async def fetch_neighbour_chunks(
    *,
    store: SQLiteStore,
    index_id: str,
    retrieved: List[Dict[str, Any]],
    window: int = 1,
) -> List[Dict[str, Any]]:
    all_chunks = store.get_chunks_for_index(index_id)
    by_doc_idx = {
        (c["document_id"], c["chunk_index"]): c
        for c in all_chunks
    }
    merged: Dict[tuple, Dict[str, Any]] = {}
    for item in retrieved:
        key = (item["document_id"], item.get("chunk_index", 0))
        merged[key] = item
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            nkey = (item["document_id"], item.get("chunk_index", 0) + offset)
            row = by_doc_idx.get(nkey)
            if not row:
                continue
            merged.setdefault(
                nkey,
                {
                    "score": None,
                    "document_id": row["document_id"],
                    "chunk_id": f'{row["document_id"]}-{row["chunk_index"]}',
                    "file_name": row["file_name"],
                    "chunk_text": row["chunk_text"],
                    "chunk_index": row["chunk_index"],
                    "metadata": json.loads(row["metadata_json"]),
                },
            )
    out = list(merged.values())
    out.sort(key=lambda x: (str(x.get("document_id")), int(x.get("chunk_index", 0))))
    return out

