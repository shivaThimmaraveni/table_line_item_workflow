import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _conn(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_runs_session
                ON workflow_runs (session_id);

                CREATE TABLE IF NOT EXISTS workflow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    request_id TEXT,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_events_session
                ON workflow_events (session_id);

                CREATE TABLE IF NOT EXISTS conversation_checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    file_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    file_ext TEXT,
                    file_size INTEGER,
                    content_hash TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_job_run
                ON documents (job_id, run_id);

                CREATE TABLE IF NOT EXISTS retrieval_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    index_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_index
                ON retrieval_chunks (index_id);

                CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_doc
                ON retrieval_chunks (document_id, chunk_index);

                CREATE TABLE IF NOT EXISTS retrieval_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    index_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    retrieval_scope TEXT NOT NULL,
                    matched_files_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_run(self, job_id: str, run_id: str, session_id: str, status: str) -> None:
        now = _utc_now()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM workflow_runs WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE workflow_runs
                    SET status = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (status, now, session_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO workflow_runs (job_id, run_id, session_id, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, run_id, session_id, status, now, now),
                )

    def log_event(
        self,
        session_id: str,
        request_id: Optional[str],
        event_type: str,
        event_data: Dict[str, Any],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO workflow_events (session_id, request_id, event_type, event_data, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, request_id, event_type, json.dumps(event_data), _utc_now()),
            )

    def save_checkpoint(
        self,
        thread_id: str,
        messages_json: str,
        state_json: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO conversation_checkpoints (thread_id, messages_json, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id)
                DO UPDATE SET
                  messages_json = excluded.messages_json,
                  state_json = excluded.state_json,
                  updated_at = excluded.updated_at
                """,
                (thread_id, messages_json, state_json, _utc_now()),
            )

    def get_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT thread_id, messages_json, state_json, updated_at
                FROM conversation_checkpoints
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def add_document(
        self,
        *,
        file_id: str,
        job_id: str,
        run_id: str,
        file_name: str,
        local_path: str,
        file_ext: str,
        file_size: int,
        content_hash: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (file_id, job_id, run_id, file_name, local_path, file_ext, file_size, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    job_id,
                    run_id,
                    file_name,
                    local_path,
                    file_ext,
                    file_size,
                    content_hash,
                    _utc_now(),
                ),
            )

    def get_documents(self, job_id: str, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM documents
                WHERE job_id = ? AND run_id = ?
                ORDER BY created_at ASC
                """,
                (job_id, run_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_documents_by_ids(
        self,
        job_id: str,
        run_id: str,
        file_ids: List[str],
    ) -> List[Dict[str, Any]]:
        if not file_ids:
            return self.get_documents(job_id, run_id)
        placeholders = ",".join("?" for _ in file_ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM documents
                WHERE job_id = ? AND run_id = ? AND file_id IN ({placeholders})
                ORDER BY created_at ASC
                """,
                (job_id, run_id, *file_ids),
            ).fetchall()
            return [dict(r) for r in rows]

    def replace_chunks_for_document(
        self,
        *,
        index_id: str,
        document_id: str,
        file_name: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                DELETE FROM retrieval_chunks
                WHERE index_id = ? AND document_id = ?
                """,
                (index_id, document_id),
            )
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO retrieval_chunks
                    (index_id, document_id, file_name, chunk_index, chunk_text, embedding_json, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        index_id,
                        document_id,
                        file_name,
                        int(chunk["chunk_index"]),
                        chunk["chunk_text"],
                        json.dumps(chunk["embedding"]),
                        json.dumps(chunk.get("metadata", {})),
                        _utc_now(),
                    ),
                )

    def get_chunks_for_index(self, index_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM retrieval_chunks
                WHERE index_id = ?
                ORDER BY document_id ASC, chunk_index ASC
                """,
                (index_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_chunks_for_documents(self, index_id: str, document_ids: List[str]) -> List[Dict[str, Any]]:
        if not document_ids:
            return self.get_chunks_for_index(index_id)
        placeholders = ",".join("?" for _ in document_ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM retrieval_chunks
                WHERE index_id = ? AND document_id IN ({placeholders})
                ORDER BY document_id ASC, chunk_index ASC
                """,
                (index_id, *document_ids),
            ).fetchall()
            return [dict(r) for r in rows]

    def log_retrieval_query(
        self,
        *,
        session_id: str,
        index_id: str,
        query_text: str,
        top_k: int,
        retrieval_scope: str,
        matched_files: List[str],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_queries
                (session_id, index_id, query_text, top_k, retrieval_scope, matched_files_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    index_id,
                    query_text,
                    top_k,
                    retrieval_scope,
                    json.dumps(matched_files),
                    _utc_now(),
                ),
            )

