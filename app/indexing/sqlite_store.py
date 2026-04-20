from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from app.indexing.interfaces import ChunkRepository
from app.processing.models import Chunk


class SQLiteChunkRepository(ChunkRepository):
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_content_hash
                ON documents(content_hash)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
                ON chunks(doc_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index
                ON chunks(chunk_index)
                """
            )

    def has_document(self, doc_id: str, content_hash: str) -> bool:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM documents
                WHERE doc_id = ? AND content_hash = ?
                LIMIT 1
                """,
                (doc_id, content_hash),
            ).fetchone()
            return row is not None

    def upsert_document(
        self,
        doc_id: str,
        source_path: str,
        file_type: str,
        content_hash: str,
        metadata: dict,
    ) -> None:
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, source_path, file_type, content_hash, metadata_json
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    file_type = excluded.file_type,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (doc_id, source_path, file_type, content_hash, metadata_json),
            )

    def replace_chunks(self, doc_id: str, chunks: List[Chunk]) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, chunk_index, text, metadata_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.chunk_index,
                        chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    )
                    for chunk in chunks
                ],
            )

    def count_chunks(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
            return int(row["count"])

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT chunk_id, doc_id, chunk_index, text, metadata_json
                FROM chunks
                WHERE chunk_id = ?
                LIMIT 1
                """,
                (chunk_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "metadata": json.loads(row["metadata_json"]),
            }