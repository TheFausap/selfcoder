# selfcoder/memory.py
"""SQLite-backed vector memory. One file, no server, no extra dependencies."""

from __future__ import annotations

import json
import sqlite3
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from selfcoder.chunker import Chunk
from selfcoder.embeddings import Embedder

KINDS = ("code", "analysis", "edit", "lesson", "note", "conversation")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    source       TEXT,
    label        TEXT,
    text         TEXT NOT NULL,
    metadata     TEXT,
    dim          INTEGER NOT NULL,
    embedding    BLOB NOT NULL,
    content_hash TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_kind   ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_dedup
    ON memories(kind, source, content_hash);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _pack(vec: Sequence[float]) -> bytes:
    return array("f", vec).tobytes()


def _unpack(blob: bytes) -> array:
    a = array("f")
    a.frombytes(blob)
    return a


def _cosine(a: array, b: array) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


@dataclass
class Hit:
    id: int
    kind: str
    source: str | None
    label: str | None
    text: str
    metadata: dict
    score: float
    created_at: float


class MemoryStore:
    """Filesystem-backed vector store: one SQLite file, no server."""

    def __init__(self, path: Path, embedder: Embedder):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ meta

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ write

    def add_many(self, items: list[dict], *, embed: bool = True) -> list[int | None]:
        if not items:
            return []
        texts = [item["text"] for item in items]
        vectors = self.embedder.embed(texts) if embed else [[0.0] for _ in texts]

        now = time.time()
        ids: list[int | None] = []
        for item, vector in zip(items, vectors):
            try:
                cursor = self.conn.execute(
                    """INSERT INTO memories
                       (kind, source, label, text, metadata, dim, embedding,
                        content_hash, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item["kind"],
                        item.get("source"),
                        item.get("label"),
                        item["text"],
                        json.dumps(item.get("metadata") or {}),
                        len(vector),
                        _pack(vector),
                        item.get("content_hash"),
                        now,
                        now,
                    ),
                )
                ids.append(cursor.lastrowid)
            except sqlite3.IntegrityError:
                ids.append(None)  # duplicate (kind, source, content_hash)
        self.conn.commit()
        return ids

    def add(
        self,
        kind: str,
        text: str,
        *,
        source: str | None = None,
        label: str | None = None,
        metadata: dict | None = None,
        content_hash: str | None = None,
    ) -> int | None:
        return self.add_many(
            [{
                "kind": kind,
                "text": text,
                "source": source,
                "label": label,
                "metadata": metadata or {},
                "content_hash": content_hash,
            }]
        )[0]

    def replace_source(self, kind: str, source: str, items: list[dict]) -> int:
        """Replace every memory under (kind, source). Used for incremental re-index."""
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE kind = ? AND source = ?", (kind, source)
        )
        removed = cursor.rowcount or 0
        for item in items:
            item.setdefault("kind", kind)
            item.setdefault("source", source)
        self.add_many(items)
        return removed

    def delete(
        self,
        *,
        ids: Iterable[int] | None = None,
        kind: str | None = None,
        source_like: str | None = None,
    ) -> int:
        clauses, params = [], []
        if ids:
            id_list = list(ids)
            clauses.append("id IN (" + ",".join("?" * len(id_list)) + ")")
            params.extend(id_list)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if source_like:
            clauses.append("source LIKE ?")
            params.append(source_like)
        if not clauses:
            raise ValueError("refusing to delete with no filter")
        cursor = self.conn.execute(
            f"DELETE FROM memories WHERE {' AND '.join(clauses)}", params
        )
        self.conn.commit()
        return cursor.rowcount or 0

    # ------------------------------------------------------------------ read

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def kind_counts(self) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT kind, COUNT(*) c FROM memories GROUP BY kind ORDER BY kind"
        ).fetchall()
        return [(row["kind"], row["c"]) for row in rows]

    def get(self, memory_id: int) -> Hit | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _row_to_hit(row, 1.0) if row else None

    def list(
        self,
        *,
        kind: str | None = None,
        source_like: str | None = None,
        limit: int = 50,
    ) -> list[Hit]:
        clauses, params = [], []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if source_like:
            clauses.append("source LIKE ?")
            params.append(source_like)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM memories {where} ORDER BY id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [_row_to_hit(row, 1.0) for row in rows]

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        kinds: Sequence[str] | None = None,
        source_like: str | None = None,
        min_score: float = 0.0,
    ) -> list[Hit]:
        clauses, params = ["dim > 0"], []
        if kinds:
            clauses.append("kind IN (" + ",".join("?" * len(kinds)) + ")")
            params.extend(kinds)
        if source_like:
            clauses.append("source LIKE ?")
            params.append(source_like)
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)}", params
        ).fetchall()
        if not rows:
            return []

        q = array("f", self.embedder.embed([query])[0])
        scored = [(_cosine(q, _unpack(row["embedding"])), row) for row in rows]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        hits: list[Hit] = []
        for score, row in scored[:k]:
            if score < min_score:
                break
            hits.append(_row_to_hit(row, score))
        return hits

    def close(self) -> None:
        self.conn.close()


def _row_to_hit(row: sqlite3.Row, score: float) -> Hit:
    return Hit(
        id=row["id"],
        kind=row["kind"],
        source=row["source"],
        label=row["label"],
        text=row["text"],
        metadata=json.loads(row["metadata"] or "{}"),
        score=score,
        created_at=row["created_at"],
    )


# --------------------------------------------------------------- indexing


def index_codebase(
    store: MemoryStore,
    files: dict[str, str],
    *,
    force: bool = False,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """Sync the `code` kind with the current file contents.

    Returns (updated_files, skipped_files, total_chunks).
    """
    import hashlib

    updated = skipped = chunk_count = 0

    for rel, content in files.items():
        if content.startswith("[") and content.rstrip().endswith("]"):
            continue  # omitted / unreadable placeholder

        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        meta_key = f"codehash::{rel}"
        if not force and store.get_meta(meta_key) == file_hash:
            skipped += 1
            continue

        chunks: list[Chunk] = chunk_file(rel, content)
        items = [
            {
                "kind": "code",
                "source": rel,
                "label": chunk.label,
                "text": f"{chunk.label}\n{chunk.text}",
                "metadata": {
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                },
                "content_hash": chunk.content_hash,
            }
            for chunk in chunks
        ]

        store.replace_source("code", rel, items)
        store.set_meta(meta_key, file_hash)
        updated += 1
        chunk_count += len(chunks)
        if verbose:
            print(f"  indexed {rel} ({len(chunks)} chunks)")

    return updated, skipped, chunk_count


def render_hits(hits: list[Hit], *, max_chars: int = 40_000) -> str:
    """Format retrieved memories for inclusion in a prompt."""
    parts: list[str] = []
    used = 0
    for hit in hits:
        header = f"[{hit.kind}]"
        if hit.source:
            header += f" {hit.source}"
        if hit.label:
            header += f" :: {hit.label}"
        header += f"  (score {hit.score:.3f})"
        block = f"{header}\n{hit.text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)
