"""Content-addressed disk cache for paid or slow calls.

Keyed by a hash of the exact request, so an interrupted sweep resumes instead of
re-billing, and two configurations that share an embedding or a judgement only
pay for it once.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key   TEXT PRIMARY KEY,
    kind  TEXT NOT NULL,
    model TEXT NOT NULL,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_kind_model ON entries (kind, model);
"""


class DiskCache:
    """A small SQLite-backed key/value store, safe to share across threads."""

    def __init__(self, directory: str | Path = ".rageval_cache") -> None:
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path / "cache.sqlite", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @staticmethod
    def _key(kind: str, model: str, payload: str) -> str:
        digest = hashlib.sha256()
        digest.update(kind.encode())
        digest.update(b"\x00")
        digest.update(model.encode())
        digest.update(b"\x00")
        digest.update(payload.encode())
        return digest.hexdigest()

    def get(self, kind: str, model: str, payload: str) -> dict[str, Any] | None:
        key = self._key(kind, model, payload)
        with self._lock:
            row = self._conn.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        result: dict[str, Any] = json.loads(row[0])
        return result

    def put(self, kind: str, model: str, payload: str, value: dict[str, Any]) -> None:
        key = self._key(kind, model, payload)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries (key, kind, model, value) VALUES (?, ?, ?, ?)",
                (key, kind, model, json.dumps(value)),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
