"""SQLite-backed storage for short links. See docs/adr/0001-sqlite-storage.md."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    created_at TEXT NOT NULL,
    ttl_hours INTEGER
)
"""


@dataclass
class Link:
    id: int
    url: str
    created_at: datetime  # timezone-aware UTC
    ttl_hours: int | None


class LinkStore:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.execute(SCHEMA)

    def add(self, url: str, ttl_hours: int | None = None) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO links (url, created_at, ttl_hours) VALUES (?, ?, ?)",
            (url, created_at, ttl_hours),
        )
        self.conn.commit()
        return cur.lastrowid

    def get(self, link_id: int) -> Link | None:
        row = self.conn.execute(
            "SELECT id, url, created_at, ttl_hours FROM links WHERE id = ?", (link_id,)
        ).fetchone()
        if row is None:
            return None
        return Link(row[0], row[1], datetime.fromisoformat(row[2]), row[3])
