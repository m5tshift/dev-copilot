"""SQLite-backed storage for short links. See docs/adr/0001-sqlite-storage.md."""

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL
)
"""


class LinkStore:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.execute(SCHEMA)

    def add(self, url: str) -> int:
        cur = self.conn.execute("INSERT INTO links (url) VALUES (?)", (url,))
        self.conn.commit()
        return cur.lastrowid

    def get(self, link_id: int) -> str | None:
        row = self.conn.execute("SELECT url FROM links WHERE id = ?", (link_id,)).fetchone()
        return row[0] if row else None
