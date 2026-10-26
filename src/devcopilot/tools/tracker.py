"""Tools over the issue tracker (SQLite). Published by `devcopilot/servers/tracker.py`."""

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import Field

from devcopilot.config import get_settings
from devcopilot.tools._common import ToolError, truncate

BOT = "devcopilot"
IdempotencyKey = Annotated[
    str, Field(description="Unique per logical operation; a retry with the same key is a no-op.")
]


def _connect(read_only: bool = True) -> sqlite3.Connection:
    db = get_settings().tracker_db
    if not db.exists():
        raise ToolError("Issue tracker is not initialized. Ask the user to run `devcopilot seed`.")
    # Read tools open the DB with mode=ro: even a buggy read tool cannot modify it.
    uri = f"file:{db}?mode=ro" if read_only else f"file:{db}?mode=rw"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _get_issue_row(conn: sqlite3.Connection, issue_id: int) -> sqlite3.Row:
    issue = conn.execute("SELECT * FROM issues WHERE id = ?", (issue_id,)).fetchone()
    if issue is None:
        raise ToolError(f"Issue #{issue_id} does not exist. Use search_issues to find issues.")
    return issue


def _idempotent(key: str, tool: str, write: Callable[[sqlite3.Connection], str]) -> str:
    """Run `write` at most once per key; a repeated key returns the first result.

    BEGIN IMMEDIATE takes the write lock up front, so two concurrent calls with
    the same key cannot both pass the "not seen yet" check.
    """
    with closing(_connect(read_only=False)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            done = conn.execute("SELECT result FROM operations WHERE idempotency_key = ?", (key,)).fetchone()
            if done:
                conn.execute("ROLLBACK")
                return json.loads(done["result"]) + " (already done earlier, not repeated)"
            result = write(conn)
            conn.execute("INSERT INTO operations VALUES (?, ?, ?, ?)", (key, tool, json.dumps(result), _now()))
            conn.execute("COMMIT")
            return result
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise


# ---------------------------------------------------------------- read tools


def search_issues(
    query: Annotated[str, Field(description='Text to find in title or body. "" returns all.')] = "",
    status: Annotated[Literal["open", "closed", "all"], Field(description="Filter by status.")] = "open",
) -> str:
    """Find issues whose title or body contains the query (case-insensitive substring match)."""
    sql = "SELECT id, title, status, labels, assignee FROM issues WHERE (title LIKE ? OR body LIKE ?)"
    params: list = [f"%{query}%", f"%{query}%"]
    if status != "all":
        sql += " AND status = ?"
        params.append(status)
    with closing(_connect()) as conn:
        rows = conn.execute(sql + " ORDER BY id", params).fetchall()
    if not rows:
        return f"No issues matching {query!r} (status={status})."
    return "\n".join(
        f"#{r['id']} [{r['status']}] {r['title']} (labels: {r['labels'] or '-'}; "
        f"assignee: {r['assignee'] or 'unassigned'})"
        for r in rows
    )


def get_issue(issue_id: Annotated[int, Field(description="Issue number without '#', e.g. 3.")]) -> str:
    """Get the full issue: description, metadata and comments."""
    with closing(_connect()) as conn:
        issue = _get_issue_row(conn, issue_id)
        comments = conn.execute(
            "SELECT author, body, created_at FROM comments WHERE issue_id = ? ORDER BY id",
            (issue_id,),
        ).fetchall()
    lines = [
        f"#{issue['id']} {issue['title']}",
        f"status: {issue['status']} | labels: {issue['labels'] or '-'} | "
        f"author: {issue['author']} | assignee: {issue['assignee'] or 'unassigned'}",
        f"created: {issue['created_at']} | updated: {issue['updated_at']}",
        "",
        issue["body"],
    ]
    if comments:
        lines += ["", "Comments:"]
        lines += [f"- {c['author']} ({c['created_at']}): {c['body']}" for c in comments]
    return truncate("\n".join(lines))


# --------------------------------------------------------------- write tools


def create_issue(
    title: Annotated[str, Field(description="Short summary of the problem.")],
    body: Annotated[str, Field(description="Details: symptoms, evidence (path:line, commits, CI runs).")],
    labels: Annotated[list[str], Field(description='E.g. ["bug", "p1"].')],
    idempotency_key: IdempotencyKey,
) -> str:
    """Create a new issue in the tracker."""

    def write(conn: sqlite3.Connection) -> str:
        now = _now()
        cur = conn.execute(
            "INSERT INTO issues (title, body, status, labels, author, assignee, created_at, updated_at)"
            " VALUES (?, ?, 'open', ?, ?, NULL, ?, ?)",
            (title, body, ",".join(labels), BOT, now, now),
        )
        return f"Created issue #{cur.lastrowid}: {title}"

    return _idempotent(idempotency_key, "create_issue", write)


def add_comment(
    issue_id: Annotated[int, Field(description="Issue number without '#'.")],
    body: Annotated[str, Field(description="Comment text.")],
    idempotency_key: IdempotencyKey,
) -> str:
    """Add a comment to an existing issue."""

    def write(conn: sqlite3.Connection) -> str:
        _get_issue_row(conn, issue_id)
        now = _now()
        conn.execute(
            "INSERT INTO comments (issue_id, author, body, created_at) VALUES (?, ?, ?, ?)",
            (issue_id, BOT, body, now),
        )
        conn.execute("UPDATE issues SET updated_at = ? WHERE id = ?", (now, issue_id))
        return f"Commented on issue #{issue_id}."

    return _idempotent(idempotency_key, "add_comment", write)


def update_issue(
    issue_id: Annotated[int, Field(description="Issue number without '#'.")],
    status: Annotated[Literal["open", "closed"] | None, Field(description="New status.")] = None,
    assignee: Annotated[str | None, Field(description="New assignee login.")] = None,
    labels: Annotated[list[str] | None, Field(description="Replaces all labels.")] = None,
) -> str:
    """Change status, assignee or labels of an issue. Setting the same value twice is harmless."""
    changes = {k: v for k, v in {"status": status, "assignee": assignee, "labels": labels}.items() if v is not None}
    if not changes:
        raise ToolError("Nothing to update: pass at least one of status, assignee, labels.")
    if "labels" in changes:
        changes["labels"] = ",".join(changes["labels"])
    with closing(_connect(read_only=False)) as conn:
        before = _get_issue_row(conn, issue_id)
        sets = ", ".join(f"{k} = ?" for k in changes)
        # Column names come from the fixed dict above, never from the caller.
        conn.execute(
            f"UPDATE issues SET {sets}, updated_at = ? WHERE id = ?",
            (*changes.values(), _now(), issue_id),
        )
    # Values as the reads show them ('-', 'unassigned'), not repr: None -> 'bob' was not
    # recognised by memory's fact extraction, and an empty label list looked like a change.
    diff = ", ".join(f"{k}: '{_shown(k, before[k])}' -> '{_shown(k, v)}'" for k, v in changes.items())
    return f"Updated issue #{issue_id}: {diff}"


def _shown(field: str, value: str | None) -> str:
    return value or {"labels": "-", "assignee": "unassigned"}.get(field, "")
