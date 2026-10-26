"""Build the workspace from the fixtures in `data/`.

Layout produced:
    workspace/repo/       git repository of the `shortlink` service (4 commits)
    workspace/tracker.db  SQLite issue tracker
    workspace/ci/         CI runs (runs.json) and their logs

Commit authors and dates are fixed, so commit hashes are the same on every seed.
That keeps CI logs, eval datasets and docs referring to real hashes.
"""

import json
import os
import shutil
import sqlite3
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template

from devcopilot.gitutil import git, identity

DATA_DIR = Path(__file__).parent / "data"
MARKER = ".devcopilot-workspace"


@dataclass(frozen=True)
class Commit:
    key: str  # used as $key placeholder in CI logs
    snapshot: str  # directory under data/shortlink/commits with changed files
    author: str
    email: str
    date: str
    message: str


HISTORY = [
    Commit(
        "c1", "01-initial", "alice", "alice@shortlink.dev", "2026-09-01T10:00:00+00:00",
        "Initial version: base62 codes and SQLite store",
    ),
    Commit(
        "c2", "02-expiry", "bob", "bob@shortlink.dev", "2026-09-10T16:30:00+00:00",
        "feat: optional link TTL (closes #1)",
    ),
    Commit(
        "c3", "03-ratelimit", "carol", "carol@shortlink.dev", "2026-09-20T11:45:00+00:00",
        "feat: per-client rate limiting on shorten()",
    ),
    Commit(
        "c4", "04-injectable-clock", "bob", "bob@shortlink.dev", "2026-10-15T17:20:00+00:00",
        "refactor: allow injecting clock into is_expired for tests",
    ),
]  # fmt: skip

TRACKER_SCHEMA = """
CREATE TABLE issues (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    labels TEXT NOT NULL,          -- comma-separated
    author TEXT NOT NULL,
    assignee TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL REFERENCES issues(id),
    author TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
-- Results of non-idempotent write operations, keyed by the caller's idempotency key.
-- A retried call with the same key returns the stored result instead of writing again.
CREATE TABLE operations (
    idempotency_key TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def seed(workspace: Path) -> dict[str, str]:
    """Recreate the workspace from scratch. Returns {commit key: commit hash}."""
    _reset_dir(workspace)
    hashes = _build_repo(workspace / "repo")
    _build_tracker(workspace / "tracker.db")
    _build_ci(workspace / "ci", hashes)
    return hashes


def _reset_dir(workspace: Path) -> None:
    if workspace.exists():
        # Refuse to wipe a directory we did not create.
        if not (workspace / MARKER).exists() and any(workspace.iterdir()):
            raise RuntimeError(f"{workspace} exists and is not a devcopilot workspace")
        _rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / MARKER).touch()


def _rmtree(path: Path, attempts: int = 5) -> None:
    """shutil.rmtree that also works on Windows.

    Git marks its object files read-only, and Windows refuses to delete read-only files
    (PermissionError, WinError 5): clear the flag and retry. A file can also be briefly
    held by a tool server that is still exiting after the previous run: wait and retry.
    """

    def clear_readonly(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    for attempt in range(attempts):
        try:
            shutil.rmtree(path, onexc=clear_readonly)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def _build_repo(repo: Path) -> dict[str, str]:
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    hashes = {}
    for commit in HISTORY:
        snapshot = DATA_DIR / "shortlink" / "commits" / commit.snapshot
        # copy, not copytree's copy2: file modes must not come from the host. On a
        # Windows folder mounted into a Linux sandbox every file looks executable, git
        # stores 100755, and every commit hash differs from the dataset's (329311c...).
        shutil.copytree(snapshot, repo, dirs_exist_ok=True, copy_function=shutil.copy)
        for f in repo.rglob("*"):
            if f.is_file() and ".git" not in f.relative_to(repo).parts:
                f.chmod(0o644)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", commit.message,
            env=identity(commit.author, commit.email, commit.date))  # fmt: skip
        hashes[commit.key] = git(repo, "rev-parse", "--short=7", "HEAD").strip()
    return hashes


def _build_tracker(db_path: Path) -> None:
    data = json.loads((DATA_DIR / "tracker.json").read_text(encoding="utf-8"))
    conn = sqlite3.connect(db_path)
    conn.executescript(TRACKER_SCHEMA)
    for issue in data["issues"]:
        conn.execute(
            "INSERT INTO issues VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                issue["id"], issue["title"], issue["body"], issue["status"],
                ",".join(issue["labels"]), issue["author"], issue["assignee"],
                issue["created_at"], issue["updated_at"],
            ),
        )  # fmt: skip
        for c in issue["comments"]:
            conn.execute(
                "INSERT INTO comments (issue_id, author, body, created_at) VALUES (?, ?, ?, ?)",
                (issue["id"], c["author"], c["body"], c["created_at"]),
            )
    conn.commit()
    conn.close()


def _build_ci(ci_dir: Path, hashes: dict[str, str]) -> None:
    src = DATA_DIR / "ci"
    (ci_dir / "logs").mkdir(parents=True)
    runs = json.loads(Template((src / "runs.json").read_text(encoding="utf-8")).substitute(hashes))
    (ci_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    for log in (src / "logs").glob("*.log"):
        text = Template(log.read_text(encoding="utf-8")).safe_substitute(hashes)
        (ci_dir / "logs" / log.name).write_text(text, encoding="utf-8")
