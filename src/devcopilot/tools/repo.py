"""Tools over the service's git repository.

Plain functions: the MCP server in `devcopilot/servers/repo.py` publishes them.
Read tools never modify the repo. Write tools never touch `main`: they work in
temporary git worktrees and leave their result on a `copilot/*` branch.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from devcopilot.config import get_settings
from devcopilot.gitutil import GitError, git, identity
from devcopilot.tools._common import ToolError, minimal_env, resolve_inside, truncate

MAX_FILES = 200
MAX_READ_LINES = 300
MAX_MATCHES = 50
TEST_TIMEOUT_S = 60
# Commit-ish values the model may pass. Refuses anything starting with "-",
# so "--output=/etc/passwd" can never reach git as an option.
REF_RE = re.compile(r"^(HEAD(~\d+)?|[0-9a-fA-F]{4,40}|main|copilot/[a-z0-9][a-z0-9._-]{0,50})$")
BRANCH_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,50}$")
BOT = identity("devcopilot", "devcopilot@shortlink.dev")


def _repo() -> Path:
    repo = get_settings().repo_dir
    if not repo.exists():
        raise ToolError("Repository is not initialized. Ask the user to run `devcopilot seed`.")
    return repo


def _git(*args: str, env: dict[str, str] | None = None) -> str:
    try:
        return git(_repo(), *args, env=env)
    except GitError as e:
        raise ToolError(str(e)) from e


def _check_ref(ref: str) -> str:
    if not REF_RE.match(ref):
        raise ToolError(f"{ref!r} is not a valid ref. Use a commit hash, HEAD, HEAD~N, main or copilot/<name>.")
    return ref


def _is_hidden(rel: Path) -> bool:
    return ".git" in rel.parts or "__pycache__" in rel.parts


@contextmanager
def _worktree(ref: str, new_branch: str | None = None) -> Iterator[Path]:
    """Check out `ref` into a temporary directory; the main checkout stays untouched."""
    root = get_settings().workspace_dir / "worktrees"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=root))
    args = ["worktree", "add", "-q"]
    args += ["-b", new_branch, str(path), ref] if new_branch else ["--detach", str(path), ref]
    _git(*args)
    try:
        yield path
    finally:
        _git("worktree", "remove", "--force", str(path))
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------- read tools


def list_files(
    directory: Annotated[str, Field(description='Subdirectory, e.g. "shortlink" or "docs".')] = ".",
) -> str:
    """List files of the repository (recursively), relative to the repository root."""
    root = _repo()
    base = resolve_inside(root, directory)
    if not base.is_dir():
        raise ToolError(f"{directory!r} is not a directory. Call list_files() without arguments.")
    files = sorted(
        p.relative_to(root).as_posix() for p in base.rglob("*") if p.is_file() and not _is_hidden(p.relative_to(root))
    )
    if len(files) > MAX_FILES:
        files = files[:MAX_FILES] + [f"... and {len(files) - MAX_FILES} more"]
    return "\n".join(files) or "(empty)"


def read_file(
    path: Annotated[str, Field(description='Path from the repository root, e.g. "shortlink/api.py".')],
    start_line: Annotated[int, Field(description="First line to return (1-based).")] = 1,
    end_line: Annotated[
        int | None, Field(description="Last line to return (inclusive). Defaults to start_line + 299.")
    ] = None,
) -> str:
    """Read a text file from the repository (main branch). Lines are prefixed with their numbers."""
    file = resolve_inside(_repo(), path)
    if not file.is_file():
        raise ToolError(f"File {path!r} not found. Use list_files or search_code to find it.")
    lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(start_line, 1)
    end = min(end_line or start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1, len(lines))
    body = "\n".join(f"{i:>4} | {lines[i - 1]}" for i in range(start, end + 1))
    footer = f"\n[lines {start}-{end} of {len(lines)}]" if (start > 1 or end < len(lines)) else ""
    return f"{path}\n{body}{footer}"


def search_code(
    pattern: Annotated[str, Field(description=r'Python regex, e.g. "def is_expired" or "datetime\.now".')],
    path_glob: Annotated[str, Field(description='Only search matching files, e.g. "**/*.py".')] = "**/*",
) -> str:
    """Search repository files with a regular expression. Returns `path:line: text` matches."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"Invalid regex {pattern!r}: {e}. Escape special characters.") from e
    # anchor, not is_absolute(): on Windows "/etc/*" has a root but no drive, so it is not
    # "absolute" there, and glob() then raises instead of answering the model.
    if Path(path_glob).anchor or ".." in Path(path_glob).parts:
        raise ToolError(f"path_glob {path_glob!r} must stay inside the repository: no '..', no absolute paths.")
    root = _repo()
    real_root = root.resolve()
    matches = []
    for file in sorted(root.glob(path_glob)):
        # glob follows '..' and symlinks, and relative_to compares text: check the real path
        # (found by review: "../../.env" read files outside the repository without approval).
        if not file.resolve().is_relative_to(real_root):
            continue
        rel = file.relative_to(root)
        if not file.is_file() or _is_hidden(rel):
            continue
        for lineno, line in enumerate(file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if regex.search(line):
                # as_posix: the same `path:line` on Windows, for the model and the grounding check.
                matches.append(f"{rel.as_posix()}:{lineno}: {line.strip()}")
    if not matches:
        return f"No matches for {pattern!r} in {path_glob!r}."
    if len(matches) > MAX_MATCHES:
        extra = len(matches) - MAX_MATCHES
        matches = matches[:MAX_MATCHES] + [f"... {extra} more matches, refine the pattern"]
    return "\n".join(matches)


def git_log(
    path: Annotated[str | None, Field(description='Only commits touching this path, e.g. "shortlink".')] = None,
    limit: Annotated[int, Field(description="Maximum number of commits.")] = 10,
) -> str:
    """Show recent commits on main (hash, date, author, message)."""
    args = ["log", f"-n{min(limit, 50)}", "--date=short", "--format=%h %ad %an: %s"]
    if path:
        resolve_inside(_repo(), path)
        args += ["--", path]
    return _git(*args).strip() or "No commits found."


def git_show(
    commit: Annotated[str, Field(description="Commit hash, HEAD, HEAD~N or a copilot/<name> branch.")],
) -> str:
    """Show a commit: metadata, changed files and the diff."""
    return truncate(_git("show", "--stat", "--patch", "--date=iso", _check_ref(commit)))


# ------------------------------------------------------------ side effects


def run_tests(
    ref: Annotated[str, Field(description="What to test: main, a commit hash or copilot/<name>.")] = "main",
    test_path: Annotated[str | None, Field(description='Optional subset, e.g. "tests/test_expiry.py".')] = None,
) -> str:
    """Run the pytest suite of the repository at a given ref and return the result summary.

    Runs in a temporary checkout, never modifies branches. Takes a few seconds.
    """
    ref = _check_ref(ref)
    with _worktree(ref) as checkout:
        args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        if test_path:
            args.append(resolve_inside(checkout, test_path).relative_to(checkout).as_posix())
        # Repository code is untrusted: minimal env (no secrets), hard timeout.
        # A real system would also isolate it in a container without network.
        env = minimal_env(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        try:
            result = subprocess.run(
                args, cwd=checkout, env=env, capture_output=True, encoding="utf-8", errors="replace",
                timeout=TEST_TIMEOUT_S,
            )  # fmt: skip
        except subprocess.TimeoutExpired as e:
            raise ToolError(f"Tests did not finish in {TEST_TIMEOUT_S}s.") from e
    status = "PASSED" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    output = "\n".join(result.stdout.splitlines()[-60:])
    return truncate(f"Tests at {ref}: {status}\n\n{output}")


class TextEdit(BaseModel):
    old_text: str = Field(description="Exact text to replace, copied from the file. Must occur exactly once.")
    new_text: str = Field(description="Replacement text.")


def propose_patch(
    branch: Annotated[str, Field(description='Short name, e.g. "fix-naive-datetime". Becomes copilot/<branch>.')],
    path: Annotated[str, Field(description="File to change, relative to the repository root.")],
    edits: Annotated[list[TextEdit], Field(description="Replacements applied in order.", min_length=1)],
    commit_message: Annotated[str, Field(description="Commit message describing the fix.")],
) -> str:
    """Create branch copilot/<branch> from main with edits to one file committed. Main is not changed.

    Idempotent by branch name: if the branch already exists, nothing is changed and its diff is returned.
    """
    if not BRANCH_RE.match(branch):
        raise ToolError(f"Invalid branch name {branch!r}. Use lowercase letters, digits, '-', '.', '_'.")
    full = f"copilot/{branch}"
    exists = bool(_git("branch", "--list", full).strip())
    # Validate everything before creating the branch, so a failure leaves nothing behind.
    resolve_inside(_repo(), path)
    try:
        text = _git("show", f"main:{path}")
    except ToolError:
        raise ToolError(f"File {path!r} not found on main.") from None
    for i, edit in enumerate(edits, 1):
        count = text.count(edit.old_text)
        if count != 1:
            raise ToolError(
                f"Edit {i}: old_text occurs {count} times in {path}, expected exactly 1. "
                "Read the file and copy the exact text, including indentation. No changes were made."
            )
        text = text.replace(edit.old_text, edit.new_text)
    if text == _git("show", f"main:{path}"):
        # Checked before the branch exists: "nothing to commit" left an empty branch behind,
        # and every retry then answered "already exists" with an empty diff.
        raise ToolError("The edits change nothing: old_text equals new_text. No changes were made.")
    if exists:
        # The name alone cannot prove completion after an interrupted older operation.
        if _git("show", f"{full}:{path}") != text or not _git("diff", f"main...{full}").strip():
            raise ToolError(f"Branch {full} exists but does not contain the intended patch. Use a new branch name.")
        return f"Branch {full} already exists, no changes made.\n\n" + truncate(_git("diff", f"main...{full}"))
    try:
        with _worktree("main", new_branch=full) as checkout:
            # Preserve LF on Windows to avoid rewriting every line of the file.
            (checkout / path).write_text(text, encoding="utf-8", newline="")
            git(checkout, "commit", "-q", "-am", commit_message, env=BOT)
    except BaseException:
        if _git("branch", "--list", full).strip():
            _git("branch", "-D", full)
        raise
    return f"Created branch {full}.\n\n" + truncate(_git("diff", f"main...{full}"))
