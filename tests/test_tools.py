"""Tool functions called directly, without MCP or an LLM."""

import pytest

from devcopilot.gitutil import git
from devcopilot.tools._common import ToolError
from devcopilot.tools.ci import get_ci_log, list_ci_runs
from devcopilot.tools.repo import (
    TextEdit,
    git_log,
    git_show,
    list_files,
    propose_patch,
    read_file,
    run_tests,
    search_code,
)
from devcopilot.tools.tracker import add_comment, create_issue, get_issue, search_issues, update_issue

# ------------------------------------------------------------------ repo reads


def test_list_files_skips_git_dir():
    out = list_files()
    assert "shortlink/expiry.py" in out
    assert ".git" not in out


def test_read_file_numbers_lines():
    assert "  12 |     now = now or datetime.now()" in read_file("shortlink/expiry.py")


def test_read_file_range():
    out = read_file("shortlink/expiry.py", start_line=10, end_line=11)
    assert "[lines 10-11 of 13]" in out
    assert "   9 |" not in out


@pytest.mark.parametrize("path", ["../../etc/passwd", "/../../etc/passwd"])
def test_read_file_blocks_path_traversal(path):
    with pytest.raises(ToolError, match="outside the repository"):
        read_file(path)


def test_search_code_finds_bug_line():
    assert "shortlink/expiry.py:12:" in search_code(r"datetime\.now\(\)", "**/*.py")


@pytest.mark.parametrize("glob", ["../*", "../../.env", "**/../../*", "/etc/*"])
def test_search_code_stays_inside_the_repository(glob):
    """Regression (security): glob follows '..', so "../../.env" read files next to the
    repository (.env, the Telegram token) through a read-only tool, with no approval."""
    with pytest.raises(ToolError, match="inside the repository"):
        search_code(".", glob)


def test_search_code_skips_symlinks_out_of_the_repository(fresh_workspace, tmp_path):
    secret = tmp_path / "secret.env"
    secret.write_text("TOKEN=abc123\n", encoding="utf-8")
    (fresh_workspace / "repo" / "leak.env").symlink_to(secret)
    assert "abc123" not in search_code("TOKEN")


def test_search_code_reports_bad_regex():
    with pytest.raises(ToolError, match="Invalid regex"):
        search_code("(")


def test_git_log_for_path(commits):
    out = git_log("shortlink/expiry.py")
    assert out.splitlines()[0].startswith(commits["c4"])
    assert commits["c3"] not in out  # c3 did not touch expiry.py


@pytest.mark.parametrize("ref", ["--output=/tmp/pwned", "main; rm -rf /", "copilot/../../x"])
def test_git_show_rejects_bad_refs(ref):
    with pytest.raises(ToolError, match="not a valid ref"):
        git_show(ref)


def test_git_show_contains_diff(commits):
    out = git_show(commits["c4"])
    assert "-    return datetime.now(timezone.utc) >= expires_at(created_at, ttl_hours)" in out


# --------------------------------------------------------------- tracker reads


def test_search_issues_open_by_default():
    out = search_issues("with TTL")
    assert "#3 [open]" in out
    assert "#1" not in out  # closed


def test_get_issue_with_comments():
    out = get_issue(3)
    assert "HTTP 500" in out and "alice" in out


def test_get_missing_issue():
    with pytest.raises(ToolError, match="does not exist"):
        get_issue(999)


# -------------------------------------------------------------------- CI reads


def test_ci_runs_newest_first(commits):
    first = list_ci_runs(status="failed").splitlines()[0]
    assert first.startswith("run 105: FAILED") and commits["c4"] in first


def test_ci_log_has_traceback():
    assert "TypeError: can't compare offset-naive and offset-aware" in get_ci_log(104, tail=30)


# -------------------------------------------------------------- tracker writes


def test_create_issue_is_idempotent(fresh_workspace):
    first = create_issue("Flaky test", "details", ["bug"], idempotency_key="k1")
    retry = create_issue("Flaky test", "details", ["bug"], idempotency_key="k1")
    assert first == "Created issue #7: Flaky test"
    assert retry.startswith(first) and "not repeated" in retry
    assert search_issues("Flaky").count("#") == 1


def test_different_keys_create_different_issues(fresh_workspace):
    create_issue("A", "x", [], idempotency_key="k1")
    assert "#8" in create_issue("A", "x", [], idempotency_key="k2")


def test_failed_write_does_not_consume_key(fresh_workspace):
    with pytest.raises(ToolError):
        add_comment(999, "hi", idempotency_key="k1")
    assert add_comment(3, "hi", idempotency_key="k1") == "Commented on issue #3."
    assert "- devcopilot" in get_issue(3)


def test_update_issue_reports_change(fresh_workspace):
    out = update_issue(3, status="closed", assignee="bob")
    assert out == "Updated issue #3: status: 'open' -> 'closed', assignee: 'unassigned' -> 'bob'"
    assert "[closed]" in search_issues("with TTL", status="all")


def test_update_issue_requires_a_change(fresh_workspace):
    with pytest.raises(ToolError, match="Nothing to update"):
        update_issue(3)


# ----------------------------------------------------------------- repo writes

FIX = dict(
    branch="fix-naive-datetime",
    path="shortlink/expiry.py",
    edits=[
        TextEdit(
            old_text="from datetime import datetime, timedelta\n",
            new_text="from datetime import datetime, timedelta, timezone\n",
        ),
        TextEdit(old_text="now or datetime.now()", new_text="now or datetime.now(timezone.utc)"),
    ],
    commit_message="fix: compare timezone-aware datetimes",
)


def test_run_tests_on_main_fails():
    out = run_tests("main")
    assert out.startswith("Tests at main: FAILED")
    assert "2 failed, 13 passed" in out


def test_patch_creates_branch_tests_pass_main_untouched(fresh_workspace):
    out = propose_patch(**FIX)
    assert out.startswith("Created branch copilot/fix-naive-datetime")
    assert "+    now = now or datetime.now(timezone.utc)" in out
    assert run_tests("copilot/fix-naive-datetime").startswith("Tests at copilot/fix-naive-datetime: PASSED")
    assert "datetime.now()" in read_file("shortlink/expiry.py")  # main untouched
    author = git(fresh_workspace / "repo", "log", "-1", "--format=%an", "copilot/fix-naive-datetime")
    assert author.strip() == "devcopilot"


def test_propose_patch_is_idempotent_by_branch(fresh_workspace):
    propose_patch(**FIX)
    retry = propose_patch(**FIX)
    assert retry.startswith("Branch copilot/fix-naive-datetime already exists, no changes made.")


def test_bad_edit_leaves_nothing_behind(fresh_workspace):
    bad = {**FIX, "edits": [FIX["edits"][0], TextEdit(old_text="no such text", new_text="x")]}
    with pytest.raises(ToolError, match="Edit 2: old_text occurs 0 times"):
        propose_patch(**bad)
    assert git(fresh_workspace / "repo", "branch", "--list", "copilot/*").strip() == ""
    assert not list((fresh_workspace / "worktrees").glob("*"))
