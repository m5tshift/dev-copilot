"""MCP server: git repository of the shortlink service."""

from mcp.types import ToolAnnotations

from devcopilot.servers import READ_ONLY, create_server, register, run
from devcopilot.tools import repo

server = create_server(
    "repo", "Git repository of the shortlink service: browse code and history, run tests, propose fixes."
)

for fn in (repo.list_files, repo.read_file, repo.search_code, repo.git_log, repo.git_show):
    register(server, fn, READ_ONLY)

# Deliberate limitation: temporary checkout + minimal_env do not prevent sibling writes.
# The whole system runs in sbx (deny-all network); intended world edits use approved
# propose_patch. Approval-gating tests would exclude code_analyst from read-only questions
# (team.write_access/available_specialists) and require approval on every test run.
# Keep readOnlyHint=True until tests have their own read-only container.
register(server, repo.run_tests, ToolAnnotations(readOnlyHint=True, openWorldHint=False))
# Writes a branch. Not destructive (main is untouched), idempotent by branch name.
register(
    server,
    repo.propose_patch,
    ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)

if __name__ == "__main__":
    run(server)
