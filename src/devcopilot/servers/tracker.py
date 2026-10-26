"""MCP server: issue tracker."""

from mcp.types import ToolAnnotations

from devcopilot.servers import READ_ONLY, create_server, register, run
from devcopilot.tools import tracker

server = create_server("tracker", "Issue tracker of the shortlink team: search, read and update issues.")

register(server, tracker.search_issues, READ_ONLY)
register(server, tracker.get_issue, READ_ONLY)
# Creates new records; safe to retry only thanks to the idempotency key.
APPENDS = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
register(server, tracker.create_issue, APPENDS)
register(server, tracker.add_comment, APPENDS)
# Overwrites status/assignee/labels: the previous values are lost.
register(
    server,
    tracker.update_issue,
    ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False),
)

if __name__ == "__main__":
    run(server)
