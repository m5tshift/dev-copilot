"""MCP server: CI runs and logs."""

from devcopilot.servers import READ_ONLY, create_server, register, run
from devcopilot.tools import ci

server = create_server("ci", "CI system of the shortlink service: runs on main and their logs.")

register(server, ci.list_ci_runs, READ_ONLY)
register(server, ci.get_ci_log, READ_ONLY)

if __name__ == "__main__":
    run(server)
