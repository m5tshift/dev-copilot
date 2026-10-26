"""MCP servers that publish the tools from `devcopilot/tools/`.

Each server is a separate process speaking MCP over stdio:
    python -m devcopilot.servers.repo

Tool annotations are standard MCP hints describing what a tool does. Clients
use them for policy (our agent asks a human before any non-read-only call), but
they are *claims made by the server*: a client should trust them only for
servers it trusts. See devcopilot/policy.py.

Phase 3: each server exports its own spans. The MCP instrumentation carries the
trace context in the request's `_meta`, so a server span becomes a child of the
agent's tool-call span: one trace across processes.
"""

import functools
import os
import random
from collections.abc import Callable

from devcopilot import telemetry

# Instrument FIRST, import mcp SECOND. Instrumentation replaces functions such as
# mcp.server.stdio.stdio_server in their modules, but FastMCP does
# `from mcp.server.stdio import stdio_server` and keeps whatever it found at import
# time. Importing anything from `mcp` (even mcp.types) imports the whole SDK, so
# setup() must run before the first mcp import, here at the top of the package.
# The client names each server process via the standard OTEL_SERVICE_NAME variable.
telemetry.setup(os.environ.get("OTEL_SERVICE_NAME", "devcopilot-mcp"))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from devcopilot.tools._common import ToolError  # noqa: E402

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def create_server(name: str, instructions: str) -> FastMCP:
    # WARNING: stdio servers log to stderr, which the client shows in the terminal.
    return FastMCP(name, instructions=instructions, log_level="WARNING")


def run(server: FastMCP) -> None:
    try:
        server.run()
    finally:
        telemetry.shutdown()  # flush spans before the process exits


def register(server: FastMCP, fn: Callable, annotations: ToolAnnotations) -> None:
    @functools.wraps(fn)  # keeps the signature and docstring FastMCP builds the schema from
    def traced(*args, **kwargs):
        with telemetry.span(f"{server.name}.{fn.__name__}", kind="TOOL", **{"tool.name": fn.__name__}) as s:
            # Chaos testing: fail on purpose to exercise error handling, metrics and alerts.
            if random.random() < telemetry.chaos_failure_rate():
                s.set_attribute("devcopilot.chaos", True)
                raise ToolError("Temporary failure (chaos test). Retrying may help.")
            return fn(*args, **kwargs)

    # structured_output=False: our tools return plain text meant for the model.
    server.tool(annotations=annotations, structured_output=False)(traced)
