"""Connects to the MCP servers and turns their tools into LangChain tools."""

import os
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from devcopilot.config import get_settings
from devcopilot.telemetry import server_env
from devcopilot.tools._common import minimal_env

SERVERS = {
    "repo": "devcopilot.servers.repo",
    "tracker": "devcopilot.servers.tracker",
    "ci": "devcopilot.servers.ci",
}


def server_connections(servers: dict[str, str] = SERVERS) -> dict[str, dict]:
    s = get_settings()
    # A stdio server is a child process. Give it only what it needs: no API keys
    # or other secrets from our environment leak into tool servers.
    env = minimal_env(
        HOME=os.environ.get("HOME", ""),
        DEVCOPILOT_WORKSPACE_DIR=str(s.workspace_dir),
        DEVCOPILOT_DOTENV="0",  # settings only from this environment, see config.get_settings
        **server_env(),  # telemetry endpoint and chaos settings, if any
    )
    return {
        name: {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", module],
            "env": {**env, "OTEL_SERVICE_NAME": f"devcopilot-mcp-{name}"},
        }
        for name, module in servers.items()
    }


async def tool_span_as_current(request, handler):
    """Interceptor: make the agent's tool-call span the current OTel span during the call.

    OpenInference builds LangChain spans from callbacks and does not make them "current"
    in the OTel context. Without this, the MCP instrumentation would propagate the
    enclosing run span, and the server's span would hang off the run, not the tool call.
    """
    if not get_settings().otel_endpoint:  # telemetry off (get_current_span() would print a warning)
        return await handler(request)
    from openinference.instrumentation.langchain import get_current_span
    from opentelemetry import trace

    span = get_current_span()
    if span is None:
        return await handler(request)
    with trace.use_span(span, end_on_exit=False):
        return await handler(request)


@asynccontextmanager
async def open_mcp_tools(servers: dict[str, str] = SERVERS) -> AsyncIterator[list[BaseTool]]:
    """Start the servers, keep one session per server open, yield all their tools.

    Without explicit sessions the adapter starts a fresh server process for every
    tool call: correct, but ~1 s of overhead per call over stdio.
    """
    client = MultiServerMCPClient(server_connections(servers))
    async with AsyncExitStack() as stack:
        tools: list[BaseTool] = []
        for name in servers:
            session = await stack.enter_async_context(client.session(name))
            for tool in await load_mcp_tools(session, server_name=name, tool_interceptors=[tool_span_as_current]):
                tool.metadata = {**(tool.metadata or {}), "mcp_server": name}
                tools.append(tool)
        yield tools
