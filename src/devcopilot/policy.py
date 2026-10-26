"""Client-side tool policy: what needs human approval, what the client injects.

The policy lives in the client, not in the servers: a server only *declares*
what its tools do (MCP annotations); the client decides how much to believe it.
"""

import copy
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

# Servers we wrote ourselves; their annotations are trusted. A third-party server
# could mark "delete_everything" as readOnlyHint=True, so for unknown servers
# every call requires approval regardless of annotations.
# "local": in-process tools of this package (load_skill, persona.py).
TRUSTED_SERVERS = {"repo", "tracker", "ci", "local"}

# Arguments filled by the client, never by the model. The model does not even
# see them in the tool schema, so it cannot forge or forget them.
INJECTED_ARGS = {"idempotency_key"}


def requires_approval(tool: BaseTool) -> bool:
    meta = tool.metadata or {}
    if meta.get("mcp_server") not in TRUSTED_SERVERS:
        return True
    # Missing annotation means "unknown", and unknown means "ask".
    return meta.get("readOnlyHint") is not True


def model_facing_schema(tool: BaseTool) -> dict:
    """Tool schema as the model sees it: without client-injected arguments."""
    # deepcopy: for MCP tools the result shares dicts with the tool's own schema.
    schema = copy.deepcopy(convert_to_openai_tool(tool))
    params = schema["function"]["parameters"]
    for arg in INJECTED_ARGS:
        params.get("properties", {}).pop(arg, None)
        if arg in params.get("required", []):
            params["required"].remove(arg)
    return schema


def inject_args(tool: BaseTool, args: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
    # MCP tools carry a JSON schema dict, native LangChain tools a pydantic model.
    schema = tool.tool_call_schema
    schema = schema if isinstance(schema, dict) else schema.model_json_schema()
    if "idempotency_key" in schema.get("properties", {}):
        return {**args, "idempotency_key": idempotency_key}
    return args
