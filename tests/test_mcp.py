"""Integration: real MCP servers over stdio, real client, no LLM."""

import asyncio

from devcopilot.mcp_client import open_mcp_tools
from devcopilot.policy import inject_args, model_facing_schema, requires_approval


def with_tools(check):
    async def run():
        async with open_mcp_tools() as tools:
            return await check({t.name: t for t in tools})

    return asyncio.run(run())


def call(tool, args, call_id="c1"):
    return tool.ainvoke({"type": "tool_call", "name": tool.name, "args": args, "id": call_id})


def test_servers_publish_tools_with_policy():
    async def check(tools):
        needs_approval = {name for name, t in tools.items() if requires_approval(t)}
        assert needs_approval == {"propose_patch", "create_issue", "add_comment", "update_issue"}
        assert len(tools) == 14

    with_tools(check)


def test_read_tool_and_error_over_mcp():
    async def check(tools):
        ok = await call(tools["get_issue"], {"issue_id": 3})
        assert ok.status == "success" and "HTTP 500" in ok.text
        err = await call(tools["get_issue"], {"issue_id": 999})
        assert err.status == "error" and "does not exist" in err.text

    with_tools(check)


def test_idempotent_write_over_mcp(fresh_workspace):
    async def check(tools):
        create = tools["create_issue"]
        assert "idempotency_key" not in model_facing_schema(create)["function"]["parameters"]["properties"]
        args = inject_args(create, {"title": "T", "body": "B", "labels": []}, idempotency_key="k")
        first = await call(create, args)
        retry = await call(create, args, "c2")
        assert first.text == "Created issue #7: T"
        assert "not repeated" in retry.text

    with_tools(check)


def test_missing_idempotency_key_is_rejected_by_server(fresh_workspace):
    async def check(tools):
        out = await call(tools["create_issue"], {"title": "T", "body": "B", "labels": []})
        assert out.status == "error" and "idempotency_key" in out.text

    with_tools(check)
