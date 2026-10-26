"""Fakes shared by the team and telemetry tests: a scripted model and fake tools."""

import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import Field

from devcopilot.agents.team import SPECIALISTS, Assignment, Review, SupervisorDecision
from devcopilot.runner import reject_all, run_question


class FakeTeamModel(GenericFakeChatModel):
    """Chat calls return `messages` in order; structured calls pop from `structured[schema name]`."""

    structured: dict[str, list] = Field(default_factory=dict)

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        queue = self.structured.setdefault(schema.__name__, [])

        def pop(_):
            assert queue, f"unexpected extra {schema.__name__} call"
            item = queue.pop(0)
            # A scripted string simulates a model that answered in plain text.
            if isinstance(item, str):
                return {"parsed": None, "parsing_error": None, "raw": AIMessage(item)}
            return {"parsed": item, "parsing_error": None, "raw": AIMessage("")}

        return RunnableLambda(pop)


def fake_model(chat: list, *structured) -> FakeTeamModel:
    by_type: dict[str, list] = {}
    for obj in structured:
        # ("Intent", "text") scripts a plain-text reply where an Intent was expected.
        name, obj = obj if isinstance(obj, tuple) else (type(obj).__name__, obj)
        by_type.setdefault(name, []).append(obj)
    return FakeTeamModel(messages=iter(chat), structured=by_type)


class World:
    """Fake versions of all tools the specialists need; records writes."""

    def __init__(self):
        self.created: list[tuple[str, str]] = []

    def tools(self):
        def make(name, read_only=True, fn=None):
            async def default(**kwargs):
                return f"{name}: nothing here"

            tool = StructuredTool(
                name=name,
                description=name,
                args_schema={"type": "object", "properties": {}},
                coroutine=fn or default,
            )
            tool.metadata = {"mcp_server": "tracker", "readOnlyHint": read_only}
            return tool

        async def get_issue(**kwargs):
            return "#3 Short links fail with HTTP 500 (assignee: alice)"

        async def create_issue(title: str = "", idempotency_key: str = ""):
            self.created.append((title, idempotency_key))
            return "Created issue #7"

        names = {t for spec in SPECIALISTS.values() for t in spec.tools}
        writes = {"propose_patch", "create_issue", "add_comment", "update_issue"}
        tools = {n: make(n, read_only=n not in writes) for n in names}
        tools["get_issue"] = make("get_issue", fn=get_issue)
        create = make("create_issue", read_only=False, fn=create_issue)
        create.args_schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "idempotency_key": {"type": "string"}},
        }
        tools["create_issue"] = create
        return list(tools.values())


def call(name, args, call_id="call_1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def decide(*assignments):
    return SupervisorDecision(thoughts="t", assignments=[Assignment(agent=a, task=t) for a, t in assignments])


OK = Review(problems=[], verdict="ok")


def run(model, world=None, approve=reject_all, question="q"):
    updates = []

    async def go():
        return await run_question(
            question, "team", (world or World()).tools(), approve=approve, model=model,
            on_update=lambda ns, u: updates.append((ns, u)),
        )  # fmt: skip

    return asyncio.run(go()), updates
