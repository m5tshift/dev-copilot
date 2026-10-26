"""Agent graph mechanics with a scripted fake model and fake tools (no LLM, no MCP)."""

import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from devcopilot.agents.single import build_agent, initial_state
from devcopilot.tools._common import ToolError


class ScriptedModel(GenericFakeChatModel):
    """Returns pre-written messages in order; ignores tool binding."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages):
    return ScriptedModel(messages=iter(messages))


def tool_call(name, args, call_id="call_1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def fake_tool(fn, read_only=True):
    tool = StructuredTool.from_function(fn)
    tool.metadata = {"mcp_server": "tracker", "readOnlyHint": read_only}
    return tool


class Tracker:
    """Fake tracker that records write calls."""

    def __init__(self):
        self.created = []

    def tools(self):
        def get_issue(issue_id: int) -> str:
            """Get an issue."""
            if issue_id != 3:
                raise ToolError(f"Issue #{issue_id} does not exist.")
            return "#3 Links fail with HTTP 500"

        def create_issue(title: str, idempotency_key: str) -> str:
            """Create an issue."""
            self.created.append((title, idempotency_key))
            return f"Created issue #7: {title}"

        return [fake_tool(get_issue), fake_tool(create_issue, read_only=False)]


CONFIG = {"configurable": {"thread_id": "t1"}}


def run(agent, payload):
    return asyncio.run(agent.ainvoke(payload, CONFIG))


def tool_messages(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


# ------------------------------------------------------------------ the loop


def test_tool_call_then_answer():
    model = scripted(tool_call("get_issue", {"issue_id": 3}), AIMessage("Issue #3 is about HTTP 500."))
    result = run(build_agent(Tracker().tools(), model), initial_state("what is issue 3?"))

    kinds = [type(m).__name__ for m in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    msg = tool_messages(result)[0]
    assert msg.status == "success" and "HTTP 500" in msg.content
    assert result["steps"] == 2


def test_tool_output_is_wrapped_as_untrusted():
    model = scripted(tool_call("get_issue", {"issue_id": 3}), AIMessage("ok"))
    content = tool_messages(run(build_agent(Tracker().tools(), model), initial_state("x")))[0].content
    assert content.startswith('<tool_output tool="get_issue">') and content.endswith("</tool_output>")


def test_tool_error_is_returned_to_model_not_raised():
    model = scripted(tool_call("get_issue", {"issue_id": 999}), AIMessage("No such issue."))
    msg = tool_messages(run(build_agent(Tracker().tools(), model), initial_state("x")))[0]
    assert msg.status == "error" and "does not exist" in msg.content


def test_unknown_tool_is_reported():
    model = scripted(tool_call("delete_repo", {}), AIMessage("ok"))
    msg = tool_messages(run(build_agent(Tracker().tools(), model), initial_state("x")))[0]
    assert msg.status == "error" and "unknown tool" in msg.content


def test_step_budget_forces_final_answer():
    # The model would call tools forever; with max_steps=3 the 3rd call is made without tools.
    looping = [tool_call("get_issue", {"issue_id": 3}, f"call_{i}") for i in range(2)]
    model = scripted(*looping, AIMessage("Partial answer."))
    result = run(build_agent(Tracker().tools(), model, max_steps=3), initial_state("loop"))
    assert result["steps"] == 3
    assert result["messages"][-1].content == "Partial answer."


# ------------------------------------------------------------ human in the loop


def start_write(tracker, title="Bug"):
    model = scripted(tool_call("create_issue", {"title": title}), AIMessage("Done."))
    # interrupt() needs a checkpointer to save the paused graph.
    agent = build_agent(tracker.tools(), model, checkpointer=InMemorySaver())
    run(agent, initial_state("file a bug"))
    return agent


def test_write_call_pauses_for_approval():
    tracker = Tracker()
    agent = start_write(tracker)

    state = asyncio.run(agent.aget_state(CONFIG))
    assert state.next == ("review",)
    assert state.interrupts[0].value["calls"][0]["name"] == "create_issue"
    assert tracker.created == []  # nothing written while waiting


def test_approve_executes_once_with_injected_key():
    tracker = Tracker()
    agent = start_write(tracker)
    result = run(agent, Command(resume={"call_1": {"action": "approve"}}))

    assert tracker.created == [("Bug", "t1:call_1")]
    assert result["messages"][-1].content == "Done."


def test_reject_skips_call_and_tells_model():
    tracker = Tracker()
    agent = start_write(tracker)
    result = run(agent, Command(resume={"call_1": {"action": "reject", "reason": "duplicate of #3"}}))

    assert tracker.created == []
    msg = tool_messages(result)[0]
    assert msg.status == "error" and "duplicate of #3" in msg.content


def test_edit_replaces_arguments():
    tracker = Tracker()
    agent = start_write(tracker)
    run(agent, Command(resume={"call_1": {"action": "edit", "args": {"title": "Better title"}}}))
    assert tracker.created == [("Better title", "t1:call_1")]


def test_missing_decision_means_reject():
    tracker = Tracker()
    agent = start_write(tracker)
    run(agent, Command(resume={}))
    assert tracker.created == []


def test_model_does_not_see_injected_args():
    from devcopilot.policy import model_facing_schema

    create = Tracker().tools()[1]
    params = model_facing_schema(create)["function"]["parameters"]
    assert "idempotency_key" not in params["properties"]
    assert "idempotency_key" in create.tool_call_schema.model_json_schema()["properties"]


def test_approval_warns_about_ungrounded_specifics():
    agent = start_write(Tracker(), title="See https://github.com/acme/x and #42")
    payload = asyncio.run(agent.aget_state(CONFIG)).interrupts[0].value
    assert payload["warnings"] == {"call_1": ["#42", "https://github.com/acme/x"]}


def test_single_run_counts_approval_escalation():
    from devcopilot.runner import run_question

    tracker = Tracker()
    result = asyncio.run(
        run_question(
            "file a bug",
            "single",
            tracker.tools(),
            model=scripted(tool_call("create_issue", {"title": "Bug"}), AIMessage("Done.")),
            approve=lambda payload: {c["id"]: {"action": "approve"} for c in payload["calls"]},
        )
    )
    assert result.write_requests == 1
    assert result.escalations == {"approval": 1, "unverified": 0}
