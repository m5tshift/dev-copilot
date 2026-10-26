"""A single tool-using agent with human approval for side effects.

    START -> agent --(read-only calls)--> tools -> agent -> ... -> END
                   \\-(needs approval)--> review --> tools

Phase 0 built the ReAct loop (agent <-> tools). Phase 1 adds:
- tools come from MCP servers (see devcopilot/mcp_client.py), the loop does not care;
- review: pauses the graph with interrupt() until a human approves, rejects or
  edits every call that changes something (policy in devcopilot/policy.py);
- idempotency keys injected by the client, derived from thread id + tool call id;
- tool output is marked as untrusted data (defence against prompt injection).

Phase 2 reuses this graph as a specialist inside a team (devcopilot/agents/team.py),
with its own prompt and tool subset, and adds grounding warnings to approvals.
Phase 3 adds metrics and logs (devcopilot/telemetry.py); spans come from auto-instrumentation.
"""

import json
import logging
import operator
import time
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from devcopilot.agents.prompts import SINGLE_AGENT
from devcopilot.config import get_settings
from devcopilot.grounding import ungrounded
from devcopilot.llm import get_chat_model, llm_chaos, node_retry_policy, retry_observed
from devcopilot.persona import LOAD_SKILL, LOCAL_SERVER, skill_tool, skills_index
from devcopilot.policy import inject_args, model_facing_schema, requires_approval
from devcopilot.telemetry import log, metrics

BUDGET_EXHAUSTED = (
    "Tool budget exhausted. Do not request more tools. Answer now using only the evidence "
    "gathered so far, and state clearly what remains unverified."
)


class Decision(TypedDict):
    action: Literal["approve", "reject", "edit"]
    args: NotRequired[dict[str, Any]]  # for "edit": replaces the model's arguments
    reason: NotRequired[str]  # for "reject": shown to the model


class AgentState(TypedDict):
    # add_messages is a reducer: nodes return *new* messages and LangGraph appends them.
    messages: Annotated[list[AnyMessage], add_messages]
    # Number of LLM calls made so far; guards against infinite tool loops.
    steps: int
    budget_exhausted: int
    # Human decisions on tool calls, by tool_call_id. operator.or_ merges dicts.
    decisions: Annotated[dict[str, Decision], operator.or_]


def wrap_untrusted(tool_name: str, text: str) -> str:
    # Escape closing tags so data cannot "close" the wrapper and pose as instructions.
    text = text.replace("</tool_output", "&lt;/tool_output")
    return f'<tool_output tool="{tool_name}">\n{text}\n</tool_output>'


def build_agent(
    tools: list[BaseTool],
    model: BaseChatModel | None = None,
    max_steps: int | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt: str = SINGLE_AGENT,
    name: str = "single",
):
    """Build the ReAct graph.

    interrupt() needs a checkpointer. Pass one for a top-level agent; leave None when the
    graph runs inside another graph's node: it then uses the parent's checkpointer.
    """
    model = model or get_chat_model()
    max_steps = max_steps or get_settings().max_steps
    # Skills of this agent (persona.py): an index in the prompt, the playbooks on demand.
    if (load_skill := skill_tool(name)) is not None:
        tools = [*tools, load_skill]
        system_prompt += skills_index(name)
    # The model sees schemas without client-injected args (idempotency_key).
    model_with_tools = model.bind_tools([model_facing_schema(t) for t in tools])
    tools_by_name = {t.name: t for t in tools}

    def needs_approval(call: dict) -> bool:
        tool = tools_by_name.get(call["name"])
        return tool is not None and requires_approval(tool)

    async def call_model(state: AgentState) -> dict:
        llm_chaos()
        messages = [SystemMessage(system_prompt), *compact(state["messages"])]
        steps = state["steps"] + 1
        if steps >= max_steps:
            # Last allowed step: call the model without tools so it must answer.
            # Usually a sign of a loop: worth a metric and an alert.
            metrics.budget_exhausted.add(1)
            log.warning("step budget exhausted", extra={"agent": name, "max_steps": max_steps})
            response = await model.ainvoke([*messages, HumanMessage(BUDGET_EXHAUSTED)])
        else:
            response = await model_with_tools.ainvoke(messages)
        if not response.tool_calls:
            metrics.agent_steps.record(steps, {"devcopilot.agent": name})
        return {"messages": [response], "steps": steps, "budget_exhausted": int(steps >= max_steps)}

    async def review(state: AgentState) -> dict:
        last = state["messages"][-1]
        pending = [c for c in last.tool_calls if needs_approval(c) and c["id"] not in state["decisions"]]
        # interrupt() saves the state via the checkpointer and stops the graph.
        # The caller resumes with Command(resume={tool_call_id: Decision}), and
        # interrupt() then returns that value. The node re-runs from the top on resume,
        # which is why side effects never happen in this node (metrics and logs included:
        # they would be counted twice; run_tools records them).
        answer = interrupt({"calls": pending, "warnings": _grounding_warnings(state, pending)})
        # Anything the human did not explicitly decide is rejected.
        return {
            "decisions": {c["id"]: answer.get(c["id"], {"action": "reject", "reason": "no decision"}) for c in pending}
        }

    def _grounding_warnings(state: AgentState, calls: list[dict]) -> dict[str, list[str]]:
        # Sources: what the agent was told (task) and what tools returned. Its own
        # earlier messages do not count: the model may have invented things there too.
        # Recalled memory (runner.next_turn) is a hint, not a source.
        sources = [m.text for m in state["messages"] if isinstance(m, HumanMessage | ToolMessage) and not _is_memory(m)]
        warnings = {}
        for call in calls:
            if found := ungrounded(json.dumps(call["args"], ensure_ascii=False), sources):
                warnings[call["id"]] = found
        return warnings

    async def run_tools(state: AgentState, config: RunnableConfig) -> dict:
        thread_id = config["configurable"].get("thread_id", "no-thread")
        last = state["messages"][-1]
        warnings = _grounding_warnings(state, [c for c in last.tool_calls if needs_approval(c)])
        results = []
        for call in last.tool_calls:
            if found := warnings.get(call["id"]):
                metrics.ungrounded.add(len(found), {"where": "approval"})
                log.warning(
                    "write request mentions unseen entities",
                    extra={"agent": name, "tool": call["name"], "entities": found},
                )
            results.append(await _execute(call, state["decisions"].get(call["id"]), thread_id))
        return {"messages": results}

    async def _execute(call: dict, decision: Decision | None, thread_id: str) -> ToolMessage:
        tool_name, call_id = call["name"], call["id"]

        def reply(text: str, status: str = "success", outcome: str | None = None, trusted: bool = False) -> ToolMessage:
            # artifact is not sent to the model; it tells evals "rejected" from "failed".
            return ToolMessage(
                text if trusted else wrap_untrusted(tool_name, text),
                tool_call_id=call_id,
                name=tool_name,
                status=status,
                artifact={"outcome": outcome or status},
            )

        def record(status: str, started: float | None = None) -> None:
            attrs = {"tool": tool_name, "status": status}
            metrics.tool_calls.add(1, attrs)
            if started is not None:
                metrics.tool_duration.record(time.perf_counter() - started, {"tool": tool_name})
            level = logging.INFO if status == "success" else logging.WARNING
            log.log(level, "tool call", extra={"agent": name, **attrs})

        tool = tools_by_name.get(tool_name)
        if tool is None:
            record("unknown_tool")
            available = ", ".join(tools_by_name)
            return reply(f"Error: unknown tool {tool_name!r}. Available: {available}.", "error", "unknown_tool")
        args, note = call["args"], ""
        if requires_approval(tool):
            action = decision["action"] if decision else "reject"
            metrics.approvals.add(1, {"tool": tool_name, "decision": action})
            if action == "reject":
                reason = (decision or {}).get("reason") or "not given"
                record("rejected")
                return reply(f"The user rejected this call (reason: {reason}). Do not retry it.", "error", "rejected")
            if action == "edit":
                args = decision["args"]
                note = f"[The user edited the arguments before execution: {args}]\n"
        # Same tool call -> same key, so re-running this node (crash, retry,
        # resume) never duplicates a write.
        args = inject_args(tool, args, idempotency_key=f"{thread_id}:{call_id}")
        started = time.perf_counter()
        try:
            result = await tool.ainvoke({"type": "tool_call", "name": tool_name, "args": args, "id": call_id})
        except Exception as e:  # transport failures, bad arguments...
            record("error", started)
            return reply(f"Error: {e}", "error")
        record(result.status, started)
        # A skill playbook is our own instructions, not data: not marked untrusted.
        trusted = (tool.metadata or {}).get("mcp_server") == LOCAL_SERVER
        return reply(note + result.text, result.status, trusted=trusted)

    def route(state: AgentState) -> Literal["review", "tools", "__end__"]:
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return END
        if any(needs_approval(c) and c["id"] not in state["decisions"] for c in last.tool_calls):
            return "review"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", retry_observed(call_model), retry_policy=node_retry_policy())
    graph.add_node("review", review)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("review", "tools")
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


COMPACT_CHARS = 400  # tool output of earlier turns is cut to this


def compact(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Working memory compaction: tool output from earlier turns of a dialogue is cut.

    The current turn (from the last human message on) is kept whole: the agent needs
    its evidence. Earlier turns keep their questions, answers and the head of each
    tool output, which is enough to follow up on and much smaller. The state itself
    is not changed: only what the model sees. Memory blocks of earlier turns are dropped:
    the current turn has its own, fresher recall, and old copies only repeat it.
    """
    questions = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage) and not _is_memory(m)]
    last_human = questions[-1] if questions else 0
    out = []
    for i, m in enumerate(messages):
        if i < last_human and isinstance(m, HumanMessage) and _is_memory(m):
            continue
        if i < last_human and isinstance(m, ToolMessage) and len(m.text) > COMPACT_CHARS:
            cut = m.text[:COMPACT_CHARS] + "\n[... cut: output of an earlier turn]"
            if m.text.startswith("<tool_output"):
                cut += "\n</tool_output>"  # keep the untrusted-data wrapper closed
            m = m.model_copy(update={"content": cut})
        out.append(m)
    return out


def _is_memory(m: AnyMessage) -> bool:
    return m.text.startswith("<memory>")


def count_tool_calls(messages: list[AnyMessage]) -> int:
    """Calls of environment tools. Loading a skill reads no data and changes nothing,
    so it does not count against the efficiency checks of evals."""
    return sum(1 for m in messages if isinstance(m, AIMessage) for c in m.tool_calls if c["name"] != LOAD_SKILL)


def initial_state(question: str) -> AgentState:
    return {"messages": [HumanMessage(question)], "steps": 0, "decisions": {}}
