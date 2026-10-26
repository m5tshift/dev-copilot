"""Runs a question through the single agent or the team, handling approvals.

Used by the CLI (interactive approvals, live trajectory) and by the benchmark
(automatic approvals, metrics only).
"""

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from openinference.instrumentation import using_session
from opentelemetry.trace import Status, StatusCode

from devcopilot.agents.single import build_agent, compact, count_tool_calls, initial_state
from devcopilot.agents.team import UNVERIFIED_MARK, build_team, initial_team_state
from devcopilot.config import get_settings
from devcopilot.decide import start_counting
from devcopilot.grounding import extract_entities, ungrounded
from devcopilot.llm import is_transient, recovery_counts
from devcopilot.memory import Memory, digest, repo_head
from devcopilot.telemetry import log, metrics, span
from devcopilot.trajectory import Trajectory
from devcopilot.usage import UsageTracker

Mode = Literal["single", "team"]
# Gets an interrupt payload {"calls": [...], "warnings": {...}}, returns {call_id: Decision}.
Approver = Callable[[dict], dict]
# Gets (namespace, update) for every node update, for live display.
UpdateHandler = Callable[[tuple[str, ...], dict], None]


@dataclass
class RunResult:
    answer: str
    seconds: float  # excluding time spent waiting for approvals
    llm_calls: int
    tokens_in: int
    tokens_out: int
    tool_calls: int
    write_requests: int  # calls that asked for approval
    escalations: dict[str, int] = field(default_factory=dict)
    recoveries: dict[str, int] = field(default_factory=dict)
    budget_exhausted: int = 0
    llm_calls_by_agent: dict[str, int] = field(default_factory=dict)
    tokens_in_by_agent: dict[str, int] = field(default_factory=dict)
    blocked: list[str] = field(default_factory=list)  # team: assignments refused by the gate
    ungrounded: list[str] = field(default_factory=list)  # specifics in the answer seen in no tool output
    reports: int = 0
    # Phase 4: what the agents did, for trajectory evals (see trajectory.py).
    trajectory: list[dict] = field(default_factory=list)
    routing: list[dict] = field(default_factory=list)
    allow_changes: bool | None = None
    # Validator verdicts: {"stage": "report" | "answer", "verdict": "ok" | "revise"}.
    validations: list[dict] = field(default_factory=list)
    # Typed decisions (decide.py): not LLM calls, so counted separately.
    decider_calls: int = 0
    decider_seconds: float = 0.0
    thread_id: str = ""
    # Inference (from Ollama's timings): mean time to first token, generation speed.
    ttft_s: float = 0.0
    tokens_per_s: float = 0.0
    # Size of the memory/history block every reader got this turn (after compaction).
    context_chars: int = 0
    # Team coordination (see coordination_of); None for the single agent.
    rounds: int | None = None
    revisions: int | None = None
    reported_entities: int | None = None
    lost_entities: list[str] = field(default_factory=list)
    cross_agent_duplicates: int | None = None


def compact_history(before: dict) -> tuple[str, list[str]]:
    """Team working memory for a follow-up: the earlier turns as a digest plus the entities
    their tools showed. Every reader (supervisor, synthesizer, reviewer) would otherwise get
    all reports of all turns on every call, growing without bound."""
    s = get_settings()
    answer = before.get("answer", "").split(UNVERIFIED_MARK)[0]
    turn = f"Q: {digest(before.get('question', ''), 200)}\nA: {digest(answer, 600)}"
    turns = [t for t in (before.get("history") or "").split(TURN_SEP) if t] + [turn]
    while len(turns) > 1 and len(TURN_SEP.join(turns)) > s.history_chars:
        turns.pop(0)  # the oldest turns go first: follow-ups refer to the recent ones
    facts = set(before.get("known_facts", [])) | {f for r in before.get("reports", []) for f in r["facts"]}
    return TURN_SEP.join(turns), sorted(facts)


TURN_SEP = "\n---\n"


def next_turn(mode: Mode, question: str, recalled: str, follow_up: bool, before: dict | None = None) -> dict:
    """Input of one turn. Memory goes to the agents' context, never into the question
    itself: the gate must decide from the user's words alone. `before`: the thread's state
    after the previous turn (team follow-ups compact it, see compact_history)."""
    if mode == "single":
        # A message of its own: approvals check write arguments against the user's words and
        # tool output, and a hash seen only in memory must still raise a warning (single.py).
        state = initial_state(question)
        if recalled:
            state["messages"].append(HumanMessage(recalled))
        return {"messages": state["messages"], "steps": 0} if follow_up else state
    if not follow_up:
        return {**initial_team_state(question), "memory": recalled}
    history, known_facts = compact_history(before or {})
    # Per-turn fields start over; None resets the reducers (see team.add_or_reset).
    return {
        "question": question, "memory": recalled, "rounds": 0, "revisions": 0, "feedback": [], "answer": "",
        "blocked": None, "reports": None, "history": history, "known_facts": known_facts,
    }  # fmt: skip


def reject_all(payload: dict) -> dict:
    return {c["id"]: {"action": "reject", "reason": "writes are disabled in this run"} for c in payload["calls"]}


async def run_question(
    question: str,
    mode: Mode,
    tools: list[BaseTool],
    approve: Approver = reject_all,
    on_update: UpdateHandler | None = None,
    model: BaseChatModel | None = None,
    jev_for: str | None = None,
    models: dict[str, BaseChatModel] | None = None,
    memory: Memory | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
    validate_reports: bool | None = None,
) -> RunResult:
    """jev_for: team components decided by Jev (decide.py); None = settings.jev_for.
    models: team roles on their own models (llm.py); None = settings.routes.
    memory: episodes and facts (memory.py); None = no long-term memory.
    checkpointer + thread_id: continue an earlier conversation (short-term memory);
    by default every run is a new thread in an in-memory checkpointer."""
    thread_id = thread_id or uuid.uuid4().hex
    checkpointer = checkpointer or InMemorySaver()
    # One root span for the whole run. Without it every resume after an interrupt
    # would start a separate trace. session.id groups runs in Phoenix's Sessions view.
    with (
        using_session(thread_id),
        span("devcopilot.run", **{"input.value": question, "session.id": thread_id, "devcopilot.mode": mode}) as root,
    ):
        started = time.perf_counter()
        log.info("run started", extra={"mode": mode, "question": question, "thread_id": thread_id})
        try:
            counts = {"node_retries": 0, "resumes": 0}
            token = recovery_counts.set(counts)
            result = await _run(
                question, mode, tools, approve, on_update, model, thread_id, jev_for, models, memory, checkpointer,
                validate_reports,
            )  # fmt: skip
        except (Exception, asyncio.CancelledError) as e:
            metrics.runs.add(1, {"mode": mode, "outcome": "error"})
            log.exception("run failed", extra={"mode": mode, "thread_id": thread_id})
            root.record_exception(e)
            root.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            recovery_counts.reset(token)
        result.recoveries.update(counts)
        metrics.runs.add(1, {"mode": mode, "outcome": "ok"})
        metrics.run_duration.record(time.perf_counter() - started, {"mode": mode})
        if result.ungrounded:
            metrics.ungrounded.add(len(result.ungrounded), {"where": "answer"})
        record_flow_metrics(result, mode)
        summary = {
            k: v for k, v in asdict(result).items() if k not in ("answer", "trajectory", "routing", "validations")
        }
        root.set_attribute("output.value", result.answer)
        for key, value in summary.items():
            if isinstance(value, int | float):
                root.set_attribute(f"devcopilot.{key}", value)
        log.info("run finished", extra={"mode": mode, "thread_id": thread_id, **summary})
        return result


async def _run(
    question, mode, tools, approve, on_update, model, thread_id, jev_for, models, memory, checkpointer, validate_reports
) -> RunResult:
    head = repo_head()
    recalled = await memory.context(question, head) if memory else ""
    if mode == "single":
        graph = await asyncio.to_thread(build_agent, tools, model, checkpointer=checkpointer)
    else:
        graph = await asyncio.to_thread(
            build_team,
            tools,
            model,
            checkpointer=checkpointer,
            jev_for=jev_for,
            models=models,
            validate_reports=validate_reports,
        )
    usage = UsageTracker()
    decisions = start_counting()
    # thread_id names this run in the checkpointer; resuming must use the same one.
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [usage],
        "metadata": {"devcopilot.mode": mode},
        "recursion_limit": 100,
    }
    # A thread with history: a follow-up question. The earlier messages (single) or
    # reports (team) stay in the state: that is the short-term memory of the dialogue.
    before = (await graph.aget_state(config)).values
    follow_up = bool(before)
    # Counts below are of this turn: the single agent's thread keeps earlier messages.
    earlier_messages = len(before.get("messages", [])) if mode == "single" else 0
    payload = next_turn(mode, question, recalled, follow_up, before)
    # What every reader of memory gets on each call this turn: the compaction's result.
    context_chars = len(recalled) + len(payload.get("history") or "")
    if mode == "single" and earlier_messages:
        # Compact with the new question present so earlier tool output is actually cut.
        history = compact([*before["messages"], *payload["messages"]])
        context_chars += sum(len(m.text) for m in history[: -len(payload["messages"])])
    started, waited, write_requests = time.perf_counter(), 0.0, 0
    trajectory = Trajectory()

    resumes = 0
    while True:
        interrupts: dict[str, dict] = {}
        # subgraphs=True also streams updates from inside specialists, with a namespace
        # like ("code_analyst:<task id>",) telling which subgraph produced them.
        try:
            async for namespace, update in graph.astream(payload, config, stream_mode="updates", subgraphs=True):
                if "__interrupt__" in update:
                    # Parallel specialists may pause at the same time: one interrupt each.
                    for intr in update["__interrupt__"]:
                        interrupts[intr.id] = intr.value
                    continue
                trajectory.observe(namespace, update)
                if on_update:
                    on_update(namespace, update)
        except Exception as error:
            if not is_transient(error) or resumes >= get_settings().run_resume_attempts:
                raise
            resumes += 1
            recovery_counts.get()["resumes"] = resumes
            metrics.recoveries.add(1, {"kind": "resume"})
            log.warning("resuming failed graph from checkpoint", extra={"thread_id": thread_id, "resumes": resumes})
            payload = None
            continue
        if not interrupts:
            break
        write_requests += sum(len(v["calls"]) for v in interrupts.values())
        wait_started = time.perf_counter()
        # The wait for a human is a span of its own: visible as a gap in the trace.
        with span("human approval", calls=json.dumps([c["name"] for v in interrupts.values() for c in v["calls"]])):
            # Resume every paused interrupt at once, addressed by its id.
            payload = Command(resume={iid: approve(value) for iid, value in interrupts.items()})
        waited += time.perf_counter() - wait_started

    state = (await graph.aget_state(config)).values
    if memory:
        # Without the reviewer's warning: its unverified entities are not memories.
        final = state["messages"][-1].text if mode == "single" else state["answer"].split(UNVERIFIED_MARK)[0]
        # One episode per turn: keyed by the thread alone, a follow-up would overwrite the last.
        await memory.record(f"{thread_id}:{uuid.uuid4().hex[:8]}", question, final, head, mode, trajectory.outputs)
    if mode == "single":
        messages = state["messages"]
        answer = messages[-1].text
        tool_calls = count_tool_calls(messages[earlier_messages:])
        sources = [question, *(m.text for m in messages if isinstance(m, ToolMessage))]
        blocked, reports, checked = [], 0, answer
        coordination = {}
    else:
        answer = state["answer"]
        # The reviewer's own warning lists the unverified entities; do not count them twice.
        checked = answer.split(UNVERIFIED_MARK)[0]
        tool_calls = sum(r["tool_calls"] for r in state["reports"])
        sources = [question, *state.get("known_facts", []), *(f for r in state["reports"] for f in r["facts"])]
        blocked, reports = state["blocked"], len(state["reports"])
        coordination = coordination_of(state["reports"], checked, trajectory)
        coordination["revisions"] = state["revisions"]
    return RunResult(
        answer=answer,
        recoveries={"resumes": resumes},
        seconds=time.perf_counter() - started - waited,
        llm_calls=usage.llm_calls,
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        tool_calls=tool_calls,
        write_requests=write_requests,
        escalations={"approval": write_requests, "unverified": int(mode == "team" and UNVERIFIED_MARK in answer)},
        budget_exhausted=trajectory.budget_exhausted,
        llm_calls_by_agent=usage.llm_calls_by_agent,
        tokens_in_by_agent=usage.tokens_in_by_agent,
        blocked=blocked,
        ungrounded=ungrounded(checked, sources),
        reports=reports,
        trajectory=trajectory.tool_dicts(),
        routing=trajectory.routing,
        allow_changes=trajectory.allow_changes,
        validations=trajectory.validations,
        decider_calls=len(decisions),
        decider_seconds=sum(seconds for _, seconds, _ in decisions),
        thread_id=thread_id,
        ttft_s=sum(usage.ttft_s) / len(usage.ttft_s) if usage.ttft_s else 0.0,
        tokens_per_s=usage.tokens_per_s,
        context_chars=context_chars,
        **coordination,
    )


def record_flow_metrics(result: RunResult, mode: Mode) -> None:
    """Agent metrics of one run: the memory every reader got and, for the team, how it
    worked as one flow (rounds, revisions, handoff, duplicated work)."""
    for kind, count in result.escalations.items():
        metrics.escalations.add(count, {"kind": kind})
    metrics.memory_context.record(result.context_chars, {"mode": mode})
    if result.rounds is None:
        return
    metrics.team_rounds.record(result.rounds)
    metrics.team_revisions.add(result.revisions or 0)
    lost = len(result.lost_entities)
    metrics.handoff_entities.add(lost, {"outcome": "lost"})
    metrics.handoff_entities.add((result.reported_entities or 0) - lost, {"outcome": "delivered"})
    metrics.duplicate_calls.add(result.cross_agent_duplicates or 0)


def coordination_of(reports: list[dict], answer: str, trajectory: Trajectory) -> dict:
    """How well the team works as one flow, from what it already records.

    - handoff: entities the specialists reported (hashes, #N, files, runs) that reached
      the final answer. Lost ones are found work the user never sees; not every finding
      belongs in the answer, so this is a signal to read with the answer, not a score.
    - duplicated calls: the same tool with the same arguments by different specialists,
      work done twice because the supervisor's tasks overlapped.
    - rounds and revisions: how fast the team converges.
    """
    reported = set().union(*(extract_entities(r["summary"].split("\n\nValidator:")[0]) for r in reports))
    lost = sorted(ungrounded(" ".join(sorted(reported)), [answer])) if reported else []
    by_call: dict[tuple[str, str], set[str]] = {}
    for step in trajectory.tools:
        by_call.setdefault((step.tool, json.dumps(step.args, sort_keys=True)), set()).add(step.agent)
    return {
        "rounds": len(trajectory.routing),
        "revisions": trajectory.revisions,
        "reported_entities": len(reported),
        "lost_entities": lost,
        "cross_agent_duplicates": sum(len(agents) - 1 for agents in by_call.values()),
    }
