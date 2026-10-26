"""Phase 2: a team of agents led by a supervisor.

    START -> gate -> supervisor --Send--> ci_investigator --\\
                        ^   |   --Send--> code_analyst -----+--> supervisor ...
                        |   |   --Send--> tracker_analyst --/
                        |   `--(enough)--> synthesize -> review --(ok)--> END
                        `-------------------(revise)-----------'

- gate: decides from the user's request ALONE whether changes are allowed. It runs
  before any untrusted data (issues, commits, logs) enters any model's context, so an
  injection cannot influence it. Write-capable specialists are cut off in code if not.
- supervisor: an LLM with structured output that delegates. It sees reports, never raw
  tool output, and never calls tools itself.
- specialists: the single-agent graph from phase 1, each with its own prompt and a
  minimal tool set. They run as subgraphs, in parallel via Send, and return a short report.
- synthesize + review: one LLM writes the answer, another checks it against the reports;
  a deterministic grounding check catches invented hashes, issue numbers and links.
"""

from dataclasses import dataclass
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from devcopilot.agents import prompts
from devcopilot.agents.single import build_agent, count_tool_calls, initial_state
from devcopilot.config import get_settings
from devcopilot.decide import JevDecider, JevError, Noul, uses_jev
from devcopilot.grounding import extract_entities, ungrounded
from devcopilot.llm import (
    ROLES,
    get_chat_model,
    is_transient,
    llm_chaos,
    node_retry_policy,
    retry_observed,
    routed_models,
    same_family,
    structured,
)
from devcopilot.policy import requires_approval
from devcopilot.telemetry import log, metrics

MAX_PARALLEL = 3
REPORT_EVIDENCE_CHARS = 16_000  # tool output shown to the report validator, all calls together
UNVERIFIED_MARK = "\n\n---\n⚠ Reviewer could not verify:\n"


@dataclass(frozen=True)
class Specialist:
    description: str  # what the supervisor reads when choosing whom to call
    role: str  # what the specialist reads about itself
    tools: tuple[str, ...]  # least privilege: only what the role needs


SPECIALISTS = {
    "ci_investigator": Specialist(
        description="Finds failing CI runs, the error from the logs and the commit where failures started.",
        role="You analyze CI: which runs fail, since which commit, what the error is. "
        "Correlate runs with commits using git_log and git_show.",
        tools=("list_ci_runs", "get_ci_log", "git_log", "git_show"),
    ),
    "code_analyst": Specialist(
        description="Reads code and history to find root causes; can run tests to confirm a hypothesis.",
        role="You analyze source code and its history to find root causes: read code, search, "
        "inspect diffs, run tests to confirm hypotheses.",
        tools=("list_files", "read_file", "search_code", "git_log", "git_show", "run_tests"),
    ),
    "tracker_analyst": Specialist(
        description="Finds and summarizes issues in the tracker: status, owner, discussion. Read-only.",
        role="You analyze the issue tracker: find issues related to the task, summarize status, "
        "owners and discussion. You cannot change the tracker.",
        tools=("search_issues", "get_issue"),
    ),
    "fixer": Specialist(
        description="Implements a fix on a new copilot/* branch and runs the tests on it.",
        role="You implement fixes: read the code, create a branch with propose_patch, run tests on "
        "the branch, report the branch name and the test result.",
        tools=("read_file", "search_code", "propose_patch", "run_tests"),
    ),
    "tracker_writer": Specialist(
        description="Creates issues, adds comments, changes status/assignee in the tracker.",
        role="You change the issue tracker exactly as the task says: create issues, add comments, "
        "update status or assignee. Do only what the task says. Do not add links or facts "
        "that are not in the task.",
        tools=("search_issues", "get_issue", "create_issue", "add_comment", "update_issue"),
    ),
}

AgentName = Literal["ci_investigator", "code_analyst", "tracker_analyst", "fixer", "tracker_writer"]


# ------------------------------------------------------------ structured outputs
# Field order matters for small models: they generate left to right, so the
# reasoning field comes before the decision it should inform.
# With method="function_calling" each schema becomes a tool: the class name is the
# tool name and the docstring its description. The supervisor sometimes "called" a
# specialist as a tool (Unknown tool type: 'tracker_analyst'), found via the
# StructuredOutputRetries alert. Docstrings alone did not help; fixing the prompt that
# said "never use tools" halved it.


class Intent(BaseModel):
    """Submit whether the user's request allows changing anything. Call this exactly once."""

    reason: str = Field(description="One sentence: what the user asks for.")
    allow_changes: bool = Field(description="True only if the user explicitly asks to change something.")


class Assignment(BaseModel):
    agent: AgentName
    task: str = Field(description="Self-contained task with all facts the specialist needs.")


class SupervisorDecision(BaseModel):
    """Submit your decision. This is the only tool you can call: specialists are NOT tools,
    you run them by listing them in `assignments`."""

    thoughts: str = Field(description="1-3 sentences: what is known and what is still missing.")
    assignments: list[Assignment] = Field(
        description="Specialists to run next, in parallel. Empty list: enough information to answer."
    )


class Review(BaseModel):
    """Submit your review of the draft answer. Call this exactly once."""

    problems: list[str] = Field(description="Concrete problems in the draft. Empty if none.")
    verdict: Literal["ok", "revise"]


class ReportCheck(BaseModel):
    """Submit your check of the report. Call this exactly once."""

    unsupported: list[str] = Field(description="Claims of the report that the tool output does not support.")
    verdict: Literal["ok", "revise"]


def evidence_for_validator(messages: list[BaseMessage], budget: int = REPORT_EVIDENCE_CHARS) -> str:
    """Every tool call's output, each cut to an equal share of the budget.

    Not the tail of everything: the first measured version showed the validator the last
    12 000 characters, early outputs (git_log, read_file) fell out, and it rejected 100%
    of reports, correct `path:line` citations included, for lack of evidence it was not shown.
    """
    calls = {c["id"]: c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    outputs = [m for m in messages if isinstance(m, ToolMessage)]
    if not outputs:
        return ""
    share = budget // len(outputs)
    parts = []
    for m in outputs:
        call = calls.get(m.tool_call_id, {})
        text = m.text if len(m.text) <= share else m.text[:share] + "\n[... cut]"
        parts.append(f"## {call.get('name', m.name)}({call.get('args', {})})\n{text}")
    return "\n\n".join(parts)


def _warn_same_family(generator: BaseChatModel, validator: BaseChatModel) -> None:
    a, b = getattr(generator, "model", None), getattr(validator, "model", None)
    if isinstance(a, str) and isinstance(b, str) and same_family(a, b):
        log.warning("validator shares the generator's model family", extra={"generator": a, "validator": b})


# ------------------------------------------------------------------------ state


class Report(TypedDict):
    agent: str
    task: str
    summary: str  # the specialist's final message
    facts: list[str]  # entities seen in its tool outputs, for the grounding check
    steps: int
    tool_calls: int
    validator: NotRequired[str]  # "ok" | "revise": grounding, plus the LLM check when validate_reports


def add_or_reset(old: list, new: list | None) -> list:
    """operator.add that a new turn can reset with None: a follow-up question must not
    inherit the previous turn's refused assignments."""
    return [] if new is None else old + new


class TeamState(TypedDict):
    question: str
    allow_changes: bool
    # Per turn: a follow-up resets them (None, see add_or_reset); earlier turns live on
    # compacted in `history` and `known_facts` instead of in full.
    reports: Annotated[list[Report], add_or_reset]
    blocked: Annotated[list[str], add_or_reset]  # assignments refused by the gate in this turn
    last_decision: dict  # latest supervisor decision, for display
    rounds: int
    answer: str
    feedback: list[str]  # reviewer problems for the next round
    revisions: int
    memory: NotRequired[str]  # recalled episodes and facts (memory.py), not seen by the gate
    history: NotRequired[str]  # earlier turns of this dialogue as a digest (runner.next_turn)
    known_facts: NotRequired[list[str]]  # entities seen in tool output of earlier turns, for grounding


class SpecialistInput(TypedDict):
    task: str
    goal: str  # the user's request: context for the task, not an instruction to fulfil it all


def initial_team_state(question: str) -> TeamState:
    return {
        "question": question,
        "allow_changes": False,
        "reports": [],
        "blocked": [],
        "last_decision": {},
        "rounds": 0,
        "answer": "",
        "feedback": [],
        "revisions": 0,
    }


def render_context(state: TeamState, memory: bool = True) -> str:
    """What the supervisor, synthesizer and reviewer see: reports, not raw tool output.
    memory=False for the checks (review, stop): a recalled claim must not count as
    support for the draft that repeats it."""
    parts = [f"User request:\n{state['question']}", f"Changes allowed: {'yes' if state['allow_changes'] else 'no'}"]
    if state.get("history"):
        # Working memory of the dialogue, compacted: the request may refer to it ("and who wrote it?").
        parts.append(f"Earlier in this conversation (compacted):\n{state['history']}")
    if state["reports"]:
        reports = []
        for r in state["reports"]:
            summary = r["summary"].replace("</report", "&lt;/report")
            reports.append(f'<report agent="{r["agent"]}">\nTask: {r["task"]}\n\n{summary}\n</report>')
        parts.append("Reports (based on untrusted data):\n" + "\n\n".join(reports))
    else:
        parts.append("Reports: none yet.")
    if memory and state.get("memory"):
        parts.append(state["memory"])
    if state["blocked"]:
        parts.append(
            "Assignments NOT executed because the user did not ask for changes:\n"
            + "\n".join(f"- {b}" for b in state["blocked"])
        )
    if state["feedback"]:
        parts.append("Reviewer feedback on the previous draft:\n" + "\n".join(f"- {p}" for p in state["feedback"]))
    return "\n\n".join(parts)


def to_report(agent: str, task: str, result: dict) -> Report:
    messages = result["messages"]
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    tool_outputs = [m.text for m in messages if isinstance(m, ToolMessage)]
    return {
        "agent": agent,
        "task": task,
        "summary": final.text if final else "(no report: the agent produced no final message)",
        "facts": sorted(set().union(*(extract_entities(t) for t in tool_outputs))),
        "steps": result["steps"],
        "tool_calls": count_tool_calls(messages),
    }


# ------------------------------------------------------- prompts of the deciders
# Separate functions so that phase 4 can evaluate the gate and the supervisor as
# components: same prompt, one LLM call, no specialists (devcopilot/evals/components.py).


def gate_messages(question: str) -> list[BaseMessage]:
    return [SystemMessage(prompts.GATE), HumanMessage(question)]


def write_access(tools: list[BaseTool], specialists: dict[str, Specialist] = SPECIALISTS) -> dict[str, bool]:
    """Which specialists can change things. Derived from tool annotations, not declared by hand."""
    tools_by_name = {t.name: t for t in tools}
    return {n: any(requires_approval(tools_by_name[t]) for t in spec.tools) for n, spec in specialists.items()}


def available_specialists(
    state: TeamState, can_write: dict[str, bool], specialists: dict[str, Specialist] = SPECIALISTS
) -> dict[str, Specialist]:
    return {n: spec for n, spec in specialists.items() if state["allow_changes"] or not can_write[n]}


def supervisor_messages(
    state: TeamState, available: dict[str, Specialist], can_write: dict[str, bool]
) -> list[BaseMessage]:
    s = get_settings()
    agents = "\n".join(
        f"- {n}: {spec.description}{' (changes things)' if can_write[n] else ''}" for n, spec in available.items()
    )
    rule = prompts.CHANGES_ALLOWED if state["allow_changes"] else prompts.CHANGES_FORBIDDEN
    system = prompts.SUPERVISOR.format(agents=agents, team_rules=prompts.TEAM_RULES, changes_rule=rule)
    context = render_context(state) + f"\n\nRound {state['rounds'] + 1} of {s.team_max_rounds}."
    return [SystemMessage(system), HumanMessage(context)]


# ------------------------------------------------------------------------ graph


def build_team(
    tools: list[BaseTool],
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    specialists: dict[str, Specialist] = SPECIALISTS,
    jev_for: str | None = None,
    models: dict[str, BaseChatModel] | None = None,
    validate_reports: bool | None = None,
):
    """jev_for: components decided by Jev instead of the LLM (devcopilot/decide.py);
    None = settings.jev_for. Evals pass it per configuration.

    models: chat models by role (llm.ROLES); a role without one uses `model`.
    None = the routes from settings (llm.routed_models).
    validate_reports: the validator checks each specialist report; None = settings.
    """
    s = get_settings()
    model = model or get_chat_model()
    models = routed_models() if models is None else models
    gate_model, supervisor_model, specialist_model, synthesizer_model, validator_model = (
        models.get(role) or model for role in ROLES
    )
    for generator in {id(m): m for m in (specialist_model, synthesizer_model)}.values():
        _warn_same_family(generator, validator_model)
    check_reports = s.validate_reports if validate_reports is None else validate_reports
    # Jev variants of the three yes/no decisions; None = the LLM decides as before.
    jev_gate = JevDecider("gate", threshold=s.jev_gate_threshold) if uses_jev("gate", jev_for) else None
    jev_stop = JevDecider("stop") if uses_jev("stop", jev_for) else None
    jev_review = JevDecider("review") if uses_jev("review", jev_for) else None
    tools_by_name = {t.name: t for t in tools}
    decide_intent = structured(gate_model, Intent)
    decide_next = structured(supervisor_model, SupervisorDecision)
    review_draft = structured(validator_model, Review)
    # Report validation runs inside a specialist node: override its inherited agent
    # label so coordination usage is not charged to the specialist.
    check_report = RunnableLambda(structured(validator_model, ReportCheck)).with_config(
        metadata={"devcopilot.agent": "validator"}
    )

    # Specialists: the phase-1 agent with a role prompt and a subset of tools.
    subgraphs = {}
    for name, spec in specialists.items():
        missing = [t for t in spec.tools if t not in tools_by_name]
        if missing:
            raise ValueError(f"{name}: unknown tools {missing}")
        own_tools = [tools_by_name[t] for t in spec.tools]
        prompt = prompts.SPECIALIST.format(name=name, role=spec.role, rules=prompts.WORK_RULES)
        # checkpointer=None: the subgraph uses the team's checkpointer, so interrupt()
        # inside a specialist pauses the whole team.
        subgraphs[name] = build_agent(
            own_tools, specialist_model, max_steps=s.specialist_max_steps, system_prompt=prompt, name=name
        )
    can_write = write_access(tools, specialists)

    async def gate(state: TeamState) -> dict:
        if jev_gate:
            # Same property as the LLM gate: the state is the user's request alone.
            try:
                a = (await jev_gate.decide(state["question"], {"change": Noul(prompts.JEV_GATE)}))["change"]
            except JevError as e:
                # The decision server is down: the LLM gate decides, with the same input.
                # Loud on purpose (metric + alert): a silent fallback would hide the outage.
                metrics.decider_fallbacks.add(1, {"component": "gate"})
                log.warning("decision model unavailable, LLM gate decides", extra={"error": str(e)})
            else:
                log.info("gate decision", extra={"allow_changes": a.value, "p": a.p, "backend": "jev"})
                return {"allow_changes": a.value, "last_decision": {"gate": f"jev: p(change requested) = {a.p:.2f}"}}
        intent = await decide_intent(gate_messages(state["question"]))
        log.info("gate decision", extra={"allow_changes": intent.allow_changes, "reason": intent.reason})
        return {"allow_changes": intent.allow_changes, "last_decision": {"gate": intent.reason}}

    async def supervisor(state: TeamState) -> Command:
        if state["rounds"] >= s.team_max_rounds:
            return Command(goto="synthesize")
        if jev_stop and state["reports"] and not state["feedback"]:
            # The last round of a typical run is the supervisor saying "enough, answer":
            # a full LLM call for one bit. Jev answers it; when unsure, the LLM decides.
            try:
                context = render_context(state, memory=False)
                a = (await jev_stop.decide(context, {"enough": Noul(prompts.JEV_STOP)}))["enough"]
            except JevError as e:  # the LLM supervisor decides, as without the decision model
                metrics.decider_fallbacks.add(1, {"component": "stop"})
                log.warning("decision model unavailable, LLM supervisor decides", extra={"error": str(e)})
            else:
                if a.p >= s.jev_stop_threshold:
                    decision = {"thoughts": f"jev: reports are enough (p = {a.p:.2f})", "assignments": []}
                    update = {"rounds": state["rounds"] + 1, "blocked": [], "last_decision": decision}
                    return Command(goto="synthesize", update=update)
        available = available_specialists(state, can_write, specialists)
        decision = await decide_next(supervisor_messages(state, available, can_write))

        sends, blocked = [], []
        # Every assignment is checked, only the runs are capped: a writer asked for in
        # 4th place is still a blocked (possibly injected) request worth counting.
        for a in decision.assignments:
            if a.agent in available:
                if len(sends) < MAX_PARALLEL:
                    sends.append(Send(a.agent, {"task": a.task, "goal": state["question"]}))
            else:
                # Enforced in code: the prompt says these agents do not exist, but a
                # model under injection may ask for them anyway. Record it, do not run it.
                blocked.append(f"{a.agent}: {a.task}")
        if blocked:
            # Read-only request, yet the supervisor wants to change things: possible injection.
            metrics.gate_blocked.add(len(blocked))
            log.warning("assignment blocked by gate", extra={"blocked": blocked, "round": state["rounds"] + 1})
        update = {
            "rounds": state["rounds"] + 1,
            "blocked": blocked,
            "last_decision": decision.model_dump(),
            "feedback": [] if sends else state["feedback"],
        }
        return Command(goto=sends or "synthesize", update=update)

    def make_specialist(name: str):
        subgraph = subgraphs[name]

        async def run_specialist(inp: SpecialistInput, config: RunnableConfig) -> dict:
            # The specialist sees its task and the team's goal, nothing else. Its whole
            # conversation stays inside the subgraph; only the report goes back.
            brief = f"Team goal (the user's request, for context): {inp['goal']}\n\nYour task: {inp['task']}"
            # The inherited checkpoint resumes the model after completed tools; their
            # messages stay in state. Replayed writes also keep thread + call-id keys.
            result = await subgraph.ainvoke(initial_state(brief), config)
            report = ground_report(to_report(name, inp["task"], result), brief, result["messages"])
            if check_reports:
                report = await validate_report(report, result["messages"])
            metrics.validator_verdicts.add(1, {"stage": "report", "verdict": report["validator"]})
            return {"reports": [report]}

        return run_specialist

    def ground_report(report: Report, brief: str, messages: list[BaseMessage]) -> Report:
        # Intermediate validation, always on: every specific of the report (hash, #N, file,
        # run N) must appear in THIS agent's tool output or in its brief. Deterministic and
        # free. An LLM check of reports (validate_report) is optional: measured on 9 real
        # reports, gemma4:e4b rejected 9/9 without thinking and took ~90 s per report with it.
        sources = [brief, *(m.text for m in messages if isinstance(m, ToolMessage))]
        invented = ungrounded(report["summary"], sources)
        verdict = "revise" if invented else "ok"
        if not invented:
            return {**report, "validator": verdict}
        log.warning("report mentions unseen entities", extra={"agent": report["agent"], "entities": invented})
        note = "\n\nValidator: not found in this agent's tool output: " + ", ".join(invented)
        return {**report, "summary": report["summary"] + note, "validator": verdict}

    async def validate_report(report: Report, messages: list[BaseMessage]) -> Report:
        # Intermediate validation: a report is the specialist's reading of tool output,
        # and the supervisor never sees that output. A model of another family checks
        # the reading before it shapes the next decision.
        evidence = evidence_for_validator(messages)
        prompt = f"Tool output:\n{evidence or '(none)'}\n\nReport:\n<report>\n{report['summary']}\n</report>"
        try:
            check = await check_report.ainvoke([SystemMessage(prompts.REPORT_VALIDATOR), HumanMessage(prompt)])
        except Exception as e:  # malformed validation still degrades to deterministic checks
            if is_transient(e):
                raise
            log.warning("report validation failed", extra={"agent": report["agent"], "error": str(e)})
            return report
        log.info("report validated", extra={"agent": report["agent"], "verdict": check.verdict})
        # Rejected by grounding or by the model: either is a rejection of the report.
        verdict = "revise" if "revise" in (check.verdict, report.get("validator")) else "ok"
        report = {**report, "validator": verdict}
        if check.verdict == "ok" or not check.unsupported:
            return report
        note = "\n\nValidator: not supported by this agent's tool output:\n" + "\n".join(
            f"- {c}" for c in check.unsupported
        )
        return {**report, "summary": report["summary"] + note}

    async def synthesize(state: TeamState) -> dict:
        llm_chaos()
        answer = await synthesizer_model.ainvoke(
            [SystemMessage(prompts.SYNTHESIZER), HumanMessage(render_context(state))]
        )
        return {"answer": answer.text}

    async def review(state: TeamState) -> Command:
        # 1. Deterministic: specifics in the answer must appear in some tool output.
        sources = [state["question"], *state.get("known_facts", []), *(f for r in state["reports"] for f in r["facts"])]
        invented = [f"Not found in any tool output: {e}" for e in ungrounded(state["answer"], sources)]
        # 2. Model review: unsupported statements, unanswered parts.
        draft = f"{render_context(state, memory=False)}\n\nDraft answer:\n<draft>\n{state['answer']}\n</draft>"
        found: list[str] = []
        try:
            if jev_review:
                # Fixed problem texts instead of the reviewer's own words: less specific
                # feedback, but the 9B reviewer's words were too noisy to show anyway.
                questions = {k: Noul(q) for k, (q, _) in prompts.JEV_REVIEW.items()}
                answers = await jev_review.decide(draft, questions)
                found = [prompts.JEV_REVIEW[k][1] for k, a in answers.items() if a.value]
            else:
                found = list((await review_draft([SystemMessage(prompts.REVIEWER), HumanMessage(draft)])).problems)
        except Exception as e:  # malformed review still leaves the deterministic check
            if is_transient(e):
                raise
            if isinstance(e, JevError):
                metrics.decider_fallbacks.add(1, {"component": "review"})
            log.warning("review failed, deterministic check only", extra={"error": str(e)})
        # One rule for routing and the metric: a "revise" without problems would send the
        # team back with no feedback to act on.
        problems = invented + found
        metrics.validator_verdicts.add(1, {"stage": "answer", "verdict": "ok" if not problems else "revise"})
        log.info("review", extra={"problems": problems, "invented": invented, "revision": state["revisions"]})
        if not problems:
            return Command(goto=END, update={"feedback": [], "last_decision": {"review": "ok"}})
        if state["revisions"] >= s.team_max_revisions:
            # Out of budget: deliver, but mark what could not be verified. Only the
            # deterministic findings: the 9B reviewer's problems are too noisy to show users.
            note = UNVERIFIED_MARK + "\n".join(f"- {p}" for p in invented) if invented else ""
            return Command(
                goto=END,
                update={"answer": state["answer"] + note, "feedback": problems, "last_decision": {"review": problems}},
            )
        return Command(
            goto="supervisor",
            update={"feedback": problems, "revisions": state["revisions"] + 1, "last_decision": {"review": problems}},
        )

    graph = StateGraph(TeamState)
    graph.add_node("gate", retry_observed(gate), retry_policy=node_retry_policy())
    # destinations: only for drawing the graph; Command decides at runtime.
    graph.add_node(
        "supervisor",
        retry_observed(supervisor),
        retry_policy=node_retry_policy(),
        destinations=(*specialists, "synthesize"),
    )
    for name in specialists:
        # No retry policy here: the specialist's own model node retries inside the subgraph.
        # Retrying this outer node would restart the specialist's conversation from its
        # brief, the model would ask for the same write again with a new call id (a new
        # idempotency key) and the human would be asked, and the tool run, twice.
        graph.add_node(name, retry_observed(make_specialist(name), with_config=True))
        graph.add_edge(name, "supervisor")
    graph.add_node("synthesize", retry_observed(synthesize), retry_policy=node_retry_policy())
    graph.add_node("review", retry_observed(review), retry_policy=node_retry_policy(), destinations=("supervisor", END))
    graph.add_edge(START, "gate")
    graph.add_edge("gate", "supervisor")
    graph.add_edge("synthesize", "review")
    return graph.compile(checkpointer=checkpointer)
