"""The trajectory of a run: which agent called which tool, how the team was routed.

Collected from the graph's update stream (runner.py), so it is the same for the
single agent and the team and needs no telemetry backend. Phoenix shows the same
trajectory to a human (phase 3); this is the machine-readable copy for evals (phase 4).
"""

from dataclasses import asdict, dataclass, field

from langchain_core.messages import AIMessage, ToolMessage


@dataclass
class ToolStep:
    agent: str  # "single" or the specialist's name
    tool: str
    args: dict
    # success, error, rejected, unknown_tool (see single.py); pending: no result seen.
    outcome: str = "pending"


@dataclass
class Trajectory:
    tools: list[ToolStep] = field(default_factory=list)
    # Supervisor rounds: {"assignments": [agent, ...], "blocked": [agent, ...]}.
    routing: list[dict] = field(default_factory=list)
    revisions: int = 0
    budget_exhausted: int = 0
    allow_changes: bool | None = None  # the gate's decision; None for the single agent
    # Validator verdicts in order: {"stage": "report" | "answer", "verdict": "ok" | "revise"}.
    validations: list[dict] = field(default_factory=list)
    # (tool, text) of successful calls, for the facts of memory.py; not stored in evals.
    outputs: list[tuple[str, str]] = field(default_factory=list, repr=False)
    _by_id: dict[str, ToolStep] = field(default_factory=dict, repr=False)

    def observe(self, namespace: tuple[str, ...], update: dict) -> None:
        # namespace is () for the top graph, ("code_analyst:<task id>",) inside a specialist.
        agent = namespace[0].split(":")[0] if namespace else "single"
        for node, delta in update.items():
            if not delta:
                continue
            self.budget_exhausted += delta.get("budget_exhausted", 0)
            for msg in delta.get("messages", []):
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        step = ToolStep(agent, call["name"], call["args"])
                        self.tools.append(step)
                        self._by_id[call["id"]] = step
                elif isinstance(msg, ToolMessage) and (step := self._by_id.get(msg.tool_call_id)):
                    step.outcome = (msg.artifact or {}).get("outcome", msg.status)
                    if step.outcome == "success":
                        self.outputs.append((step.tool, msg.text))
            for report in delta.get("reports", []) if not namespace else []:
                if verdict := report.get("validator"):
                    self.validations.append({"stage": "report", "verdict": verdict})
            if node == "review" and "revisions" in delta:
                self.revisions = delta["revisions"]
            if node == "review" and "last_decision" in delta:
                ok = delta["last_decision"].get("review") == "ok"
                self.validations.append({"stage": "answer", "verdict": "ok" if ok else "revise"})
            if node == "gate" and "allow_changes" in delta:
                self.allow_changes = delta["allow_changes"]
            elif node == "supervisor" and "rounds" in delta:
                self.routing.append(
                    {
                        "assignments": [a["agent"] for a in delta["last_decision"]["assignments"]],
                        "blocked": [b.split(":")[0] for b in delta["blocked"]],
                    }
                )

    def tool_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.tools]
