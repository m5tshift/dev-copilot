"""Component evals: the gate and the supervisor's first decision, without specialists.

An end-to-end team run costs 10-20 LLM calls and minutes; these two decisions cost
one call each. When the team fails a task, they tell whether the routing was wrong
from the start or the specialists failed later.

The supervisor gets the EXPECTED gate decision, not the gate's actual output: each
component is measured in isolation, so gate errors do not show up as routing errors.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from devcopilot.agents import prompts
from devcopilot.agents.team import (
    Intent,
    SupervisorDecision,
    available_specialists,
    gate_messages,
    initial_team_state,
    supervisor_messages,
    write_access,
)
from devcopilot.decide import Decider, Noul
from devcopilot.evals.dataset import Task
from devcopilot.evals.stats import confusion, mean
from devcopilot.llm import StructuredOutputError, structured


async def eval_components(
    tasks: list[Task],
    model: BaseChatModel,
    tools: list[BaseTool],
    repeat: int,
    on_record: Callable[[dict], None],
    done: list[dict] = (),
    gate: Decider | None = None,
    gate_only: bool = False,
) -> list[dict]:
    """`done`: records from an interrupted eval; their (task, repeat) pairs are skipped.
    `gate`: decide the gate with this decider (Jev) instead of the LLM; records its p.
    `gate_only`: skip the supervisor (comparing gate backends needs no LLM)."""
    can_write = write_access(tools)
    decide_intent, decide_next = structured(model, Intent), structured(model, SupervisorDecision)
    records = list(done)
    skip = {(r["task"], r["repeat"]) for r in done}
    for i in range(repeat):
        for task in tasks:
            if (task.id, i) in skip:
                continue
            record = {"task": task.id, "repeat": i, "gate": None, "gate_reason": "", "assignments": [], "error": None}
            try:
                if gate:
                    a = (await gate.decide(task.question, {"change": Noul(prompts.JEV_GATE)}))["change"]
                    record["gate"], record["gate_p"], record["gate_reason"] = a.value, a.p, gate.name
                else:
                    intent = await decide_intent(gate_messages(task.question))
                    record["gate"], record["gate_reason"] = intent.allow_changes, intent.reason
                if gate_only:  # measure the gate alone: no LLM call at all
                    records.append(record)
                    on_record(record)
                    continue
                state = initial_team_state(task.question)
                state["allow_changes"] = task.allow_changes
                available = available_specialists(state, can_write)
                decision = await decide_next(supervisor_messages(state, available, can_write))
                record["assignments"] = [{"agent": a.agent, "task": a.task} for a in decision.assignments]
                record["available"] = sorted(available)
            except StructuredOutputError as e:
                record["error"] = str(e)
            except Exception as e:  # Ollama errors etc.: one failed decision must not stop the eval
                record["error"] = f"{type(e).__name__}: {e}"
            records.append(record)
            on_record(record)
    return records


def summarize_components(records: list[dict], tasks: dict[str, Task]) -> dict:
    # A resumed --out file may hold tasks of another split or removed from the dataset.
    records = [r for r in records if r["task"] in tasks]
    gated = [r for r in records if r["gate"] is not None]
    truth = [tasks[r["task"]].allow_changes for r in gated]
    pred = [r["gate"] for r in gated]
    gate = confusion(truth, pred)

    routed = [r for r in records if not r["error"] and tasks[r["task"]].route]
    hits, precisions, empty, unavailable = [], [], 0, 0
    misses: dict[str, list[str]] = {}
    for r in routed:
        expected = set(tasks[r["task"]].route)
        assigned = [a["agent"] for a in r["assignments"]]
        if not assigned:
            empty += 1  # answered without asking anyone: a guess
        unavailable += sum(a not in r.get("available", assigned) for a in assigned)
        hits.append(bool(expected & set(assigned)))
        if assigned:
            precisions.append(len([a for a in assigned if a in expected]) / len(assigned))
        if not expected & set(assigned):
            misses.setdefault(r["task"], []).append(",".join(assigned) or "(none)")
    return {
        "decisions": len(records),
        "errors": sum(bool(r["error"]) for r in records),
        "gate": {
            **gate,
            "accuracy": mean([t == p for t, p in zip(truth, pred, strict=True)]),
            # Allowed changes the user did not ask for: the dangerous direction.
            "false_allow": [r["task"] for r, t in zip(gated, truth, strict=True) if r["gate"] and not t],
            "false_deny": [r["task"] for r, t in zip(gated, truth, strict=True) if not r["gate"] and t],
        },
        "routing": {
            "hit_rate": mean(hits),  # at least one relevant specialist in round 1
            "precision": mean(precisions),  # share of assigned specialists that are relevant
            "empty": empty,
            "unavailable": unavailable,  # would be blocked by the gate in a real run
            "misses": misses,
        },
    }
