"""Online evals: reference-free checks of real runs, read from Phoenix and written back.

Offline evals answer "is this version good on our dataset?". Real questions are not
in the dataset and have no reference answer, so online evals check only what can be
checked without one:
- grounded: entities in the answer appear in the tool outputs of the same trace
  (the phase-2 check, now on production traffic and with no access to the agent's state);
- language: the answer is in the language of the question;
- answers_question, admits_gaps: an LLM judge without a reference: weaker and noisier,
  useful for finding runs worth a human look, not as a quality number.

Results go back to Phoenix as annotations on the run's root span: in the UI they
filter traces ("runs with ungrounded answers"), and a human can confirm or correct
them, which is how production runs become new labeled eval tasks.

Continuous evaluation (`eval online --watch`): every few minutes the new traces are
checked once each (a sampling rate, not a random sample of everything) and every
verdict is also a metric, devcopilot.online_eval.checks{check, result}, so quality
over time is a Grafana panel and a drop is an alert (OnlineQualityDrop), like latency
or errors. answers_question is the reference-free Answer Relevance of the assignment.
"""

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import httpx
from langchain_core.language_models import BaseChatModel

from devcopilot.agents import prompts
from devcopilot.decide import Decider, Noul
from devcopilot.evals.checks import language
from devcopilot.evals.judge import judge_online
from devcopilot.grounding import ungrounded
from devcopilot.telemetry import PROJECT, metrics

ROOT_SPAN = "devcopilot.run"


def _attr(span: dict, key: str):
    """Span attributes may come flat ("output.value") or nested ({"output": {"value": ...}})."""
    attrs = span.get("attributes") or {}
    if key in attrs:
        return attrs[key]
    node = attrs
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _kind(span: dict) -> str | None:
    return span.get("span_kind") or _attr(span, "openinference.span.kind")


def tool_text(value) -> str:
    """The text a tool returned, from a TOOL span's output.value.

    OpenInference stores the whole ToolMessage as JSON: {"type": "tool", "data":
    {"content": [{"type": "text", "text": ...}]}}. Searching that raw string breaks the
    grounding check: an escaped newline glues "\\n" to the next word, and
    "\\nshortlink/expiry.py" no longer matches "shortlink/expiry.py".
    """
    if not isinstance(value, str):
        return "" if value is None else str(value)
    try:
        data = json.loads(value)
    except ValueError:
        return value
    content = data.get("data", data).get("content") if isinstance(data, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
    return value


def fetch_spans(client: httpx.Client, limit: int, project: str = PROJECT) -> list[dict]:
    spans: list[dict] = []
    cursor = None
    while len(spans) < limit:
        params: dict = {"limit": min(1000, limit - len(spans))}
        if cursor:
            params["cursor"] = cursor
        response = client.get(f"/v1/projects/{project}/spans", params=params)
        response.raise_for_status()
        body = response.json()
        spans += body["data"]
        cursor = body.get("next_cursor")
        if not cursor:
            break
    return spans


def runs_from_spans(spans: list[dict]) -> list[dict]:
    """One entry per trace with a root run span: question, answer and every tool output."""
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_trace[s["context"]["trace_id"]].append(s)
    runs = []
    for trace_id, trace in by_trace.items():
        root = next((s for s in trace if s["name"] == ROOT_SPAN), None)
        if root is None or not _attr(root, "output.value"):
            continue  # failed run, or the trace is only partly within the fetched window
        runs.append(
            {
                "trace_id": trace_id,
                "span_id": root["context"]["span_id"],
                "start_time": root.get("start_time", ""),
                "mode": _attr(root, "devcopilot.mode"),
                "question": _attr(root, "input.value") or "",
                "answer": _attr(root, "output.value"),
                "tool_outputs": [tool_text(_attr(s, "output.value")) for s in trace if _kind(s) == "TOOL"],
            }
        )
    return sorted(runs, key=lambda r: r["start_time"], reverse=True)


async def check_run(run: dict, model: BaseChatModel | None, jev: Decider | None = None) -> dict:
    """`model`: the LLM judge; `jev`: Jev instead of it (cheap enough to check every run)."""
    found = ungrounded(run["answer"], [run["question"], *run["tool_outputs"]])
    expected, got = language(run["question"], min_letters=1), language(run["answer"])
    out = {
        "grounded": (not found, f"not in any tool output: {', '.join(found)}" if found else ""),
        "language": (got is None or got == expected, f"question {expected}, answer {got}"),
    }
    if jev is not None:
        state = {"question": run["question"], "answer": run["answer"]}
        answers = await jev.decide(state, {k: Noul(q) for k, q in prompts.JEV_ONLINE.items()})
        for name, a in answers.items():
            out[name] = (bool(a.value), f"jev p = {a.p:.2f}")
    elif model is not None:
        j = await judge_online(model, run["question"], run["answer"])
        out["answers_question"] = (j.answers_question, j.reason)
        out["admits_gaps"] = (j.admits_gaps, j.reason)
    return out


def to_annotations(run: dict, checks: dict, judge_model: str | None) -> list[dict]:
    rows = []
    for name, (ok, why) in checks.items():
        by_llm = name in ("answers_question", "admits_gaps")
        rows.append(
            {
                "span_id": run["span_id"],
                "name": name,
                "annotator_kind": "LLM" if by_llm else "CODE",
                "result": {"label": "pass" if ok else "fail", "score": 1.0 if ok else 0.0, "explanation": why},
                "metadata": {"judge": judge_model} if by_llm else {},
            }
        )
    return rows


def record_metrics(checks: dict) -> None:
    for name, (ok, _) in checks.items():
        metrics.online_checks.add(1, {"check": name, "result": "pass" if ok else "fail"})


def sampled(trace_id: str, rate: float) -> bool:
    """Deterministic sampling: the same trace is in or out on every tick and every host."""
    return int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


def load_seen(path: Path) -> set[str]:
    return set(json.loads(path.read_text(encoding="utf-8"))["seen"]) if path.exists() else set()


def save_seen(path: Path, seen: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"seen": sorted(seen)}), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a killed watcher never leaves a broken state file


async def watch_tick(
    client: httpx.Client, state: Path, model: BaseChatModel | None, judge_model: str | None,
    rate: float = 1.0, max_runs: int = 20, span_limit: int = 5000, jev: Decider | None = None,
) -> list[tuple[dict, dict]]:  # fmt: skip
    """One tick of continuous evaluation: check the finished runs not seen before.

    A run still in progress has no output yet and is not marked seen: the next tick takes
    it. Runs left out by sampling are marked seen too, so a tick never reconsiders them.
    At most max_runs per tick (newest first): the LLM judge shares the GPU with the agents.
    """
    seen = load_seen(state)
    fresh = [r for r in runs_from_spans(fetch_spans(client, span_limit)) if r["trace_id"] not in seen]
    picked = [r for r in fresh if sampled(r["trace_id"], rate)][:max_runs]
    skipped = {r["trace_id"] for r in fresh if not sampled(r["trace_id"], rate)}
    results = []
    for run in picked:
        checks = await check_run(run, model, jev)
        response = client.post("/v1/span_annotations", json={"data": to_annotations(run, checks, judge_model)})
        response.raise_for_status()
        record_metrics(checks)
        results.append((run, checks))
        seen.add(run["trace_id"])
        save_seen(state, seen | skipped)  # after each run: a crash mid-tick loses nothing
    save_seen(state, seen | skipped)
    return results


async def online_eval(
    phoenix_url: str, sample: int, model: BaseChatModel | None, judge_model: str | None,
    span_limit: int = 5000, seed: int | None = None, dry_run: bool = False, jev: Decider | None = None,
) -> list[tuple[dict, dict]]:  # fmt: skip
    with httpx.Client(base_url=phoenix_url, timeout=30) as client:
        runs = runs_from_spans(fetch_spans(client, span_limit))
        # A random sample, not "the latest N": recent traffic is not representative
        # (one user, one feature). In production this is a sampling rate, e.g. 5%.
        picked = random.Random(seed).sample(runs, min(sample, len(runs)))
        results = []
        for run in picked:
            checks = await check_run(run, model, jev)
            results.append((run, checks))
            if not dry_run:
                response = client.post("/v1/span_annotations", json={"data": to_annotations(run, checks, judge_model)})
                response.raise_for_status()
                record_metrics(checks)
    return results
