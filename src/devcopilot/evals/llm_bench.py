"""Level-1 evals: base LLMs alone, before they become agents (evals/llm_bench.yaml).

The agent evals (run.py) measure the whole system; when a model fails there, it is
unclear whether it cannot reason, cannot call tools or drowns in a long context.
This bench isolates the four abilities the assignment names (TZ 2.2): reasoning,
faithfulness, structured output, tool calling. Short prompts, deterministic checks,
temperature 0: a model is chosen for a role by the ability the role needs.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from devcopilot.config import PROJECT_ROOT
from devcopilot.llm import get_chat_model, has_capability

BENCH = PROJECT_ROOT / "evals" / "llm_bench.yaml"
CRITERIA = ("reasoning", "faithfulness", "structured", "tools")

ANSWER_SYSTEM = "Answer using only the context. Be brief."
JSON_SYSTEM = "Reply with a single JSON object and nothing else."
TOOLS_SYSTEM = "You are an assistant with tools. Call a tool when the request needs one; otherwise just reply."


@dataclass
class Item:
    id: str
    criterion: str
    question: str
    context: str = ""
    expect: list[str] = field(default_factory=list)
    any_of: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    regex: str = ""
    schema: dict | None = None
    strict_json: bool = False
    calls: list[dict] | None = None  # expected tool calls; [] = no call


def load_bench(path: Path = BENCH) -> tuple[list[Item], list[dict]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    contexts = data["contexts"]
    items = []
    for raw in data["items"]:
        raw = dict(raw)
        ctx = contexts.get(raw.pop("context", ""), "")
        if extra := raw.pop("context_extra", None):
            ctx += "\n" + contexts[extra]
        if "call" in raw:
            call = raw.pop("call")
            raw["calls"] = [] if call is None else [call]
        item = Item(context=ctx, **raw)
        if item.criterion not in CRITERIA:
            raise ValueError(f"{item.id}: unknown criterion {item.criterion!r}")
        items.append(item)
    tools = [{"type": "function", "function": t} for t in data["tools"]]
    return items, tools


def check_text(item: Item, text: str) -> tuple[bool, str]:
    low = text.lower()
    missing = [e for e in item.expect if e.lower() not in low]
    if missing:
        return False, f"missing {missing}"
    if item.any_of and not any(a.lower() in low for a in item.any_of):
        return False, f"none of {item.any_of}"
    if bad := [f for f in item.forbid if f.lower() in low]:
        return False, f"forbidden {bad}"
    if item.regex and not re.search(item.regex, text.strip(), re.IGNORECASE):
        return False, f"no match for {item.regex!r}"
    return True, ""


def parse_json(text: str, strict: bool) -> object:
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
    if fenced:
        body = fenced.group(1)
    if not strict and not body.startswith("{"):
        match = re.search(r"\{.*\}", body, re.DOTALL)  # prose around the JSON is tolerated
        body = match.group(0) if match else body
    return json.loads(body)


def check_json(item: Item, text: str) -> tuple[bool, str]:
    try:
        data = parse_json(text, item.strict_json)
    except json.JSONDecodeError as e:
        return False, f"not JSON: {e.msg}"
    try:
        jsonschema.validate(data, item.schema or {})
    except jsonschema.ValidationError as e:
        return False, f"schema: {e.message[:120]}"
    return True, ""


def _arg_matches(expected, actual) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.lower() in actual.lower()
    if isinstance(expected, bool):  # bool("false") is True
        return str(actual).lower() == str(expected).lower()
    try:
        return type(expected)(actual) == expected  # "107" for 107 is fine
    except (TypeError, ValueError):
        return False


def check_calls(item: Item, calls: list[dict]) -> tuple[bool, str]:
    expected = item.calls or []
    if not expected:
        return (not calls, "" if not calls else f"called {[c['name'] for c in calls]} but no tool was needed")
    remaining = list(calls)
    for exp in expected:
        hit = next(
            (c for c in remaining if c["name"] == exp["name"]
             and all(_arg_matches(v, c["args"].get(k)) for k, v in exp["args"].items())),
            None,
        )  # fmt: skip
        if hit is None:
            got = [(c["name"], c["args"]) for c in calls]
            return False, f"expected {exp['name']}({exp['args']}), got {got}"
        remaining.remove(hit)
    if remaining:
        return False, f"extra calls {[c['name'] for c in remaining]}"
    return True, ""


async def run_item(model_name: str, item: Item, tools: list[dict], reasoning: bool = False) -> dict:
    model = await asyncio.to_thread(get_chat_model, model_name, reasoning=reasoning)
    started = time.perf_counter()
    try:
        if item.criterion == "tools":
            if not await asyncio.to_thread(has_capability, model_name, "tools"):
                return {"item": item.id, "criterion": item.criterion, "pass": False, "detail": "no tool support",
                        "seconds": 0.0}  # fmt: skip
            msg = await model.bind_tools(tools).ainvoke([SystemMessage(TOOLS_SYSTEM), HumanMessage(item.question)])
            ok, detail = check_calls(item, msg.tool_calls)
            answer = json.dumps([{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]) or msg.text
        else:
            system = JSON_SYSTEM if item.criterion == "structured" else ANSWER_SYSTEM
            prompt = f"Context:\n{item.context}\n\n{item.question}" if item.context else item.question
            msg = await model.ainvoke([SystemMessage(system), HumanMessage(prompt)])
            answer = msg.text
            ok, detail = check_json(item, answer) if item.criterion == "structured" else check_text(item, answer)
    except Exception as e:  # a failing model is a failed item, not a crashed bench
        ok, detail, answer = False, f"{type(e).__name__}: {e}"[:200], ""
    return {
        "item": item.id,
        "criterion": item.criterion,
        "pass": ok,
        "detail": detail,
        "answer": answer[:500],
        "seconds": round(time.perf_counter() - started, 2),
    }


def summarize(records: list[dict]) -> dict[str, dict[str, float]]:
    """{model: {criterion: pass rate, ..., "all": ..., "seconds": mean}}."""
    out: dict[str, dict[str, float]] = {}

    def key(r):
        host = r.get("host", "localhost:11434")
        return r["model"] + ("" if host == "localhost:11434" else f"@{host}")

    for model in dict.fromkeys(key(r) for r in records):
        rs = [r for r in records if key(r) == model]
        row = {c: sum(r["pass"] for r in rs if r["criterion"] == c) / max(1, sum(r["criterion"] == c for r in rs))
               for c in CRITERIA}  # fmt: skip
        row["all"] = sum(r["pass"] for r in rs) / len(rs)
        row["seconds"] = sum(r["seconds"] for r in rs) / len(rs)
        out[model] = row
    return out
