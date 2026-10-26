"""Runs tasks through configurations and stores raw results as JSONL; judges them later.

Running and judging are separate passes on purpose:
- the judge can be changed and rerun on the same answers (calibration needs this);
- only one model is loaded at a time: 16 GB do not fit an agent model and a judge.

A results file can be resumed: records already in it are skipped. A run takes
from 10 s to several minutes, so a matrix of configurations is hours, not minutes.
"""

import asyncio
import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from devcopilot.config import PROJECT_ROOT, get_settings
from devcopilot.decide import JevDecider, jev_components
from devcopilot.evals.dataset import Task
from devcopilot.evals.judge import judge_answer, judge_answer_jev
from devcopilot.llm import ROLES, get_chat_model, parse_routes, routed_models, unload_model
from devcopilot.mcp_client import open_mcp_tools
from devcopilot.runner import Mode, reject_all, run_question
from devcopilot.telemetry import log
from devcopilot.world.seed import seed

# In the sandbox the project is read-only: results go to the writable volume.
RESULTS = Path(os.environ.get("DEVCOPILOT_RESULTS_DIR", PROJECT_ROOT / "results")) / "evals"


@dataclass(frozen=True)
class Config:
    model: str
    reasoning: bool = False
    mode: Mode = "single"
    jev: str = ""  # team components decided by Jev, e.g. "gate,review" (decide.py)
    routes: str = ""  # team roles on their own models, e.g. "validator=gemma4:e4b" (llm.py)
    validate: bool = False  # the validator also checks every specialist report

    chaos: float = 0.0
    llmchaos: float = 0.0

    @property
    def name(self) -> str:
        routes = "".join(f"+{r}={m}" for r, m in parse_routes(self.routes).items())
        return (
            f"{self.mode}/{self.model}{'+think' if self.reasoning else ''}"
            f"{f'+jev={self.jev}' if self.jev else ''}{routes}{'+validate' if self.validate else ''}"
            f"{f'+chaos={self.chaos}' if self.chaos else ''}"
            f"{f'+llmchaos={self.llmchaos}' if self.llmchaos else ''}"
        )

    @classmethod
    def parse(cls, text: str) -> "Config":
        """'single/qwen3.5:9b', 'team/qwen3.5:9b+think', 'team/qwen3.5:9b+jev=gate,review,stop',
        'team/qwen3.5:9b+gate=qwen3.5:4b+validator=gemma4:e4b'."""
        usage = (
            "expected single|team/<model>[+think][+jev=<components>][+<role>=<model>...]"
            "[+validate][+chaos=<0..1>][+llmchaos=<0..1>]"
        )
        mode, _, spec = text.partition("/")
        model, *flags = spec.split("+")
        jev = next((f.removeprefix("jev=") for f in flags if f.startswith("jev=")), "")
        route_flags = [f for f in flags if f.partition("=")[0] in ROLES]
        chaos_flags = [f for f in flags if f.startswith("chaos=")]
        chaos = float(chaos_flags[0].partition("=")[2]) if chaos_flags else 0.0
        if len(chaos_flags) > 1 or not 0 <= chaos <= 1:
            raise ValueError("chaos must be a probability between 0 and 1")
        llm_flags = [f for f in flags if f.startswith("llmchaos=")]
        llmchaos = float(llm_flags[0].partition("=")[2]) if llm_flags else 0.0
        if len(llm_flags) > 1 or not 0 <= llmchaos <= 1:
            raise ValueError("llmchaos must be a probability between 0 and 1")
        known = (
            {"think", "validate"}
            | {f for f in flags if f.startswith("jev=")}
            | set(route_flags)
            | set(chaos_flags)
            | set(llm_flags)
        )
        if mode not in ("single", "team") or not model or any(f not in known for f in flags):
            raise ValueError(f"bad config {text!r}: {usage}")
        if (jev or route_flags or "validate" in flags) and mode != "team":
            raise ValueError(f"bad config {text!r}: Jev and routes are about team components, use team/...")
        jev_components(jev)  # validates the names
        # Canonical order (llm.ROLES), so the same routes always give the same name.
        routes = parse_routes(",".join(route_flags))
        routes_text = ",".join(f"{r}={routes[r]}" for r in ROLES if r in routes)
        return cls(model, "think" in flags, mode, jev, routes_text, "validate" in flags, chaos, llmchaos)


JEV = "jev"  # judge "model" name that selects the Jev backend


def judge_key(model: str, reasoning: bool) -> str:
    if model == JEV:
        return f"jev:{get_settings().jev_model}"
    return f"{model}{'+think' if reasoning else ''}"


def llm_host() -> str:
    """Where the model ran. Seconds from another machine (a PC GPU over a tunnel)
    are not comparable with this Mac's, so report.py keeps them apart."""
    return urlparse(get_settings().ollama_base_url).netloc.replace("127.0.0.1", "localhost")


def approve_all(payload: dict) -> dict:
    """For tasks that ask for changes. Safe here: the world is reseeded before every run."""
    return {c["id"]: {"action": "approve"} for c in payload["calls"]}


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            tail = line[error.pos :].strip()
            incomplete = (
                error.pos >= len(line.rstrip())
                or error.msg.startswith("Unterminated string")
                or (
                    error.msg == "Expecting value"
                    and tail in {"t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul"}
                )
            )
            if i != len(lines) - 1 or line.endswith("\n") or not incomplete:
                raise
            log.warning("dropping incomplete trailing JSONL record", extra={"path": str(path)})
            save_records(path, records)
    return records


def append_record(path: Path, record: dict) -> None:
    # A complete final JSON value may also lack its newline after an interrupted append.
    with path.open("ab+") as stream:
        stream.seek(0, 2)
        if stream.tell():
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))


@contextmanager
def chaos_settings(config: Config):
    values = {"DEVCOPILOT_CHAOS_TOOL_ERROR_RATE": config.chaos, "DEVCOPILOT_CHAOS_LLM_ERROR_RATE": config.llmchaos}
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update({key: str(value) for key, value in values.items()})
        get_settings.cache_clear()
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


def save_records(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, path)  # atomic: an interrupted save never leaves half a file


def record_config(config: Config) -> dict:
    return {
        **asdict(config),
        "name": config.name,
        "host": llm_host(),
        # Which decision model answered for the Jev components, if any.
        "jev_model": get_settings().jev_model if config.jev else None,
        # Prompt-level switches that are settings, not configs: a regression gate
        # compares files that differ only in them.
        "skills": get_settings().skills,
        # Every effective setting (temperature, num_ctx, budgets...): resume checks them all.
        # No secrets live in Settings: the Telegram token is a file the code never reads.
        "settings": get_settings().model_dump(mode="json"),
    }


def stale_reason(record: dict, config: Config, task: Task) -> str | None:
    """Why a stored run is not a run of this experiment, or None. Regression (review 2026-10-26):
    resume keyed on config|task|repeat only and kept a run made with another question, host or skills."""
    if "question" not in record or "settings" not in record["config"]:
        return "recorded before experiment checks: no question or settings in the record"
    current, stored = record_config(config), record["config"]
    changed = [k for k, v in current.items() if k != "settings" and stored.get(k) != v]
    changed += [f"settings.{k}" for k, v in current["settings"].items() if stored["settings"].get(k) != v]
    if record["question"] != task.question:
        changed.append("question")
    return f"{', '.join(changed)} changed" if changed else None


async def run_one(task: Task, config: Config, model, repeat: int, timeout_s: float, models=None) -> dict:
    # Same world for every run: earlier runs may have created issues or branches.
    seed(get_settings().workspace_dir)
    approve = approve_all if task.allow_changes else reject_all
    started = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        async with open_mcp_tools() as tools:
            # jev_for is always explicit: settings.jev_for must not leak into a baseline config.
            # models is explicit too: settings.routes must not leak into a config without routes.
            run = run_question(
                task.question, config.mode, tools, approve, model=model, jev_for=config.jev, models=models or {},
                validate_reports=config.validate,
            )  # fmt: skip
            result, error = asdict(await asyncio.wait_for(run, timeout_s)), None
    except Exception as e:  # a crashed or hung run is a failed run, not a reason to stop the eval
        result, error = None, f"{type(e).__name__}: {e}"
    return {
        "id": f"{config.name}|{task.id}|{repeat}",
        "config": record_config(config),
        "task": task.id,
        # The question as asked: a resumed eval must not count a run of an older wording (run_eval).
        "question": task.question,
        "repeat": repeat,
        "started_at": started,
        "result": result,
        "error": error,
        "judges": {},
    }


async def run_eval(
    tasks: list[Task],
    configs: list[Config],
    repeat: int,
    out: Path,
    on_record: Callable[[dict], None],
    timeout_s: float = 900,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"]: r for r in load_records(out)}
    # Check every stored run of this experiment before writing anything: a mismatch found halfway
    # left the file with new runs appended to old ones (review 2026-10-26).
    for config in configs:
        with chaos_settings(config):
            for i in range(repeat):
                for task in tasks:
                    prev = done.get(f"{config.name}|{task.id}|{i}")
                    if prev and (why := stale_reason(prev, config, task)):
                        raise ValueError(f"{out} has {prev['id']} from another experiment ({why}):"
                                         " use a new --out or remove the old records")  # fmt: skip
    for config in configs:
        with chaos_settings(config):
            try:
                model = await asyncio.to_thread(get_chat_model, config.model, config.reasoning)
                models = await asyncio.to_thread(routed_models, config.routes, config.reasoning)
                # Repeats in the outer loop: after the first pass every task has one run,
                # so an eval stopped halfway is still usable.
                for i in range(repeat):
                    for task in tasks:
                        if f"{config.name}|{task.id}|{i}" in done:
                            continue
                        record = await run_one(task, config, model, i, timeout_s, models)
                        append_record(out, record)
                        on_record(record)
            finally:
                for name in {config.model, *parse_routes(config.routes).values()}:
                    await asyncio.to_thread(unload_model, name)
    return out


async def judge_file(
    path: Path, tasks: dict[str, Task], model_name: str, reasoning: bool, on_verdict: Callable[[dict, dict], None]
) -> int:
    """Add verdicts of one judge to every record that lacks a fresh one. Returns how many were judged."""
    key = judge_key(model_name, reasoning)
    if model_name == JEV:
        jev = JevDecider("judge")

        async def grade(task: Task, answer: str) -> dict:
            return await judge_answer_jev(jev, task, answer)
    else:
        model = await asyncio.to_thread(get_chat_model, model_name, reasoning)

        async def grade(task: Task, answer: str) -> dict:
            return await judge_answer(model, task, answer)

    records = load_records(path)
    judged = 0
    try:
        for record in records:
            task = tasks.get(record["task"])
            if task is None or record["error"]:
                continue
            old = record["judges"].get(key)
            if old and old["rubric"] == task.rubric_hash and old["error"] is None:
                continue
            verdict = await grade(task, record["result"]["answer"])
            record["judges"][key] = verdict
            save_records(path, records)
            judged += 1
            on_verdict(record, verdict)
    finally:
        if model_name != JEV:
            await asyncio.to_thread(unload_model, model_name)
    return judged


def new_results_path(tag: str) -> Path:
    return RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}.jsonl"
