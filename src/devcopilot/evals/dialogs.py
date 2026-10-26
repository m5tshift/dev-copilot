"""Multi-turn dialog evals: does quality hold on turns 2-4 of one conversation?

The tasks of dataset.yaml are single questions, so they never see the two ways a long
dialog degrades here:
- compaction: tool output of earlier turns is cut (single.compact, team digests), and a
  detail from turn 1 must be looked up again rather than half-remembered;
- staleness: the world changes between turns (`before` mutations), and an answer that
  repeats the earlier turn is wrong.

A dialog runs in one fresh world, one MCP session and one thread (InMemorySaver +
thread_id), like a user asking follow-ups in `devcopilot ask --thread`. One JSONL record
per turn, id "{config}|{dialog}|{turn}|{repeat}". A dialog is the unit of resume: the
thread lives in memory, so a dialog stopped halfway is run again from turn 1.
"""

import asyncio
import json
import sqlite3
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, closing, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver

from devcopilot.config import PROJECT_ROOT, get_settings
from devcopilot.evals.checks import language
from devcopilot.evals.dataset import WRITE_TOOLS
from devcopilot.evals.report import config_key
from devcopilot.evals.run import Config, approve_all, llm_host, load_records, save_records
from devcopilot.evals.stats import mean
from devcopilot.gitutil import git, identity
from devcopilot.runner import reject_all, run_question
from devcopilot.tools import tracker
from devcopilot.world.seed import seed

DIALOGS = PROJECT_ROOT / "evals" / "dialogs.yaml"
MUTATIONS = ("update_issue", "comment", "commit", "ci_run")
CHECKS = ("facts", "forbid", "grounded", "language", "safe")


@dataclass(frozen=True)
class Turn:
    question: str
    facts: tuple[tuple[str, ...], ...] = ()  # each: alternatives, one must appear (case-insensitive)
    forbid: tuple[str, ...] = ()  # must not appear: mostly the wording of a stale answer
    before: tuple[dict, ...] = ()  # world mutations applied right before this turn
    allow_changes: bool = False  # writes of this turn are approved (the world is reseeded per dialog)

    @property
    def after_change(self) -> bool:
        """The world changed since the previous turn: a repeated earlier answer is stale."""
        return bool(self.before)


@dataclass(frozen=True)
class Dialog:
    id: str
    probes: str  # what degradation the dialog is about, for the report
    turns: tuple[Turn, ...] = field(default_factory=tuple)


def _alternatives(item: str | list[str]) -> tuple[str, ...]:
    return tuple(item) if isinstance(item, list) else (item,)


def parse_dialog(raw: dict) -> Dialog:
    if unknown := set(raw) - {"id", "probes", "turns"}:
        raise ValueError(f"{raw.get('id')}: unknown keys {sorted(unknown)}")
    turns = []
    for i, t in enumerate(raw["turns"], 1):
        if unknown := set(t) - set(Turn.__dataclass_fields__):
            raise ValueError(f"{raw['id']} turn {i}: unknown keys {sorted(unknown)}")
        turns.append(
            Turn(
                question=t["question"],
                facts=tuple(_alternatives(f) for f in t.get("facts", [])),
                forbid=tuple(t.get("forbid", [])),
                before=tuple(t.get("before", [])),
                allow_changes=t.get("allow_changes", False),
            )
        )
    dialog = Dialog(raw["id"], raw.get("probes", ""), tuple(turns))
    _validate(dialog)
    return dialog


def _validate(dialog: Dialog) -> None:
    if len(dialog.turns) < 2:
        raise ValueError(f"{dialog.id}: a dialog needs at least two turns")
    for i, turn in enumerate(dialog.turns, 1):
        where = f"{dialog.id} turn {i}"
        # The YAML comment trap of dataset.py: `question: Что с задачей #3?` loads as "Что с задачей".
        if not turn.question.rstrip().endswith((".", "?", "!", ")", "»", '"')):
            raise ValueError(f"{where}: {turn.question!r} looks truncated: quote strings that contain ' #'")
        if not turn.facts:
            raise ValueError(f"{where}: no facts: a turn without them is never checked")
        if i == 1 and turn.before:
            raise ValueError(f"{where}: `before` on the first turn changes nothing a later answer could repeat")
        facts = {a.lower() for alts in turn.facts for a in alts}
        if clash := facts & {f.lower() for f in turn.forbid}:
            raise ValueError(f"{where}: both a fact and forbidden: {sorted(clash)}")
        for m in turn.before:
            if len(m) != 1 or next(iter(m)) not in MUTATIONS:
                raise ValueError(f"{where}: a mutation is one of {MUTATIONS}, got {m}")


def load_dialogs(path: Path = DIALOGS) -> list[Dialog]:
    dialogs = [parse_dialog(raw) for raw in yaml.safe_load(path.read_text(encoding="utf-8"))["dialogs"]]
    ids = [d.id for d in dialogs]
    if dupes := {i for i in ids if ids.count(i) > 1}:
        raise ValueError(f"duplicate dialog ids: {sorted(dupes)}")
    return dialogs


# ------------------------------------------------------------------ world mutations


def apply_mutation(mutation: dict, refs: dict[str, str]) -> str:
    """Change the seeded world the way a teammate would between two questions.

    refs: commit keys of the seed ($c1..$c4) plus $head after a `commit` mutation; updated
    in place. Returns a one-line description for the record.
    """
    (kind, args), s = next(iter(mutation.items())), get_settings()
    if kind == "update_issue":
        return tracker.update_issue(**args)
    if kind == "comment":
        # Not tracker.add_comment: that one always signs as the bot, and a human's
        # comment is the point here ("bob: taking it").
        with closing(sqlite3.connect(s.tracker_db)) as conn, conn:
            now = datetime.now(UTC).isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO comments (issue_id, author, body, created_at) VALUES (?, ?, ?, ?)",
                (args["issue_id"], args["author"], args["body"], now),
            )
            conn.execute("UPDATE issues SET updated_at = ? WHERE id = ?", (now, args["issue_id"]))
        return f"comment on #{args['issue_id']} by {args['author']}"
    if kind == "commit":
        repo = s.repo_dir
        for edit in args["edits"]:
            path = repo / edit["path"]
            text = path.read_text(encoding="utf-8")
            if text.count(edit["old"]) != 1:
                raise ValueError(f"commit edit: {edit['old']!r} must occur once in {edit['path']}")
            path.write_text(text.replace(edit["old"], edit["new"]), encoding="utf-8")
        git(repo, "add", "-A")
        # A fixed author and date, as in seed.py: the same mutation gives the same hash.
        git(repo, "commit", "-q", "-m", args["message"], env=identity(args["author"], args["email"], args["date"]))
        refs["head"] = git(repo, "rev-parse", "--short=7", "HEAD").strip()
        return f"commit {refs['head']} by {args['author']}: {args['message']}"
    if kind == "ci_run":
        commit = _ref(str(args["commit"]), refs)
        keys = ("id", "status", "trigger", "started_at", "duration_s")
        run = {**{k: args[k] for k in keys}, "commit": commit, "branch": "main"}
        runs_path = s.ci_dir / "runs.json"
        runs = json.loads(runs_path.read_text(encoding="utf-8"))
        runs_path.write_text(json.dumps([*runs, run], indent=2), encoding="utf-8")
        log = (
            f"[ci] run {run['id']} | branch main | commit {commit}\n[ci] $ python -m pytest -v\n"
            + ("15 passed\n" if run["status"] == "passed" else "tests failed\n")
            + f"[ci] status: {run['status']}\n"
        )
        (s.ci_dir / "logs" / f"{run['id']}.log").write_text(log, encoding="utf-8")
        return f"CI run {run['id']} {run['status']} on {commit}"
    raise ValueError(f"unknown mutation {kind!r}")


def _ref(value: str, refs: dict[str, str]) -> str:
    return refs[value[1:]] if value.startswith("$") else value


# ------------------------------------------------------------------ checks


def check_turn(turn: Turn, result: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    """The deterministic checks of evals/checks.py that make sense per turn."""
    answer = result["answer"].lower()
    missing = [alts[0] for alts in turn.facts if not any(a.lower() in answer for a in alts)]
    forbidden = [f for f in turn.forbid if f.lower() in answer]
    writes = [s for s in result["trajectory"] if s["tool"] in WRITE_TOOLS]
    unsafe = [] if turn.allow_changes else writes
    got_lang = language(result["answer"])
    checks = {
        "facts": not missing,
        "forbid": not forbidden,
        "grounded": not result["ungrounded"],
        # A follow-up in Russian answered in English is a degradation too.
        "language": got_lang is None or got_lang == language(turn.question, min_letters=1),
        "safe": not unsafe,
    }
    details = {"missing_facts": missing, "forbidden": forbidden, "ungrounded": result["ungrounded"],
               "unsafe_calls": [s["tool"] for s in unsafe], "language": got_lang}  # fmt: skip
    return checks, details


# ------------------------------------------------------------------ running


def variant(config: Config, memory: bool) -> str:
    """The name in records: long-term memory is a switch of this eval, not of Config."""
    return f"{config.name}{'+memory' if memory else ''}"


def record_id(name: str, dialog: str, turn: int, repeat: int) -> str:
    return f"{name}|{dialog}|{turn}|{repeat}"


@dataclass
class DialogRun:
    """What a dialog needs besides the config; tests replace the tools and the memory."""

    model: Any
    models: dict | None = None
    timeout_s: float = 900
    memory: bool = False
    # () -> async context manager yielding the tools; default: the MCP servers.
    open_tools: Callable[[], AbstractAsyncContextManager] | None = None
    # (directory) -> async context manager yielding a Memory; default: memory.open_memory.
    open_memory: Callable[[Path], AbstractAsyncContextManager] | None = None


async def run_dialog(dialog: Dialog, config: Config, repeat: int, run: DialogRun) -> list[dict]:
    """All turns of one dialog in one thread. Returns one record per turn."""
    from devcopilot.mcp_client import open_mcp_tools
    from devcopilot.memory import open_memory

    open_tools, open_mem = run.open_tools or open_mcp_tools, run.open_memory or open_memory
    name = variant(config, run.memory)
    # Same world for every dialog: earlier ones moved issues and added commits and runs.
    refs = dict(seed(get_settings().workspace_dir))
    checkpointer, thread_id = InMemorySaver(), uuid.uuid4().hex
    base = {
        "config": {**asdict(config), "name": name, "host": llm_host(), "memory": run.memory,
                   "jev_model": get_settings().jev_model if config.jev else None, "skills": get_settings().skills},
        "dialog": dialog.id, "repeat": repeat, "thread_id": thread_id,
    }  # fmt: skip
    records: list[dict] = []
    failed: str | None = None
    try:
        with tempfile.TemporaryDirectory() as mem_dir:
            # Memory of this dialog only: what turn 1 stored is what turn 3 may recall (stale facts).
            memory_cm = open_mem(Path(mem_dir)) if run.memory else nullcontext(None)
            async with open_tools() as tools, memory_cm as memory:
                for i, turn in enumerate(dialog.turns, 1):
                    thread = (checkpointer, thread_id)
                    records.append(await _turn(turn, i, config, run, tools, memory, thread, refs, failed))
                    failed = failed or records[-1]["error"]
    except Exception as e:  # tools or memory failed to open: every turn not yet run fails
        failed = f"{type(e).__name__}: {e}"
        records += [_failed(t, i, failed) for i, t in enumerate(dialog.turns, 1) if i > len(records)]
    return [{"id": record_id(name, dialog.id, r["turn"], repeat), **base, **r} for r in records]


def _failed(turn: Turn, i: int, error: str) -> dict:
    return {"turn": i, "question": turn.question, "after_change": turn.after_change, "mutations": [],
            "started_at": None, "result": None, "error": error, "checks": {}, "details": {}}  # fmt: skip


async def _turn(turn: Turn, i: int, config: Config, run: DialogRun, tools, memory, thread, refs, failed) -> dict:
    if failed:
        # The thread is broken or incomplete: later turns would answer another conversation.
        return {**_failed(turn, i, f"skipped: an earlier turn failed ({failed[:80]})"), "skipped": True}
    checkpointer, thread_id = thread
    record = _failed(turn, i, "")
    record["started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        record["mutations"] = [apply_mutation(m, refs) for m in turn.before]
        approve = approve_all if turn.allow_changes else reject_all
        # jev_for and models are explicit, as in run.py: settings must not leak into a config.
        coro = run_question(
            turn.question, config.mode, tools, approve, model=run.model, jev_for=config.jev, models=run.models or {},
            memory=memory, checkpointer=checkpointer, thread_id=thread_id,
            validate_reports=config.validate,
        )  # fmt: skip
        result = asdict(await asyncio.wait_for(coro, run.timeout_s))
    except Exception as e:  # a crashed or hung turn is a failed turn, not a reason to stop the eval
        record["error"] = f"{type(e).__name__}: {e}"
        return record
    record["result"], record["error"] = result, None
    record["checks"], record["details"] = check_turn(turn, result)
    return record


async def run_dialogs(
    dialogs: list[Dialog],
    configs: list[Config],
    repeat: int,
    out: Path,
    on_record: Callable[[dict], None],
    timeout_s: float = 900,
    memory: bool = False,
) -> Path:
    from devcopilot.llm import get_chat_model, parse_routes, routed_models, unload_model

    out.parent.mkdir(parents=True, exist_ok=True)
    from devcopilot.evals.run import append_record, chaos_settings

    for config in configs:
        with chaos_settings(config):
            name = variant(config, memory)
            run = DialogRun(await asyncio.to_thread(get_chat_model, config.model, config.reasoning),
                        await asyncio.to_thread(routed_models, config.routes, config.reasoning),
                        timeout_s, memory)  # fmt: skip
            try:
                for i in range(repeat):
                    for dialog in dialogs:
                        if not _pending(out, name, dialog, i):
                            continue
                        for record in await run_dialog(dialog, config, i, run):
                            append_record(out, record)
                            on_record(record)
            finally:
                for model in {config.model, *parse_routes(config.routes).values()}:
                    await asyncio.to_thread(unload_model, model)
    return out


def _pending(out: Path, name: str, dialog: Dialog, repeat: int) -> bool:
    """True if the dialog has to run. A partly written dialog is dropped and run again:
    its thread was in memory, a resumed turn 3 would have no turns 1-2 behind it."""
    ids = {record_id(name, dialog.id, i, repeat) for i in range(1, len(dialog.turns) + 1)}
    records = load_records(out)
    present = ids & {r["id"] for r in records}
    if present == ids:
        return False
    if present:
        save_records(out, [r for r in records if r["id"] not in ids])
    return True


# ------------------------------------------------------------------ summary


def score(record: dict, dialogs: dict[str, Dialog]) -> dict[str, bool]:
    """Checks against the current dialogs.yaml (like report.py), stored ones if the turn is gone."""
    if record["error"]:
        return {**dict.fromkeys(CHECKS, False), "pass": False}
    dialog = dialogs.get(record["dialog"])
    turn = dialog.turns[record["turn"] - 1] if dialog and record["turn"] <= len(dialog.turns) else None
    checks = check_turn(turn, record["result"])[0] if turn and turn.question == record["question"] else record["checks"]
    return {**checks, "pass": all(checks.values())}


def summarize_dialogs(records: list[dict], dialogs: dict[str, Dialog]) -> dict[str, dict]:
    """Per config: pass rate and cost by turn index, after a world change, per dialog.

    Turn 1 is the baseline: a drop on turns 2+ is the degradation this eval is about.
    Errors count as failures (as in report.py); infra errors are removed and rerun. Turns
    skipped after an earlier failure are left out: counting them would show a crash of
    turn 1 as degradation of turns 2-4.
    """
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_config[config_key(r)].append(r)
    summary = {}
    for name, rs in by_config.items():
        scored = [(r, score(r, dialogs)) for r in rs if not r.get("skipped")]

        def block(items: list[tuple[dict, dict]]) -> dict:
            ok = [r["result"] for r, _ in items if not r["error"]]
            ctx = [x["context_chars"] for x in ok if x.get("context_chars") is not None]
            return {
                "runs": len(items),
                "errors": sum(bool(r["error"]) for r, _ in items),
                "pass": mean([s["pass"] for _, s in items]),
                **{c: mean([s[c] for _, s in items]) for c in CHECKS},
                "tokens_in": mean([x["tokens_in"] for x in ok]),
                # Size of the history/memory block the agents got (runner, if it reports it).
                "context_chars": mean(ctx),
                "tool_calls": mean([x["tool_calls"] for x in ok]),
                "seconds": mean([x["seconds"] for x in ok]),
            }

        turns, dialog_ids = sorted({r["turn"] for r, _ in scored}), sorted({r["dialog"] for r in rs})
        later = [x for x in scored if x[0]["turn"] > 1]
        summary[name] = {
            "by_turn": {t: block([x for x in scored if x[0]["turn"] == t]) for t in turns},
            "later_turns": block(later),
            # Turns right after a world change: the staleness probe.
            "after_change": block([x for x in later if x[0]["after_change"]]),
            "no_change": block([x for x in later if not x[0]["after_change"]]),
            "by_dialog": {d: mean([s["pass"] for r, s in scored if r["dialog"] == d]) for d in dialog_ids},
        }
        first = summary[name]["by_turn"].get(1, {}).get("pass", float("nan"))
        summary[name]["degradation"] = first - summary[name]["later_turns"]["pass"]
        summary[name]["skipped"] = sum(bool(r.get("skipped")) for r in rs)
    return summary
