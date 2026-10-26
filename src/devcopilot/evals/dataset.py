"""Eval tasks over the simulated world: expected facts, tool calls and a rubric.

The tasks live in evals/dataset.yaml: data, reviewed like code. Every expectation
refers to the world built by `devcopilot seed`, whose commit hashes are deterministic.

Two splits: `dev` is what we look at while changing prompts; `holdout` is run only
to check that an improvement is real and not fitted to the dev questions.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from devcopilot.agents.team import SPECIALISTS
from devcopilot.config import PROJECT_ROOT

DATASET = PROJECT_ROOT / "evals" / "dataset.yaml"
WRITE_TOOLS = frozenset({"create_issue", "add_comment", "update_issue", "propose_patch"})
TOOLS = frozenset(t for spec in SPECIALISTS.values() for t in spec.tools)
Split = Literal["dev", "holdout"]


@dataclass(frozen=True)
class CallSpec:
    """A tool call pattern. Arg values: a string of a free-text argument (TEXT_ARGS) matches as a
    case-insensitive substring, any other string case-insensitively and exactly; a list matches if
    every item is in the actual list, anything else must be equal. A key "a|b" matches if either
    argument matches."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def matches(self, tool: str, args: dict[str, Any]) -> bool:
        return tool == self.tool and all(
            any(_arg_matches(want, args.get(k), k) for k in key.split("|")) for key, want in self.args.items()
        )

    def covers(self, tool: str, args: dict[str, Any]) -> bool:
        """Matches and changes nothing the pattern does not name: update_issue(3, assignee="bob",
        labels=["p3"]) matched "assign #3 to bob" and relabelled the issue unnoticed."""
        named = {k for key in self.args for k in key.split("|")}
        changed = {k for k in CHANGED_FIELDS.get(tool, ()) if args.get(k) is not None}
        return self.matches(tool, args) and changed <= named


# Free text: the expected value is a part of it ("329311c" in a comment, "copilot/" in a branch).
# Logins, statuses and the like are compared whole: a substring let assignee "bobby" pass for "bob".
TEXT_ARGS = frozenset({"body", "title", "path", "ref"})
# Fields a write changes on an existing object; the other write tools create something new whole.
CHANGED_FIELDS = {"update_issue": ("status", "assignee", "labels")}


def _arg_matches(want: Any, got: Any, key: str) -> bool:
    if isinstance(want, str):
        if not isinstance(got, str):
            return False
        return want.lower() in got.lower() if key in TEXT_ARGS else want.lower() == got.strip().lower()
    if isinstance(want, list):
        got_items = {str(g).lower() for g in got} if isinstance(got, list) else set()
        return all(str(w).lower() in got_items for w in want)
    if isinstance(want, int | float) and not isinstance(want, bool) and isinstance(got, str):
        # The tools accept "3" for 3 (pydantic coerces it), so the check does too.
        try:
            return float(got) == want
        except ValueError:
            return False
    return want == got


@dataclass(frozen=True)
class Task:
    id: str
    split: Split
    category: str
    question: str
    reference: str  # what an expert would answer; context for the judge
    rubric: tuple[str, ...]  # criteria the judge checks one by one
    # Deterministic expectations about the answer.
    facts: tuple[tuple[str, ...], ...] = ()  # each: alternatives, one must appear (case-insensitive)
    forbid: tuple[str, ...] = ()  # must not appear in the answer
    # Expectations about the trajectory.
    tools: tuple[tuple[str, ...], ...] = ()  # each group: at least one of these tools is called
    max_tool_calls: int = 10
    calls: tuple[CallSpec, ...] = ()  # write calls the agent must request; no other write may succeed
    forbid_calls: tuple[CallSpec, ...] = ()  # write calls it must not even request (checked if writes allowed)
    allow_changes: bool = False  # the user asks for changes: expected gate decision; writes approved
    route: tuple[str, ...] = ()  # specialists relevant to the task (team routing)
    critical: bool = False  # a regression here fails the gate regardless of the average

    @property
    def rubric_hash(self) -> str:
        """Verdicts made with another rubric or reference are stale."""
        text = json.dumps([self.question, self.reference, self.rubric], ensure_ascii=False)
        return hashlib.sha256(text.encode()).hexdigest()[:12]


def _alternatives(item: str | list[str]) -> tuple[str, ...]:
    return tuple(item) if isinstance(item, list) else (item,)


def _calls(items: list[dict]) -> tuple[CallSpec, ...]:
    return tuple(CallSpec(c["tool"], c.get("args", {})) for c in items)


def parse_task(raw: dict) -> Task:
    known = set(Task.__dataclass_fields__)
    if unknown := set(raw) - known:
        raise ValueError(f"{raw.get('id')}: unknown keys {sorted(unknown)}")
    task = Task(
        **{
            **raw,
            "rubric": tuple(raw["rubric"]),
            "facts": tuple(_alternatives(f) for f in raw.get("facts", [])),
            "forbid": tuple(raw.get("forbid", [])),
            "tools": tuple(_alternatives(t) for t in raw.get("tools", [])),
            "calls": _calls(raw.get("calls", [])),
            "forbid_calls": _calls(raw.get("forbid_calls", [])),
            "route": tuple(raw.get("route", [])),
        }
    )
    _validate(task)
    return task


def _validate(task: Task) -> None:
    named = {t for group in task.tools for t in group} | {c.tool for c in task.calls + task.forbid_calls}
    if unknown := named - TOOLS:
        raise ValueError(f"{task.id}: unknown tools {sorted(unknown)}")
    if unknown := set(task.route) - set(SPECIALISTS):
        raise ValueError(f"{task.id}: unknown specialists {sorted(unknown)}")
    if task.split not in ("dev", "holdout"):
        raise ValueError(f"{task.id}: split must be dev or holdout")
    if not task.rubric:
        raise ValueError(f"{task.id}: empty rubric")
    if task.calls and not task.allow_changes:
        raise ValueError(f"{task.id}: expects write calls but allow_changes is false")
    # In YAML " #" starts a comment: `question: Назначь задачу #3 на bob.` loads as
    # "Назначь задачу". This happened; a sentence must end like a sentence.
    for text in (task.question, task.reference, *task.rubric):
        if not text.rstrip().endswith((".", "?", "!", ")", "»", '"')):
            raise ValueError(f"{task.id}: {text!r} looks truncated: quote strings that contain ' #'")


def load_dataset(path: Path = DATASET) -> list[Task]:
    tasks = [parse_task(raw) for raw in yaml.safe_load(path.read_text(encoding="utf-8"))["tasks"]]
    ids = [t.id for t in tasks]
    if dupes := {i for i in ids if ids.count(i) > 1}:
        raise ValueError(f"duplicate task ids: {sorted(dupes)}")
    return tasks


def select(tasks: list[Task], split: str = "all", ids: str = "", categories: str = "") -> list[Task]:
    """Filter by split ("dev", "holdout", "all"), comma-separated ids and categories."""
    return [
        t
        for t in tasks
        if (split == "all" or t.split == split)
        and (not ids or t.id in ids.split(","))
        and (not categories or t.category in categories.split(","))
    ]
