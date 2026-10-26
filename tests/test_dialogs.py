"""Multi-turn dialog evals (evals/dialogs.py): dataset, world mutations, threads, resume, summary."""

import asyncio
import json
import math
import re
from contextlib import asynccontextmanager

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import Field

from devcopilot.evals.checks import language
from devcopilot.evals.dialogs import (
    DIALOGS,
    Dialog,
    DialogRun,
    Turn,
    _pending,
    apply_mutation,
    load_dialogs,
    parse_dialog,
    record_id,
    run_dialog,
    summarize_dialogs,
)
from devcopilot.evals.run import Config
from devcopilot.gitutil import git
from devcopilot.tools import ci, tracker
from devcopilot.world.seed import seed

# ------------------------------------------------------------------ the dataset


def test_dialogs_load_and_are_consistent():
    dialogs = load_dialogs()
    assert len(dialogs) >= 5
    assert all(3 <= len(d.turns) <= 4 for d in dialogs)
    assert any(language(d.turns[0].question, min_letters=1) == "en" for d in dialogs)
    # Staleness needs world changes; most dialogs have one.
    assert sum(any(t.before for t in d.turns) for d in dialogs) >= 3
    assert all(d.probes for d in dialogs)


def test_strings_with_hash_are_quoted():
    # `question: Что с задачей #3?` loads as "Что с задачей": the dataset lost 23 lines to this.
    for line in DIALOGS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*-?\s*(question|probes|body):\s*(.*)", line)
        if m and " #" in m.group(2):
            assert m.group(2)[0] in "\"'", line


def test_truncated_question_is_rejected():
    raw = {"id": "x", "turns": [{"question": "Что с задачей", "facts": ["#3"]}, {"question": "А?", "facts": ["a"]}]}
    with pytest.raises(ValueError, match="truncated"):
        parse_dialog(raw)


def test_unknown_mutation_and_turn_without_facts_are_rejected():
    first = {"question": "Q?", "facts": ["a"]}
    with pytest.raises(ValueError, match="mutation"):
        parse_dialog({"id": "x", "turns": [first, {"question": "Q?", "facts": ["a"], "before": [{"drop_db": {}}]}]})
    with pytest.raises(ValueError, match="no facts"):
        parse_dialog({"id": "x", "turns": [first, {"question": "Q?"}]})


# ------------------------------------------------------------------ world mutations


def test_every_mutation_of_the_dataset_applies_to_a_fresh_world(fresh_workspace):
    for dialog in load_dialogs():
        refs = dict(seed(fresh_workspace))
        for turn in dialog.turns:
            for m in turn.before:
                assert apply_mutation(m, refs)


def test_mutations_change_what_the_tools_see(fresh_workspace):
    refs = dict(seed(fresh_workspace))
    ci_dialog = next(d for d in load_dialogs() if d.id == "ci_regression")
    for m in ci_dialog.turns[3].before:
        apply_mutation(m, refs)
    apply_mutation({"update_issue": {"issue_id": 3, "assignee": "bob"}}, refs)
    apply_mutation({"comment": {"issue_id": 3, "author": "bob", "body": "Taking it."}}, refs)

    issue = tracker.get_issue(3)
    assert "assignee: bob" in issue and "- bob (" in issue
    runs = ci.list_ci_runs()
    assert runs.startswith(f"run 106: PASSED commit {refs['head']}")
    assert "status: passed" in ci.get_ci_log(106)
    repo = fresh_workspace / "repo"
    assert "datetime.now(timezone.utc)" in (repo / "shortlink" / "expiry.py").read_text(encoding="utf-8")
    assert git(repo, "log", "-1", "--format=%an %s").strip() == "bob fix: timezone-aware now in is_expired"
    # A fixed date: the same mutation gives the same commit on every machine.
    again = dict(seed(fresh_workspace))
    apply_mutation(ci_dialog.turns[3].before[0], again)
    assert again["head"] == refs["head"]


# ------------------------------------------------------------------ a dialog through the runner


class RecordingModel(GenericFakeChatModel):
    """Scripted replies; remembers the messages of every call."""

    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def get_issue_tool():
    """The real tracker read: a mutation between turns must be visible to the agent."""

    def get_issue(issue_id: int) -> str:
        """Get an issue."""
        return tracker.get_issue(issue_id)

    tool = StructuredTool.from_function(get_issue)
    tool.metadata = {"mcp_server": "tracker", "readOnlyHint": True}
    return tool


@asynccontextmanager
async def fake_tools():
    yield [get_issue_tool()]


def call(issue_id: int, call_id: str) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "get_issue", "args": {"issue_id": issue_id}, "id": call_id}])


DIALOG = Dialog(
    "owner",
    "assignee changes",
    (
        Turn("Who works on issue #3?", facts=(("#3",), ("unassigned",))),
        Turn("What did alice suspect?", facts=(("rate limit",),)),
        Turn(
            "And now?",
            facts=(("bob",),),
            forbid=("still unassigned",),
            before=({"update_issue": {"issue_id": 3, "assignee": "bob"}},),
        ),
    ),
)


def scripted_model() -> RecordingModel:
    return RecordingModel(
        messages=iter(
            [
                call(3, "c1"),
                AIMessage("Issue #3 is unassigned."),
                AIMessage("Alice suspected the new rate limiter."),
                call(3, "c2"),
                AIMessage("Issue #3 is now assigned to bob."),
            ]
        )
    )


def test_dialog_turns_share_one_thread_and_see_the_mutation(fresh_workspace):
    model = scripted_model()
    run = DialogRun(model, open_tools=fake_tools)
    records = asyncio.run(run_dialog(DIALOG, Config.parse("single/fake"), 0, run))

    assert [r["id"] for r in records] == [record_id("single/fake", "owner", t, 0) for t in (1, 2, 3)]
    assert all(r["error"] is None and all(r["checks"].values()) for r in records), [r["details"] for r in records]
    assert len({r["thread_id"] for r in records}) == 1
    # Turn 2 is answered from the thread: its model call sees turn 1's question and answer.
    turn2 = model.seen[2]
    texts = [m.text for m in turn2]
    assert "Who works on issue #3?" in texts and "Issue #3 is unassigned." in texts
    assert isinstance(turn2[-1], HumanMessage) and turn2[-1].text == "What did alice suspect?"
    # Tool calls of each turn, not of the whole thread: the thread has two by turn 3.
    assert [r["result"]["tool_calls"] for r in records] == [1, 0, 1]
    # The mutation happened before turn 3 and the agent's tool saw it.
    assert [r["after_change"] for r in records] == [False, False, True]
    assert records[2]["mutations"] == ["Updated issue #3: assignee: 'unassigned' -> 'bob'"]
    assert "assignee: bob" in model.seen[4][-1].text


def test_stale_answer_fails_forbid(fresh_workspace):
    model = scripted_model()
    model.messages = iter(
        [
            call(3, "c1"),
            AIMessage("Issue #3 is unassigned."),
            AIMessage("The rate limiter."),
            AIMessage("It is still unassigned, bob."),
        ]  # fmt: skip
    )
    records = asyncio.run(run_dialog(DIALOG, Config.parse("single/fake"), 0, DialogRun(model, open_tools=fake_tools)))
    assert records[2]["checks"]["forbid"] is False
    assert records[2]["details"]["forbidden"] == ["still unassigned"]


def test_failed_turn_skips_the_rest(fresh_workspace):
    close_99 = {"update_issue": {"issue_id": 99, "status": "closed"}}
    broken = Dialog(
        "broken", "", (DIALOG.turns[0], Turn("And #99?", facts=(("x",),), before=(close_99,)), DIALOG.turns[1])
    )
    model = scripted_model()
    records = asyncio.run(run_dialog(broken, Config.parse("single/fake"), 0, DialogRun(model, open_tools=fake_tools)))
    assert records[0]["error"] is None
    assert "does not exist" in records[1]["error"]
    assert records[2]["error"].startswith("skipped") and records[2]["skipped"]


def test_memory_is_opened_once_per_dialog(fresh_workspace):
    class FakeMemory:
        def __init__(self):
            self.recorded = []

        async def context(self, question, head):
            return ""

        async def record(self, key, question, answer, *rest):
            self.recorded.append(question)

    opened = []

    @asynccontextmanager
    async def open_memory(directory):
        opened.append(directory)
        yield (mem := FakeMemory())
        assert len(mem.recorded) == 3

    run = DialogRun(scripted_model(), memory=True, open_tools=fake_tools, open_memory=open_memory)
    records = asyncio.run(run_dialog(DIALOG, Config.parse("single/fake"), 0, run))
    assert len(opened) == 1
    assert {r["config"]["name"] for r in records} == {"single/fake+memory"}
    assert records[0]["id"].startswith("single/fake+memory|owner|1|")


# ------------------------------------------------------------------ resume and summary


def _record(name, dialog, turn, repeat, answer="", error=None, tokens=100, after_change=False, **result):
    return {
        "id": record_id(name, dialog, turn, repeat), "config": {"name": name}, "dialog": dialog, "turn": turn,
        "repeat": repeat, "question": DIALOG.turns[turn - 1].question, "after_change": after_change, "error": error,
        "result": None if error else {"answer": answer, "ungrounded": [], "trajectory": [], "tokens_in": tokens,
                                      "tool_calls": 1, "seconds": 1.0, **result},
        "checks": {},
    }  # fmt: skip


def test_partly_written_dialog_is_dropped_and_rerun(tmp_path):
    out = tmp_path / "dialogs.jsonl"
    rows = [_record("single/m", "owner", t, 0) for t in (1, 2, 3)] + [_record("single/m", "owner", 1, 1)]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    assert not _pending(out, "single/m", DIALOG, 0)  # complete: skipped
    assert _pending(out, "single/m", DIALOG, 1)  # turn 1 only: its thread is gone, run again
    ids = [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert ids == [record_id("single/m", "owner", t, 0) for t in (1, 2, 3)]
    assert _pending(out, "single/fake", DIALOG, 0)  # another config


def test_summary_by_turn_and_after_change():
    good = ["Issue #3 is unassigned.", "Alice blamed the rate limiter.", "Now bob."]
    records = [
        # repeat 0: everything right
        *(_record("c", "owner", t, 0, good[t - 1], tokens=100 * t, after_change=t == 3) for t in (1, 2, 3)),
        # repeat 1: turn 2 misses its fact, turn 3 repeats the stale answer
        _record("c", "owner", 1, 1, good[0], tokens=100, context_chars=50),
        _record("c", "owner", 2, 1, "No idea.", tokens=200, context_chars=150),
        _record("c", "owner", 3, 1, "Still unassigned, ask bob.", tokens=600, after_change=True),
        # repeat 2: turn 1 crashed, the rest was skipped: not counted as degradation
        _record("c", "owner", 1, 2, error="TimeoutError: "),
        {**_record("c", "owner", 2, 2, error="skipped: an earlier turn failed"), "skipped": True},
    ]
    s = summarize_dialogs(records, {"owner": DIALOG})["c"]

    assert s["by_turn"][1]["pass"] == pytest.approx(2 / 3)  # the crash counts as a failure
    assert s["by_turn"][1]["errors"] == 1
    assert s["by_turn"][2]["pass"] == 0.5 and s["by_turn"][3]["pass"] == 0.5
    assert s["by_turn"][3]["forbid"] == 0.5
    assert s["after_change"] == {**s["after_change"], "runs": 2, "pass": 0.5}
    assert s["no_change"]["runs"] == 2
    assert s["by_turn"][3]["tokens_in"] == 450  # (300 + 600) / 2: context grows with the dialog
    assert s["by_turn"][2]["context_chars"] == 150  # only records that report it
    assert math.isnan(s["by_turn"][3]["context_chars"])
    assert s["degradation"] == pytest.approx(2 / 3 - 0.5)
    assert s["skipped"] == 1
    assert s["by_dialog"]["owner"] == pytest.approx(4 / 7)
