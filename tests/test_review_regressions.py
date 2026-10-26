"""Review regressions: no live models, network, or production workspace mutations."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fakes import World, decide, fake_model, run
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_dialogs import DIALOG, _record
from test_evals import TASK, record, result

from devcopilot import llm, runner
from devcopilot.agents.team import Intent, Review
from devcopilot.config import get_settings
from devcopilot.decide import Choice, JevDecider, JevError, Noul
from devcopilot.evals import dialogs, llm_bench
from devcopilot.evals.checks import check
from devcopilot.evals.dataset import parse_task
from devcopilot.evals.report import assignment_metrics
from devcopilot.evals.run import Config, append_record, load_records
from devcopilot.gitutil import git
from devcopilot.tools import repo
from devcopilot.tools._common import ToolError


@pytest.mark.parametrize("outcome", ["error", "pending", "rejected", "unknown_tool", "success"])
def test_required_call_must_succeed_but_attempt_is_unsafe(outcome):
    # Failed/rejected writes used to satisfy completion; safety must still see attempts.
    task = parse_task({**TASK, "allow_changes": True, "calls": [{"tool": "update_issue", "args": {"issue_id": 3}}]})
    res = result(steps=[("update_issue", {"issue_id": 3})])
    res["trajectory"][0]["outcome"] = outcome
    assert check(task, res)[0]["calls"] == (outcome == "success")
    assert not check(parse_task(TASK), res)[0]["safe"]
    forbidden = parse_task({**TASK, "allow_changes": True, "forbid_calls": [{"tool": "update_issue"}]})
    assert not check(forbidden, res)[0]["safe"]


@pytest.mark.parametrize("fail", [False, True])
def test_dialog_chaos_applied_and_restored(tmp_path, monkeypatch, fail):
    # Dialog configs used to ignore both flags and inherit ambient chaos in baselines.
    monkeypatch.setenv("DEVCOPILOT_CHAOS_TOOL_ERROR_RATE", "0.7")
    monkeypatch.setenv("DEVCOPILOT_CHAOS_LLM_ERROR_RATE", "0.8")
    observed = []

    async def fake_dialog(*args):
        s = get_settings()
        observed.append((s.chaos_tool_error_rate, s.chaos_llm_error_rate))
        if fail:
            raise RuntimeError("stop")
        return []

    monkeypatch.setattr(dialogs, "run_dialog", fake_dialog)
    monkeypatch.setattr(llm, "get_chat_model", lambda *a: fake_model([]))
    monkeypatch.setattr(llm, "routed_models", lambda *a: {})
    monkeypatch.setattr(llm, "unload_model", lambda *a: None)
    configs = [Config.parse("single/test+chaos=0.2+llmchaos=0.1"), Config.parse("single/test")]
    coro = dialogs.run_dialogs([DIALOG], configs, 1, tmp_path / "out.jsonl", lambda r: None)
    if fail:
        with pytest.raises(RuntimeError):
            asyncio.run(coro)
    else:
        asyncio.run(coro)
    assert observed == ([(0.2, 0.1)] if fail else [(0.2, 0.1), (0.0, 0.0)])
    assert get_settings().chaos_tool_error_rate == 0.7
    assert get_settings().chaos_llm_error_rate == 0.8


def test_ollama_control_calls_have_bounded_timeout(monkeypatch):
    # Metadata/unload clients previously used timeout=None and could hang forever.
    seen = []

    class Client:
        def __init__(self, host, timeout):
            seen.append(timeout)

        def show(self, model):
            return SimpleNamespace(capabilities=["tools"], details=SimpleNamespace(family="test"))

        def generate(self, **kwargs):
            pass

    monkeypatch.setattr("ollama.Client", Client)
    monkeypatch.setattr(llm, "_MODEL_INFO", {})
    llm.model_info("test")
    llm.unload_model("test")
    assert seen == [get_settings().ollama_control_timeout_s] * 2
    assert 0 < seen[0] < float("inf")


def test_async_model_setup_and_cleanup_leave_loop_free(tmp_path, monkeypatch):
    # Synchronous setup/cleanup formerly ran on the loop, preventing task cancellation.
    from devcopilot.evals import run as eval_run

    threads = []

    def setup(*args):
        threads.append(threading.get_ident())
        return fake_model([])

    monkeypatch.setattr(eval_run, "get_chat_model", setup)
    monkeypatch.setattr(eval_run, "routed_models", lambda *a: {})
    monkeypatch.setattr(eval_run, "unload_model", setup)
    loop_thread = threading.get_ident()
    asyncio.run(eval_run.run_eval([], [Config("test")], 1, tmp_path / "r.jsonl", lambda r: None))
    assert len(threads) == 2 and all(t != loop_thread for t in threads)


@pytest.mark.parametrize("tail", ['{"broken":', '{"broken":"unfinished', '{"broken":tru'])
def test_resume_drops_only_incomplete_trailing_record(tmp_path, caplog, tail):
    # An interrupted final append formerly made every completed run unresumable.
    path = tmp_path / "r.jsonl"
    path.write_text('{"id":1}\n' + tail, encoding="utf-8")
    assert load_records(path) == [{"id": 1}]
    assert "incomplete trailing" in caplog.text
    append_record(path, {"id": 2})
    assert load_records(path) == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("text", ["{broken}\n", "{broken}", '{"a":oops', '{"a":\n{"id":2}\n'])
def test_resume_rejects_corruption(tmp_path, text):
    # Recovery must not silently hide non-trailing or malformed complete records.
    path = tmp_path / "r.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_records(path)
    assert path.read_text(encoding="utf-8") == text


def test_append_after_complete_record_without_newline(tmp_path):
    # A complete JSON value without its final newline must not merge with the next append.
    path = tmp_path / "r.jsonl"
    path.write_text('{"id":1}', encoding="utf-8")
    load_records(path)
    append_record(path, {"id": 2})
    assert load_records(path) == [{"id": 1}, {"id": 2}]


def test_failed_patch_rolls_back_branch_and_retry_succeeds(fresh_workspace, monkeypatch):
    # A failed commit left an empty branch that retries misreported as a completed patch.
    original = repo.git

    def fail_commit(checkout, *args, **kwargs):
        if args[0] == "commit":
            raise RuntimeError("commit failed")
        return original(checkout, *args, **kwargs)

    args = (
        "review-fix",
        "shortlink/expiry.py",
        [repo.TextEdit(old_text="datetime.now()", new_text="datetime.now(timezone.utc)")],
        "fix",
    )
    monkeypatch.setattr(repo, "git", fail_commit)
    with pytest.raises(RuntimeError, match="commit failed"):
        repo.propose_patch(*args)
    assert not git(fresh_workspace / "repo", "branch", "--list", "copilot/review-fix").strip()
    monkeypatch.setattr(repo, "git", original)
    assert "Created branch" in repo.propose_patch(*args)
    assert "already exists" in repo.propose_patch(*args)


def test_existing_incomplete_patch_is_not_success(fresh_workspace):
    # Old failed operations may already have left a branch at main.
    git(fresh_workspace / "repo", "branch", "copilot/incomplete", "main")
    with pytest.raises(ToolError, match="intended patch"):
        repo.propose_patch(
            "incomplete",
            "shortlink/expiry.py",
            [repo.TextEdit(old_text="datetime.now()", new_text="datetime.now(timezone.utc)")],
            "fix",
        )


def test_git_env_strips_ambient_secrets_and_keeps_os_essentials(monkeypatch, tmp_path):
    # repo_head and every git subprocess inherited ambient credentials.
    from devcopilot import gitutil

    seen = {}
    monkeypatch.setenv("REVIEW_SYNTHETIC_SECRET", "never-forward")
    monkeypatch.setenv("SYSTEMROOT", "C:/Windows")
    monkeypatch.setenv("USERPROFILE", "C:/Users/test")
    monkeypatch.setenv("HOME", "/test-home")

    def subprocess_run(*args, **kwargs):
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(gitutil.subprocess, "run", subprocess_run)
    git(tmp_path, "status", env=gitutil.identity("bot", "bot@test"))
    assert "REVIEW_SYNTHETIC_SECRET" not in seen
    assert seen["SYSTEMROOT"] == "C:/Windows" and seen["USERPROFILE"] == "C:/Users/test"
    assert seen["HOME"] == "/test-home" and "PATH" in seen and seen["GIT_AUTHOR_NAME"] == "bot"


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("choice", [False, True])
def test_decider_rejects_invalid_probabilities(value, choice):
    # Malformed probabilities could grant changes or silently deny instead of falling back.
    decider = JevDecider("gate")
    question = Choice("pick", {"a": "A", "b": "B"}) if choice else Noul("change?")
    response = {"choice": "a", "probabilities": {"a": 0.5, "b": value}} if choice else {"noul": value}
    with pytest.raises(JevError):
        decider._decode(question, response)


def test_revision_count_is_actual_cycles():
    # Two rejected drafts with a one-cycle budget formerly reported revisions=2.
    reject = Review(problems=["unsupported"], verdict="revise")
    model = fake_model(
        [AIMessage("draft"), AIMessage("revised")],
        Intent(reason="r", allow_changes=False),
        decide(),
        reject,
        decide(),
        reject,
    )
    res, _ = run(model)
    assert res.revisions == 1
    assert sum(v["verdict"] == "revise" for v in res.validations) == 2


def test_single_context_counts_compacted_history():
    # Follow-ups previously reported zero context even though history reached the model.
    model = fake_model([AIMessage("a" * 700), AIMessage("answer")])

    async def go():
        cp = InMemorySaver()
        await runner.run_question(
            "first question", "single", World().tools(), model=model, checkpointer=cp, thread_id="ctx"
        )
        return await runner.run_question(
            "follow up", "single", World().tools(), model=model, checkpointer=cp, thread_id="ctx"
        )

    res = asyncio.run(go())
    assert res.context_chars == len("first question") + 700


def test_autonomy_requires_zero_approval_requests():
    # Requested writes counted as autonomous even when a human had to approve them.
    task = parse_task({**TASK, "allow_changes": True})
    rs = [record("t1", True), record("t1", True, repeat=1)]
    for r, requests in zip(rs, [1, 0], strict=True):
        r["result"].update(write_requests=requests, tokens_out=1)
    assert assignment_metrics(rs, {"t1": task}, None, "pass_facts")["product"]["autonomy"] == 0.5


def test_summaries_keep_hosts_separate():
    # Identical names on two machines used to become a meaningless 50.5-second mean.
    rs = [_record("c", "owner", 1, 0), _record("c", "owner", 1, 1, seconds=100)]
    rs[1]["config"]["host"] = "remote:11434"
    out = dialogs.summarize_dialogs(rs, {DIALOG.id: DIALOG})
    assert set(out) == {"c", "c@remote:11434"}
    assert out["c"]["by_turn"][1]["seconds"] == 1
    bench = [
        {"model": "m", "host": host, "criterion": "tools", "pass": True, "seconds": sec}
        for host, sec in [("localhost:11434", 1), ("remote:11434", 100)]
    ]
    out = llm_bench.summarize(bench)
    assert out["m"]["seconds"] == 1 and out["m@remote:11434"]["seconds"] == 100


def test_llm_bench_resume_appends_after_a_record_without_newline(tmp_path, monkeypatch):
    # `eval llm` wrote with its own open("a"): after an interrupted append the next record was glued
    # to the last one and the file no longer parsed (JSONDecodeError: Extra data).
    from typer.testing import CliRunner

    from devcopilot import llm as llm_module
    from devcopilot.evals import cli
    from devcopilot.evals import run as run_module

    out = tmp_path / "bench.jsonl"
    done = {"model": "m", "item": "a", "criterion": "tools", "pass": True, "seconds": 0.1}
    out.write_text(json.dumps(done), encoding="utf-8")  # no trailing newline

    async def run_item(model, item, tools, think):
        return {"item": item.id, "criterion": "tools", "pass": True, "seconds": 0.1, "detail": ""}

    items = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    monkeypatch.setattr(llm_bench, "load_bench", lambda: (items, []))
    monkeypatch.setattr(llm_bench, "run_item", run_item)
    monkeypatch.setattr(llm_module, "unload_model", lambda model: None)
    monkeypatch.setattr(run_module, "llm_host", lambda: "localhost:11434")
    result = CliRunner().invoke(cli.app, ["llm", "--models", "m", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert [r["item"] for r in load_records(out)] == ["a", "b"]
