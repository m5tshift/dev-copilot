"""Regression: transient model failures must not discard checkpointed tool writes."""

import asyncio
import uuid

import httpx
import ollama
import pytest
from fakes import OK, FakeTeamModel, World, decide, fake_model
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphInterrupt

from devcopilot.agents.team import Intent
from devcopilot.config import get_settings
from devcopilot.evals.run import Config
from devcopilot.llm import StructuredOutputError, TransientLLMError, is_transient, llm_chaos
from devcopilot.runner import run_question


class FailingModel(FakeTeamModel):
    failures: int = 1
    error: Exception = TransientLLMError("offline failure")

    async def ainvoke(self, input, config=None, **kwargs):
        if self.failures:
            self.failures -= 1
            raise self.error
        return await super().ainvoke(input, config, **kwargs)


def settings(monkeypatch, **values):
    for key, value in values.items():
        monkeypatch.setenv("DEVCOPILOT_" + key.upper(), str(value))
    get_settings.cache_clear()


@pytest.mark.parametrize("failures,resumes,retries", [(1, 0, 1), (3, 1, 2)])
def test_single_recovery(monkeypatch, failures, resumes, retries):
    settings(monkeypatch, node_retry_backoff_s=0)
    model = FailingModel(messages=iter([AIMessage("done")]), failures=failures)
    result = asyncio.run(run_question("q", "single", [], model=model))
    assert result.answer == "done"
    assert result.recoveries == {"resumes": resumes, "node_retries": retries}


def test_programming_error_is_not_retried(monkeypatch):
    settings(monkeypatch, node_retry_backoff_s=0)
    model = FailingModel(messages=iter([]), error=ValueError("bad"))
    with pytest.raises(ValueError, match="bad"):
        asyncio.run(run_question("q", "single", [], model=model))


@pytest.mark.parametrize("failures", [1, 3, 4, 9])
def test_retry_and_resume_never_repeat_an_approved_write(monkeypatch, failures):
    """Regression: the specialist node had its own retry policy, so after the model's
    retries ran out the whole specialist restarted from its brief. A real model then asks
    for the same write again with a NEW call id (a new idempotency key): the human was
    asked twice and, with enough failures, the issue was created twice. The fake below
    behaves like that model: on a fresh brief it always requests the write."""
    settings(monkeypatch, node_retry_backoff_s=0)
    base = fake_model(
        [], Intent(allow_changes=True, reason="asked"), decide(("tracker_writer", "create")), decide(), OK
    )

    class Realistic(FakeTeamModel):
        remaining: int = failures

        async def ainvoke(self, input, config=None, **kwargs):
            messages = input if isinstance(input, list) else input.to_messages()
            seen_tool = any(isinstance(m, ToolMessage) for m in messages)
            if not seen_tool and "Your task" in messages[-1].text:
                return AIMessage(
                    "", tool_calls=[{"name": "create_issue", "args": {"title": "t"}, "id": uuid.uuid4().hex}]
                )
            if seen_tool and self.remaining:
                self.remaining -= 1
                raise TransientLLMError("after write")
            if seen_tool:
                return AIMessage("created")
            return await super().ainvoke(input, config, **kwargs)

    world = World()
    model = Realistic(messages=iter([AIMessage("done")]), structured=base.structured)
    approvals = []

    def approve(payload):
        approvals.append(payload)
        return {c["id"]: {"action": "approve"} for c in payload["calls"]}

    run = run_question("create issue", "team", world.tools(), model=model, models={}, approve=approve)
    if failures > 4:  # model retries (3 attempts) and one resume (3 more) are exhausted: the run fails
        with pytest.raises(TransientLLMError):
            asyncio.run(run)
    else:
        result = asyncio.run(run)
        assert result.recoveries["node_retries"] >= 1
    assert len(world.created) == len(approvals) == 1


def test_classification():
    for error in [ValueError(), StructuredOutputError(), GraphInterrupt(), ollama.ResponseError("bad", 400)]:
        assert not is_transient(error)
    for error in [
        TransientLLMError(),
        httpx.ConnectError("down"),
        ollama.ResponseError("busy", 429),
        ollama.ResponseError("down", 503),
    ]:
        assert is_transient(error)


@pytest.mark.parametrize("rate", [0, 1])
def test_chaos(monkeypatch, rate):
    settings(monkeypatch, chaos_llm_error_rate=rate)
    if rate:
        with pytest.raises(TransientLLMError):
            llm_chaos()
    else:
        llm_chaos()


def test_llmchaos_config():
    config = Config.parse("team/test+chaos=0.2+llmchaos=0.1")
    assert config.llmchaos == 0.1
    assert Config.parse(config.name) == config
    for flag in ["-1", "1.1", "nan", "0.1+llmchaos=0.2"]:
        with pytest.raises(ValueError):
            Config.parse("single/test+llmchaos=" + flag)


def test_structured_chaos_precedes_model(monkeypatch):
    from devcopilot.llm import structured

    settings(monkeypatch, chaos_llm_error_rate=1)
    model = fake_model([], Intent(allow_changes=False, reason="read"))
    with pytest.raises(TransientLLMError):
        asyncio.run(structured(model, Intent)([]))
    assert len(model.structured["Intent"]) == 1
