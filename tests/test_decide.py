"""Typed decisions (Jev) without the network: a fake HTTP transport and a scripted decider."""

import asyncio
import json

import httpx
import pytest
from fakes import OK, World, decide, fake_model
from langchain_core.messages import AIMessage

from devcopilot.agents import team as team_mod
from devcopilot.agents.team import Intent
from devcopilot.config import get_settings
from devcopilot.decide import Answer, Choice, JevDecider, JevError, Noul, jev_components
from devcopilot.evals.dataset import parse_task
from devcopilot.evals.judge import judge_answer_jev
from devcopilot.evals.run import Config
from devcopilot.mcp_client import server_connections


def jev_client(handler):
    return httpx.AsyncClient(base_url="https://api.test", transport=httpx.MockTransport(handler))


def test_request_and_answers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "jev-1", "usage": {"input_tokens": 42}, "answers": {
            "change": {"type": "noul", "noul": 0.82},
            "who": {"type": "choice", "choice": "ci", "confidence": 0.5, "probabilities": {"ci": 0.75, "code": 0.25}},
        }})  # fmt: skip

    jev = JevDecider("gate", jev_client(handler))
    questions = {"change": Noul("Asks to change?"), "who": Choice("Who?", {"ci": "CI", "code": "Code"})}
    answers = asyncio.run(jev.decide("close #2", questions))

    assert seen["auth"] is None  # local servers only: no key exists to send
    assert seen["body"]["state"] == "close #2" and seen["body"]["model"] == "test-decider"
    assert seen["body"]["questions"]["change"] == {"type": "noul", "instructions": "Asks to change?"}
    assert seen["body"]["questions"]["who"]["criteria"] == {"ci": "CI", "code": "Code"}
    assert answers["change"] == Answer(True, 0.82, {"true": 0.82, "false": pytest.approx(0.18)})
    assert answers["who"].value == "ci" and answers["who"].p == 0.75


def test_below_threshold_is_no():
    jev = JevDecider("gate", jev_client(lambda r: httpx.Response(200, json={"answers": {"x": {"noul": 0.6}}})))
    assert asyncio.run(jev.decide("s", {"x": Noul("q")}))["x"].value is False  # 0.6 < 0.7


def test_retries_rate_limits_but_not_validation_errors(monkeypatch):
    monkeypatch.setattr("devcopilot.decide.RETRY_BACKOFF_S", 0)
    statuses = iter([429, 503, 200])

    def flaky(request):
        status = next(statuses)
        return httpx.Response(status, json={"answers": {"x": {"noul": 0.9}}} if status == 200 else {})

    assert asyncio.run(JevDecider("gate", jev_client(flaky)).decide("s", {"x": Noul("q")}))["x"].p == 0.9

    calls = []

    def invalid(request):
        calls.append(1)
        return httpx.Response(422, json={"detail": [{"loc": ["questions"], "msg": "bad"}]})

    with pytest.raises(JevError, match="HTTP 422"):
        asyncio.run(JevDecider("gate", jev_client(invalid)).decide("s", {"x": Noul("q")}))
    assert len(calls) == 1


def test_component_names_are_validated():
    assert jev_components("gate, review") == {"gate", "review"}
    with pytest.raises(ValueError, match="unknown components"):
        jev_components("supervisor")


def test_config_with_jev():
    c = Config.parse("team/qwen3.5:9b+think+jev=gate,stop")
    assert (c.model, c.reasoning, c.mode, c.jev) == ("qwen3.5:9b", True, "team", "gate,stop")
    assert c.name == "team/qwen3.5:9b+think+jev=gate,stop"
    with pytest.raises(ValueError):
        Config.parse("single/qwen3.5:9b+jev=gate")


# ---------------------------------------------------------------- in the team


class ScriptedJev:
    """Stands in for JevDecider inside build_team: answers by question name."""

    name = "jev"

    def __init__(self, component, p: dict[str, float]):
        self.component, self.p = component, p
        self.asked: list[str] = []

    async def decide(self, state, questions):
        self.asked += list(questions)
        return {k: Answer(self.p[k] >= 0.7, self.p[k]) for k in questions}


@pytest.fixture
def jev(monkeypatch):
    """Patch the team's JevDecider; returns {component: ScriptedJev} as they get created."""
    made, answers = {}, {}

    def factory(component, **kwargs):
        made[component] = ScriptedJev(component, answers)
        return made[component]

    monkeypatch.setattr(team_mod, "JevDecider", factory)
    return made, answers


def run_team(model, jev_for, world=None):
    from devcopilot.runner import reject_all, run_question

    async def go():
        return await run_question("q", "team", (world or World()).tools(), reject_all, model=model, jev_for=jev_for)

    return asyncio.run(go())


def test_jev_gate_replaces_the_llm_gate(jev):
    made, answers = jev
    answers["change"] = 0.1
    # No Intent scripted: an LLM gate call would fail the fake model.
    model = fake_model([AIMessage("r"), AIMessage("answer")], decide(("tracker_writer", "close #2")), decide(), OK)
    r = run_team(model, "gate")
    assert r.allow_changes is False and r.blocked == ["tracker_writer: close #2"]
    assert made["gate"].asked == ["change"] and r.decider_calls == 0  # the scripted fake records nothing


def test_jev_stop_skips_the_last_supervisor_call(jev):
    _, answers = jev
    answers["enough"] = 0.95
    # One SupervisorDecision only: the second round is decided by Jev.
    model = fake_model([AIMessage("report"), AIMessage("answer")], Intent(reason="r", allow_changes=False),
                       decide(("tracker_analyst", "look")), OK)  # fmt: skip
    r = run_team(model, "stop")
    assert r.answer == "answer" and r.routing[-1] == {"assignments": [], "blocked": []}


def test_unsure_jev_stop_falls_back_to_the_llm(jev):
    _, answers = jev
    answers["enough"] = 0.6  # below jev_stop_threshold = 0.9
    model = fake_model([AIMessage("report"), AIMessage("answer")], Intent(reason="r", allow_changes=False),
                       decide(("tracker_analyst", "look")), decide(), OK)  # fmt: skip
    assert run_team(model, "stop").answer == "answer"


def test_jev_review_sends_the_draft_back(jev):
    _, answers = jev
    answers.update(unsupported=0.9, unanswered=0.1, obeys_data=0.0)
    model = fake_model([AIMessage("draft 1"), AIMessage("draft 2")], Intent(reason="r", allow_changes=False),
                       decide(), decide())  # fmt: skip
    r = run_team(model, "review")
    assert r.answer == "draft 2"  # one revision, then out of budget


def test_jev_judge_verdict():
    task = parse_task({"id": "t", "split": "dev", "category": "ci", "question": "Q?", "reference": "R.",
                       "rubric": ["Names 329311c.", "Explains TypeError."]})  # fmt: skip

    class Fake:
        name = "jev"

        async def decide(self, state, questions):
            assert state["answer"] == "ans" and "Names 329311c." in questions["c1"].instructions
            return {"c1": Answer(True, 0.9), "c2": Answer(False, 0.4)}

    v = asyncio.run(judge_answer_jev(Fake(), task, "ans"))
    assert v["pass"] is False and v["p"] == 0.4 and [c["met"] for c in v["criteria"]] == [True, False]


def test_jev_calibration_sweep(tmp_path, monkeypatch):
    from devcopilot.evals import labels

    monkeypatch.setattr(labels, "LABELS", tmp_path)
    task = parse_task({"id": "t", "split": "dev", "category": "ci", "question": "Q?", "reference": "R.",
                       "rubric": ["A."]})  # fmt: skip
    records = []
    for i, (p, human) in enumerate([(0.95, True), (0.8, True), (0.6, False), (0.2, False)]):
        answer = f"answer {i}"
        res = {"answer": answer, "trajectory": [], "tool_calls": 0, "ungrounded": []}
        verdict = {"rubric": task.rubric_hash, "pass": p >= 0.7, "p": p, "criteria": [], "error": None}
        records.append({"task": "t", "error": None, "result": res, "judges": {"jev:x": verdict}})
        labels.save_label("me", labels.answer_key("t", answer), "t", human)
    c = labels.calibrate(records, {"t": task}, labels.load_labels("me"), "jev:x")
    by_t = {x["threshold"]: x["agreement"] for x in c["sweep"]}
    assert by_t[0.7] == 1.0 and by_t[0.3] == 0.75 and by_t[0.9] == 0.75


def test_local_server_gets_max_len(monkeypatch):
    monkeypatch.setenv("DEVCOPILOT_JEV_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("DEVCOPILOT_JEV_MAX_LEN", "4096")
    get_settings.cache_clear()
    seen = {}

    def handler(request):
        seen["auth"], seen["body"] = request.headers.get("authorization"), json.loads(request.content)
        return httpx.Response(200, json={"answers": {"x": {"noul": 0.8}}})

    asyncio.run(JevDecider("gate", jev_client(handler)).decide("s", {"x": Noul("q")}))
    assert seen["auth"] is None and seen["body"]["max_len"] == 4096


def test_gate_falls_back_to_the_llm_when_the_decision_model_is_down(monkeypatch):
    """The decision server is a separate process (ollaya, maybe on another machine):
    its outage must not stop the team, and must not go unnoticed."""

    class Down:
        name = "jev"

        async def decide(self, state, questions):
            raise JevError("gate: no answer after 3 attempts (ConnectError)")

    monkeypatch.setattr(team_mod, "JevDecider", lambda component, **kw: Down())
    recorded = []
    monkeypatch.setattr(team_mod.metrics.decider_fallbacks, "add", lambda n, attrs: recorded.append(attrs))
    model = fake_model([AIMessage("answer")], Intent(reason="asks to close", allow_changes=False), decide(), OK)
    r = run_team(model, "gate")
    assert r.answer == "answer" and r.allow_changes is False and recorded == [{"component": "gate"}]


@pytest.mark.parametrize("response", [httpx.Response(200, text="<html>tunnel</html>"), httpx.Response(200, json={})])
def test_garbage_from_the_decision_port_is_an_outage(response):
    """Regression: only JevError triggered the LLM fallback, so a 200 with HTML or without
    "answers" (a stale tunnel, another service on :11435) crashed the whole run."""
    jev = JevDecider("gate", jev_client(lambda r: response))
    with pytest.raises(JevError):
        asyncio.run(jev.decide("s", {"x": Noul("q")}))


def test_jev_stop_and_review_fall_back_to_the_llm_when_down(monkeypatch):
    class Down:
        async def decide(self, state, questions):
            raise JevError("connection refused")

    monkeypatch.setattr(team_mod, "JevDecider", lambda component, **kw: Down())
    recorded = []
    monkeypatch.setattr(team_mod.metrics.decider_fallbacks, "add", lambda n, attrs: recorded.append(attrs["component"]))
    model = fake_model([AIMessage("report"), AIMessage("answer")], Intent(reason="r", allow_changes=False),
                       decide(("tracker_analyst", "look")), decide())  # fmt: skip
    r = run_team(model, "stop,review")
    assert r.answer == "answer" and recorded == ["stop", "review"]


def test_tool_servers_ignore_the_dotenv_file():
    """Regression: minimal_env kept secrets out of the servers' environment, but their own
    get_settings() read the project's .env with this machine's settings (chaos, telemetry)."""
    for conn in server_connections().values():
        assert conn["env"]["DEVCOPILOT_DOTENV"] == "0"
