"""Persona files, skills and model routing: no LLM needed."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from devcopilot.agents import prompts
from devcopilot.agents.single import compact, count_tool_calls
from devcopilot.agents.team import SPECIALISTS
from devcopilot.config import get_settings
from devcopilot.evals.run import Config
from devcopilot.llm import parse_routes, route, same_family
from devcopilot.persona import LOAD_SKILL, all_skills, persona, skill_tool, skills_index
from devcopilot.policy import requires_approval


def test_persona_files_have_no_notes_or_titles():
    for name in ("identity", "soul", "user", "agents"):
        text = persona(name)
        assert text.strip() and "<!--" not in text and not text.startswith("#")


def test_agents_md_lists_every_specialist():
    roster = persona("agents")
    assert all(name in roster for name in SPECIALISTS)


def test_prompts_are_built_from_persona():
    assert persona("identity").strip() in prompts.SINGLE_AGENT
    assert "untrusted DATA" in prompts.SINGLE_AGENT and "untrusted DATA" in prompts.WORK_RULES
    assert prompts.TEAM_RULES.startswith("- Each task must be self-contained")
    assert "Answer in the language" in prompts.SYNTHESIZER


def test_skills_parse_and_belong_to_existing_agents():
    skills = all_skills()
    assert len(skills) >= 4
    for s in skills.values():
        assert s.description and s.body and s.agents
        assert set(s.agents) <= {"single", *SPECIALISTS}, s.name


def test_skill_tool_is_per_agent_read_only_and_trusted():
    tool = skill_tool("tracker_analyst")
    assert tool is not None and tool.name == LOAD_SKILL and not requires_approval(tool)
    assert "issue-triage" in tool.invoke({"name": "issue-triage"}) or "Issue triage" in tool.invoke(
        {"name": "issue-triage"}
    )
    assert "unknown skill" in tool.invoke({"name": "patch-workflow"})  # the fixer's skill
    assert "issue-triage" in skills_index("tracker_analyst") and "patch-workflow" not in skills_index("tracker_analyst")


def test_skills_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("DEVCOPILOT_SKILLS", "false")
    get_settings.cache_clear()
    assert skill_tool("single") is None and skills_index("single") == ""


def test_skill_loads_do_not_count_as_tool_calls():
    messages = [
        AIMessage("", tool_calls=[{"name": LOAD_SKILL, "args": {"name": "x"}, "id": "1"}]),
        AIMessage("", tool_calls=[{"name": "git_log", "args": {}, "id": "2"}]),
    ]
    assert count_tool_calls(messages) == 1


def test_compaction_cuts_only_earlier_turns():
    long = '<tool_output tool="git_log">\n' + "x" * 2000 + "\n</tool_output>"
    messages = [
        HumanMessage("first"),
        ToolMessage(long, tool_call_id="1"),
        HumanMessage("follow-up"),
        ToolMessage(long, tool_call_id="2"),
    ]
    out = compact(messages)
    assert len(out[1].text) < 600 and out[1].text.endswith("</tool_output>")
    assert out[3].text == long and messages[1].text == long  # state itself untouched


def test_routes():
    assert parse_routes("gate=qwen3.5:4b, validator=gemma3:12b") == {"gate": "qwen3.5:4b", "validator": "gemma3:12b"}
    assert route("validator", "qwen3.5:9b", "validator=gemma3:12b") == "gemma3:12b"
    assert route("supervisor", "qwen3.5:9b", "validator=gemma3:12b") == "qwen3.5:9b"
    with pytest.raises(ValueError, match="bad route"):
        parse_routes("judge=x")


def test_family_by_name_when_ollama_is_unknown(monkeypatch):
    monkeypatch.setattr("devcopilot.llm.model_info", lambda m: {})
    assert same_family("qwen3.5:9b", "qwen3:8b") and not same_family("qwen3.5:9b", "gemma3:12b")


def test_config_with_routes():
    c = Config.parse("team/qwen3.5:9b+validator=gemma3:12b+gate=qwen3.5:4b")
    assert c.routes == "gate=qwen3.5:4b,validator=gemma3:12b"
    assert c.name == "team/qwen3.5:9b+gate=qwen3.5:4b+validator=gemma3:12b"
    assert Config.parse(c.name) == c
    with pytest.raises(ValueError):
        Config.parse("single/qwen3.5:9b+validator=gemma3:12b")


def test_validator_runs_on_its_own_model():
    """The reviewer's structured call goes to the validator model, not the generator."""
    from fakes import OK, World, decide, fake_model

    from devcopilot.agents.team import Intent
    from devcopilot.runner import reject_all, run_question

    generator = fake_model([AIMessage("draft")], Intent(reason="r", allow_changes=False), decide())
    validator = fake_model([], OK)

    async def go():
        return await run_question("q", "team", World().tools(), reject_all, model=generator,
                                  models={"validator": validator})  # fmt: skip

    r = asyncio.run(go())
    assert r.answer == "draft" and r.validations == [{"stage": "answer", "verdict": "ok"}]


def test_validator_sees_every_tool_output_not_the_tail():
    """Regression: the tail of all output hid early git_log/read_file results, and the
    validator rejected 100% of reports for lack of evidence it was not shown."""
    from devcopilot.agents.team import evidence_for_validator

    calls = [{"name": "git_log", "args": {}, "id": "1"}, {"name": "get_ci_log", "args": {"run_id": 104}, "id": "2"}]
    messages = [
        AIMessage("", tool_calls=calls),
        ToolMessage("329311c remove timezone\n" + "x" * 30_000, tool_call_id="1"),
        ToolMessage("y" * 30_000 + "\nTypeError at expiry.py:13", tool_call_id="2"),
    ]
    evidence = evidence_for_validator(messages, budget=8_000)
    assert "329311c" in evidence and "## git_log({})" in evidence and "## get_ci_log({'run_id': 104})" in evidence
    assert len(evidence) < 8_200


def test_reports_are_grounded_deterministically():
    """Every report is checked without an LLM: an entity the specialist never saw in its
    tool output is flagged before the supervisor reads the report."""
    from fakes import World, decide, fake_model

    from devcopilot.agents.team import Intent
    from devcopilot.runner import reject_all, run_question

    # The specialist claims commit abc1234, which no tool showed it.
    intent = Intent(reason="r", allow_changes=False)
    model = fake_model([AIMessage("Caused by commit abc1234."), AIMessage("answer")],
                       intent, decide(("tracker_analyst", "look")), decide())  # fmt: skip
    validator = fake_model([], OK_REVIEW)

    async def go():
        return await run_question("q", "team", World().tools(), reject_all, model=model,
                                  models={"validator": validator})  # fmt: skip

    r = asyncio.run(go())
    assert {"stage": "report", "verdict": "revise"} in r.validations


from devcopilot.agents.team import Review  # noqa: E402

OK_REVIEW = Review(problems=[], verdict="ok")


def test_failed_model_lookup_is_not_cached(monkeypatch):
    """Regression: one failure while Ollama was down was cached for the whole eval, and
    every later capability check answered "unknown"."""
    import ollama

    from devcopilot import llm

    calls = []

    class Client:
        def __init__(self, host, timeout):
            pass

        def show(self, model):
            calls.append(model)
            raise ConnectionError("down")

    monkeypatch.setattr(ollama, "Client", Client)
    llm._MODEL_INFO.pop("nope:1b", None)
    assert llm.model_info("nope:1b") == {} and llm.model_info("nope:1b") == {}
    assert calls == ["nope:1b", "nope:1b"]
