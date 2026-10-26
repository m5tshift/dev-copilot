"""Team graph mechanics with a scripted fake model and fake tools (no LLM, no MCP)."""

import pytest
from fakes import OK, World, call, decide, fake_model, run
from langchain_core.messages import AIMessage

from devcopilot.agents.team import UNVERIFIED_MARK, Intent, Review
from devcopilot.config import get_settings


def report_agents(updates):
    return sorted(r["agent"] for ns, u in updates if not ns for d in u.values() if d for r in d.get("reports", []))


def test_gate_blocks_write_agents_in_code():
    world = World()
    model = fake_model(
        [AIMessage("#3: links fail"), AIMessage("Final answer")],
        Intent(reason="analysis", allow_changes=False),
        # As if an injection convinced the supervisor to close an issue:
        decide(("tracker_writer", "close #2"), ("tracker_analyst", "summarize #3")),
        decide(),
        OK,
    )
    result, updates = run(model, world)

    assert result.blocked == ["tracker_writer: close #2"]
    assert report_agents(updates) == ["tracker_analyst"]
    assert world.created == [] and result.write_requests == 0
    assert result.answer == "Final answer"


def test_parallel_assignments_produce_one_report_each():
    model = fake_model(
        [AIMessage("report"), AIMessage("report"), AIMessage("answer")],
        Intent(reason="r", allow_changes=False),
        decide(("tracker_analyst", "task A"), ("ci_investigator", "task B")),
        decide(),
        OK,
    )
    result, updates = run(model)
    assert report_agents(updates) == ["ci_investigator", "tracker_analyst"]
    assert result.reports == 2


def test_round_budget_stops_the_supervisor(monkeypatch):
    monkeypatch.setenv("DEVCOPILOT_TEAM_MAX_ROUNDS", "2")
    get_settings.cache_clear()
    model = fake_model(
        [AIMessage("r1"), AIMessage("r2"), AIMessage("answer")],
        Intent(reason="r", allow_changes=False),
        decide(("tracker_analyst", "again")),
        decide(("tracker_analyst", "and again")),
        # No third decision: the budget must route to synthesize without asking.
        OK,
    )
    result, _ = run(model)
    assert result.reports == 2 and result.answer == "answer"


def test_reviewer_sends_work_back_once():
    model = fake_model(
        [AIMessage("draft 1"), AIMessage("draft 2")],
        Intent(reason="r", allow_changes=False),
        decide(),
        Review(problems=["does not answer who owns it"], verdict="revise"),
        decide(),
        OK,
    )
    result, _ = run(model)
    assert result.answer == "draft 2"


def test_invented_link_is_caught_even_if_llm_reviewer_misses_it():
    invented = "See https://github.com/shortlink/repo/tree/fix"
    model = fake_model(
        [AIMessage(invented), AIMessage(invented)],
        Intent(reason="r", allow_changes=False),
        decide(),
        OK,  # the LLM reviewer is fooled...
        decide(),
        OK,  # ...twice
    )
    result, _ = run(model)
    # ...but the deterministic check is not: the answer is delivered with a warning.
    assert UNVERIFIED_MARK in result.answer
    assert result.escalations == {"approval": 0, "unverified": 1}
    assert "Not found in any tool output: https://github.com/shortlink/repo/tree/fix" in result.answer
    # The metric still counts the link: it is in the answer, only flagged.
    assert result.ungrounded == ["https://github.com/shortlink/repo/tree/fix"]


@pytest.mark.parametrize("action, created", [("approve", [("Bug", "t:call_1")]), ("reject", [])])
def test_approval_inside_specialist_pauses_the_team(action, created):
    world = World()
    model = fake_model(
        [call("create_issue", {"title": "Bug"}), AIMessage("done"), AIMessage("answer")],
        Intent(reason="user asks to file a bug", allow_changes=True),
        decide(("tracker_writer", "create issue 'Bug'")),
        decide(),
        OK,
    )
    seen = []

    def approve(payload):
        seen.append(payload["calls"][0]["name"])
        return {c["id"]: {"action": action} for c in payload["calls"]}

    result, _ = run(model, world, approve=approve)
    assert seen == ["create_issue"] and result.write_requests == 1
    # The key is thread_id:tool_call_id; the thread id is random per run.
    assert [(title, key.split(":")[-1]) for title, key in world.created] == [(t, k.split(":")[-1]) for t, k in created]
    assert result.answer == "answer"


def test_plain_text_instead_of_structured_output_is_retried():
    model = fake_model(
        [AIMessage("answer")],
        ("Intent", "Sure! The user wants an analysis."),  # no tool call: parse fails
        Intent(reason="analysis", allow_changes=False),  # the retry succeeds
        decide(),
        OK,
    )
    result, _ = run(model)
    assert result.answer == "answer"


def test_revise_without_problems_finishes():
    """Regression: routing used the verdict and the metric used the problems, so a
    "revise" with no problems sent the team back with no feedback to act on."""
    model = fake_model([AIMessage("answer")], Intent(reason="r", allow_changes=False), decide(),
                       Review(problems=[], verdict="revise"))  # fmt: skip
    result, _ = run(model)
    assert result.answer == "answer"


def test_reviewer_does_not_see_memory():
    """Invariant 6: memory is a hint for the supervisor and the synthesizer. The reviewer
    must not take a recalled claim as support for the draft that repeats it."""
    from devcopilot.agents.team import initial_team_state, render_context

    state = {**initial_team_state("q"), "memory": "<memory>\nA: #9 is the cause\n</memory>"}
    assert "<memory>" in render_context(state)
    assert "<memory>" not in render_context(state, memory=False)


def test_failed_review_keeps_the_answer():
    model = fake_model([AIMessage("answer")], Intent(reason="r", allow_changes=False), decide(),
                       ("Review", "no json"), ("Review", "still no json"))  # fmt: skip
    result, _ = run(model)
    assert result.answer == "answer"


def test_coordination_metrics_of_a_team_run():
    """Handoff: what specialists reported and the answer lost; duplicated calls across
    specialists; rounds and revisions: the team as one flow, not a set of agents."""
    from devcopilot.runner import coordination_of
    from devcopilot.trajectory import ToolStep, Trajectory

    reports = [
        {"summary": "Run 104 fails since 329311c in shortlink/expiry.py.", "facts": []},
        {"summary": "#3 is about HTTP 500.\n\nValidator: not found in this agent's tool output: #9", "facts": []},
    ]
    t = Trajectory(
        tools=[
            ToolStep("ci_investigator", "git_log", {"limit": 5}),
            ToolStep("code_analyst", "git_log", {"limit": 5}),
            ToolStep("code_analyst", "read_file", {"path": "a.py"}),
        ],  # fmt: skip
        routing=[
            {"assignments": ["ci_investigator", "code_analyst"], "blocked": []},
            {"assignments": [], "blocked": []},
        ],  # fmt: skip
        revisions=1,  # Actual cycles, not the number of rejected drafts.
        validations=[{"stage": "answer", "verdict": "revise"}, {"stage": "answer", "verdict": "ok"}],
    )
    c = coordination_of(reports, "CI is red since 329311c, see shortlink/expiry.py.", t)
    assert c["reported_entities"] == 4  # 329311c, shortlink/expiry.py, #3, run 104; not the validator's #9
    assert set(c["lost_entities"]) == {"#3", "run 104"}
    assert c["cross_agent_duplicates"] == 1 and c["rounds"] == 2 and c["revisions"] == 1
