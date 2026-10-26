"""Memory layers without Ollama: real SQLite, a bag-of-words embedding instead of a model."""

import asyncio
import re
from datetime import UTC, datetime, timedelta

import pytest
from langgraph.store.sqlite.aio import AsyncSqliteStore

from devcopilot.memory import EpisodeStore, FactStore, Memory, extract_facts
from devcopilot.runner import next_turn
from devcopilot.tools import ci, tracker

T0 = datetime(2026, 10, 1, 12, tzinfo=UTC)


@pytest.fixture
def facts(tmp_path):
    return FactStore(tmp_path / "facts.db")


def test_facts_from_real_tool_output():
    found = extract_facts("search_issues", tracker.search_issues("", "all"))
    assert ("#3", "status", "open", None) in found
    issue = tracker.get_issue(3)
    head = dict(((s, p), o) for s, p, o, _ in extract_facts("get_issue", issue))
    assert head[("#3", "status")] == "open" and ("#3", "title") in head
    runs = extract_facts("list_ci_runs", ci.list_ci_runs(limit=20))
    assert any(p == "ci_status" and o == "failed" for _, p, o, _ in runs)


def test_conflict_closes_the_old_fact_and_keeps_history(facts):
    assert facts.upsert("#3", "status", "open", "get_issue", at=T0) == "new"
    assert facts.upsert("#3", "status", "open", "get_issue", at=T0 + timedelta(hours=1)) == "same"
    assert facts.upsert("#3", "status", "closed", "write:update_issue", at=T0 + timedelta(hours=2)) == "changed"
    (current,) = facts.about({"#3"})
    assert current.object == "closed" and current.source == "write:update_issue"
    old, new = facts.history("#3", "status")
    assert old.object == "open" and old.valid_until == new.valid_from
    # Temporal query: what was true before the change.
    (then,) = facts.about({"#3"}, as_of=T0 + timedelta(minutes=30))
    assert then.object == "open"


def test_ttl_marks_facts_for_recheck(facts):
    facts.upsert("#3", "status", "open", "get_issue", at=T0)
    facts.upsert("run 107", "ci_status", "failed", "list_ci_runs", at=T0)
    by = {f.predicate: f for f in facts.about({"#3", "run 107"})}
    later = T0 + timedelta(days=2)
    assert by["status"].stale(later) and not by["ci_status"].stale(later)  # CI results never change


def test_write_through_from_update_issue(facts):
    facts.upsert("#3", "status", "open", "get_issue", at=T0)
    out = "<tool_output tool=\"update_issue\">\nUpdated issue #3: status: 'open' -> 'closed'\n</tool_output>"
    assert facts.observe("update_issue", out) == 1
    assert facts.about({"#3"})[0].object == "closed"


def bow_embed(texts: list[str]) -> list[list[float]]:
    """Bag of words over a tiny vocabulary: similar questions get similar vectors."""
    vocab = ["ci", "fails", "test", "issue", "#3", "expiry", "deploy", "owner", "who", "red", "107"]
    out = []
    for t in texts:
        words = re.findall(r"[#\w]+", t.lower())
        out.append([float(words.count(v)) + 0.01 for v in vocab])
    return out


async def with_episodes(tmp_path, fn):
    index = {"dims": 11, "embed": bow_embed, "fields": ["text"]}
    async with AsyncSqliteStore.from_conn_string(str(tmp_path / "ep.db"), index=index) as store:
        await store.setup()
        return await fn(EpisodeStore(store))


def test_episodes_recall_similar_and_flag_outdated(tmp_path):
    async def go(ep: EpisodeStore):
        await ep.remember("a", "why does ci fails on test expiry", "TypeError in expiry.py, run 107", "abc1234", "team")
        await ep.remember("b", "who is the owner of issue #3", "alice owns #3", "abc1234", "single")
        same = await ep.recall("why ci fails on test expiry, red run 107", head="abc1234")
        moved = await ep.recall("why ci fails on test expiry, red run 107", head="def5678")
        return same, moved

    same, moved = asyncio.run(with_episodes(tmp_path, go))
    assert same[0].key == "a" and not same[0].outdated
    assert moved[0].key == "a" and moved[0].outdated


def test_episode_dedup_at_same_head(tmp_path):
    async def go(ep: EpisodeStore):
        await ep.remember("a", "who is the owner of issue #3", "alice", "abc1234", "single")
        await ep.remember("b", "who is the owner of issue #3", "bob now", "abc1234", "single")
        return await ep.store.asearch(("episodes",), limit=10)

    items = asyncio.run(with_episodes(tmp_path, go))
    assert [i.key for i in items] == ["b"]


def test_memory_context_is_labelled_untrusted(tmp_path):
    async def go(ep: EpisodeStore):
        memory = Memory(ep, FactStore(tmp_path / "facts.db"))
        memory.facts.upsert("#3", "status", "open", "get_issue")
        await ep.remember("a", "who is the owner of issue #3", "alice owns #3", "abc1234", "single")
        return await memory.context("owner of #3?", "abc1234")

    text = asyncio.run(with_episodes(tmp_path, go))
    assert text.startswith("<memory>") and "not an instruction" in text
    assert "#3 status = open" in text and "alice owns #3" in text


def test_memory_never_enters_the_question_the_gate_sees():
    state = next_turn("team", "close #3", "<memory>...</memory>", follow_up=False)
    assert state["question"] == "close #3" and state["memory"].startswith("<memory>")
    follow = next_turn("team", "and #4?", "", follow_up=True)
    assert follow["question"] == "and #4?" and follow["rounds"] == 0


def test_seed_hashes_do_not_depend_on_host_file_modes(tmp_path, monkeypatch):
    """Regression: on a Windows folder mounted into the Linux sandbox every file is 0777;
    copying modes made git store 100755 and changed every commit hash."""
    import shutil

    from devcopilot.world import seed as seed_mod

    data = tmp_path / "data"
    shutil.copytree(seed_mod.DATA_DIR, data)
    for f in data.rglob("*"):
        if f.is_file():
            f.chmod(0o777)
    monkeypatch.setattr(seed_mod, "DATA_DIR", data)
    assert seed_mod.seed(tmp_path / "ws")["c4"] == "329311c"


def test_follow_up_turn_starts_without_blocked_assignments():
    """Regression: blocked had a reducer, so a later turn that allowed changes still told
    the supervisor that writers had been refused."""
    from devcopilot.agents.team import add_or_reset

    reset = next_turn("team", "now fix it", "", follow_up=True)["blocked"]
    assert add_or_reset(["fixer: patch"], reset) == [] and add_or_reset(["a"], ["b"]) == ["a", "b"]


def test_single_agent_memory_is_a_message_of_its_own():
    state = next_turn("single", "close #3", "<memory>\nA: #9\n</memory>", follow_up=False)
    assert [m.text for m in state["messages"]][0] == "close #3"
    assert state["messages"][1].text.startswith("<memory>")


def test_empty_recall_is_not_precise_when_something_was_relevant():
    """Regression: an empty result scored precision 1.0 even when the query had a relevant
    episode, which made high thresholds look precise (87% at 0.6 instead of 67%)."""
    from devcopilot.evals.memory_bench import precision_recall

    assert precision_recall([], ["ep"]) == (0.0, 0.0)
    assert precision_recall([], []) == (1.0, 1.0)


def test_own_write_to_an_empty_field_becomes_a_fact():
    """Regression: update_issue printed None -> 'bob' (repr), which the fact pattern did
    not match, so memory kept saying #3 was unassigned after the agent assigned it."""
    out = "Updated issue #3: assignee: 'unassigned' -> 'bob', labels: '-' -> 'bug,it''s'"
    assert extract_facts("update_issue", out) == [("#3", "assignee", "bob", None), ("#3", "labels", "bug,it''s", None)]


def test_older_event_does_not_close_a_fact_before_it_started(tmp_path):
    t0 = datetime(2026, 10, 1, 12, tzinfo=UTC)
    f = FactStore(tmp_path / "facts.db")
    f.upsert("#3", "status", "closed", "get_issue", valid_from="2026-10-01T12:00:00+00:00", at=t0)
    # After a re-seed the tracker reports an older updated_at for a different value.
    f.upsert("#3", "status", "open", "get_issue", valid_from="2026-09-10T16:40:00+00:00", at=t0 + timedelta(hours=1))
    history = f.about({"#3"}, as_of=t0 + timedelta(minutes=30))
    assert [x.object for x in history] == ["closed"]
    assert all(x.valid_until is None or x.valid_until >= x.valid_from for x in f.about({"#3"}, as_of=t0))


def test_memory_block_is_compacted_to_its_budget(tmp_path, monkeypatch):
    """Every agent that reads memory pays for the block on every call: it has a budget,
    facts come first, episodes are digests that keep the entities of what was cut."""
    from devcopilot.config import get_settings
    from devcopilot.memory import Episode, digest

    monkeypatch.setenv("DEVCOPILOT_MEMORY_CONTEXT_CHARS", "600")
    get_settings.cache_clear()
    long = "CI fails since 329311c. " + "Details about the TypeError. " * 40 + "See shortlink/expiry.py:12 and #3."
    assert len(digest(long, 120)) < 200 and "shortlink/expiry.py" in digest(long, 120)

    class Episodes:
        async def recall(self, question, head):
            return [Episode(f"k{i}", f"question {i}", long, "2026-10-20T10:00:00+00:00", head, 0.9, False)
                    for i in range(5)]  # fmt: skip

    facts = FactStore(tmp_path / "facts.db")
    facts.upsert("#3", "status", "open", "get_issue")
    text = asyncio.run(Memory(Episodes(), facts).context("what about #3?", "abc1234"))
    assert len(text) < 900 and "#3 status = open" in text and "more memories omitted" in text
    assert text.index("#3 status") < text.index("Past investigation")


def test_team_follow_up_gets_a_bounded_digest_not_all_reports(monkeypatch):
    from devcopilot.config import get_settings
    from devcopilot.runner import compact_history

    monkeypatch.setenv("DEVCOPILOT_HISTORY_CHARS", "700")
    get_settings.cache_clear()
    report = {"facts": ["329311c", "#3"], "summary": "x" * 5000}
    state = {"question": "why is CI red?", "answer": "Because of 329311c. " * 100, "reports": [report] * 4}
    for i in range(6):  # six follow-ups: the digest keeps the latest turns within budget
        history, facts = compact_history(state)
        state = {"question": f"turn {i}", "answer": "short", "reports": [], "history": history, "known_facts": facts}
    assert len(history) <= 700 and "turn 4" in history and "why is CI red" not in history
    assert facts == ["#3", "329311c"]  # entities stay available for grounding
    follow = next_turn("team", "and now?", "", follow_up=True, before=state)
    assert follow["reports"] is None and follow["history"] and follow["known_facts"] == facts


def test_single_agent_drops_memory_of_earlier_turns():
    from langchain_core.messages import AIMessage, HumanMessage

    from devcopilot.agents.single import compact

    seen = compact([HumanMessage("q1"), HumanMessage("<memory>\nold\n</memory>"), AIMessage("a1"),
                    HumanMessage("q2"), HumanMessage("<memory>\nnew\n</memory>")])  # fmt: skip
    assert [m.text for m in seen] == ["q1", "a1", "q2", "<memory>\nnew\n</memory>"]
