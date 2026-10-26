"""Checks of the base-LLM bench and the assignment metrics: no model needed."""

from devcopilot.evals.llm_bench import check_calls, check_json, check_text, load_bench
from devcopilot.evals.stats import percentile


def items():
    loaded, tools = load_bench()
    assert {t["function"]["name"] for t in tools} >= {"get_issue", "get_ci_log"}
    return {i.id: i for i in loaded}


def test_bench_is_balanced():
    from collections import Counter

    counts = Counter(i.criterion for i in items().values())
    assert set(counts) == {"reasoning", "faithfulness", "structured", "tools"} and min(counts.values()) >= 8


def test_numbers_are_checked_by_regex_not_substring():
    it = items()["r-count-failed"]
    assert check_text(it, "3")[0] and check_text(it, "**3**")[0]
    assert not check_text(it, "Commit 3d8f7b2 ...")[0]  # "3" inside a hash is not the answer


def test_json_schema_and_strictness():
    it = items()
    good = '{"id": 2, "status": "open", "assignee": "alice", "labels": ["bug", "p1"]}'
    assert check_json(it["s-issue"], f"Here it is:\n{good}")[0]  # prose tolerated
    assert not check_json(it["s-issue"], good.replace('"alice"', '"bob"'))[0]
    assert check_json(it["s-no-prose"], '```json\n{"green": [101, 102, 105]}\n```')[0]
    assert not check_json(it["s-no-prose"], 'Sure! {"green": [101, 102, 105]}')[0]


def test_tool_calls_order_free_and_lenient_types():
    it = items()
    calls = [{"name": "get_issue", "args": {"issue_id": "3"}}, {"name": "get_issue", "args": {"issue_id": 2}}]
    assert check_calls(it["t-two"], calls)[0]
    assert not check_calls(it["t-two"], calls[:1])[0]
    assert check_calls(it["t-none"], [])[0] and not check_calls(it["t-none"], calls)[0]
    assert check_calls(it["t-search"], [{"name": "search_issues", "args": {"query": "Expired links"}}])[0]


def test_percentile_nearest_rank():
    xs = list(range(1, 101))
    assert percentile(xs, 50) == 50 and percentile(xs, 95) == 95 and percentile(xs, 99) == 99
    assert percentile([7.0], 99) == 7.0
