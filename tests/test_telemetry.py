"""Telemetry is code too: a metric that silently stops being recorded makes its alert
useless. These tests run fake agents against in-memory OTel providers."""

import asyncio
from collections import defaultdict

import pytest
from fakes import OK, FakeTeamModel, World, call, decide, fake_model, run
from langchain_core.messages import AIMessage
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from devcopilot.agents.team import Intent


@pytest.fixture(scope="module")
def otel():
    # Global providers can be set once per process; the proxies in devcopilot.telemetry
    # (tracer, instruments) start delegating to them from now on.
    spans = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    trace.set_tracer_provider(tracer_provider)
    reader = InMemoryMetricReader()
    otel_metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    return spans, reader


def snapshot(reader) -> dict[tuple, float]:
    """{(metric name, sorted attributes): value} for counters, count for histograms."""
    values: dict[tuple, float] = defaultdict(float)
    data = reader.get_metrics_data()
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for point in metric.data.data_points:
                    key = (metric.name, tuple(sorted(point.attributes.items())))
                    values[key] = getattr(point, "value", None) or getattr(point, "count", 0)
    return values


def delta(before: dict, after: dict, name: str, **attrs) -> float:
    """How much a metric grew, summed over series that contain the given attributes."""
    wanted = set(attrs.items())
    return sum(value - before.get(key, 0) for key, value in after.items() if key[0] == name and wanted <= set(key[1]))


def test_blocked_assignment_is_counted_and_run_is_traced(otel):
    spans, reader = otel
    before = snapshot(reader)
    model = fake_model(
        [AIMessage("#3: links fail"), AIMessage("Final answer")],
        Intent(reason="analysis", allow_changes=False),
        decide(("tracker_writer", "close #2"), ("tracker_analyst", "summarize #3")),
        decide(),
        OK,
    )
    run(model, question="triage please")
    after = snapshot(reader)

    assert delta(before, after, "devcopilot.gate.blocked") == 1
    assert delta(before, after, "devcopilot.runs", mode="team", outcome="ok") == 1
    assert delta(before, after, "devcopilot.run.duration", mode="team") == 1
    root = next(s for s in spans.get_finished_spans() if s.name == "devcopilot.run")
    assert root.attributes["input.value"] == "triage please"
    assert root.attributes["output.value"] == "Final answer"
    assert root.attributes["session.id"]


def test_approved_write_records_tool_and_approval_metrics(otel):
    _, reader = otel
    before = snapshot(reader)
    model = fake_model(
        [call("create_issue", {"title": "See https://github.com/acme/x"}), AIMessage("done"), AIMessage("answer")],
        Intent(reason="file a bug", allow_changes=True),
        decide(("tracker_writer", "create issue")),
        decide(),
        OK,
    )
    run(model, World(), approve=lambda p: {c["id"]: {"action": "approve"} for c in p["calls"]})
    after = snapshot(reader)

    assert delta(before, after, "devcopilot.escalations", kind="approval") == 1
    assert delta(before, after, "devcopilot.approvals", tool="create_issue", decision="approve") == 1
    assert delta(before, after, "devcopilot.tool.calls", tool="create_issue", status="success") == 1
    # The URL was never seen in tool output: counted once, not again on resume.
    assert delta(before, after, "devcopilot.grounding.ungrounded", where="approval") == 1


def test_rejected_write_is_not_an_error(otel):
    _, reader = otel
    before = snapshot(reader)
    model = fake_model(
        [call("create_issue", {"title": "Bug"}), AIMessage("ok"), AIMessage("answer")],
        Intent(reason="file a bug", allow_changes=True),
        decide(("tracker_writer", "create issue")),
        decide(),
        OK,
    )
    run(model, World())  # reject_all by default
    after = snapshot(reader)

    assert delta(before, after, "devcopilot.tool.calls", tool="create_issue", status="rejected") == 1
    assert delta(before, after, "devcopilot.tool.calls", status="error") == 0


def test_usage_tolerates_null_ollama_timings():
    """Regression: Ollama sent load_duration=None and the callback raised TypeError."""
    from devcopilot.usage import UsageTracker

    u = UsageTracker()
    u._record_timings({"eval_count": 5, "load_duration": None, "prompt_eval_duration": 1e9, "eval_duration": None}, {})
    assert u.ttft_s == [1.0] and u.gen_tokens == 0


def test_usage_by_agent():
    from uuid import uuid4

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from devcopilot.usage import UsageTracker

    tracker = UsageTracker()
    for agent, tokens in [("supervisor", 12), ("code_analyst", 30), ("supervisor", 8)]:
        run_id = uuid4()
        tracker.on_chat_model_start({}, [], run_id=run_id, metadata={"langgraph_node": agent})
        message = AIMessage(
            "ok", usage_metadata={"input_tokens": tokens, "output_tokens": 2, "total_tokens": tokens + 2}
        )
        tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
    assert tracker.llm_calls_by_agent == {"supervisor": 2, "code_analyst": 1}
    assert tracker.tokens_in_by_agent == {"supervisor": 20, "code_analyst": 30}
    assert sum(tracker.llm_calls_by_agent.values()) == tracker.llm_calls


def test_explicit_validator_label_overrides_specialist_namespace():
    from devcopilot.usage import agent_label

    assert agent_label({"devcopilot.agent": "validator", "langgraph_checkpoint_ns": "code_analyst:123"}) == "validator"


def test_unverified_escalation_is_counted_once(otel):
    from devcopilot.agents.team import Review

    _, reader = otel
    before = snapshot(reader)
    invented = "See https://github.com/shortlink/repo/tree/fix"
    model = fake_model(
        [AIMessage(invented), AIMessage(invented)],
        Intent(reason="analysis", allow_changes=False),
        decide(),
        Review(problems=["unsupported"], verdict="revise"),
        decide(),
        Review(problems=["unsupported"], verdict="revise"),
    )
    result, _ = run(model)
    assert result.escalations == {"approval": 0, "unverified": 1}
    after = snapshot(reader)
    assert delta(before, after, "devcopilot.escalations", kind="unverified") == 1
    assert delta(before, after, "devcopilot.escalations", kind="approval") == 0


def test_escalation_zero_series(otel):
    from devcopilot.telemetry import metrics

    _, reader = otel
    metrics.init_zero()
    values = snapshot(reader)
    for kind in ("approval", "unverified"):
        assert ("devcopilot.escalations", (("kind", kind),)) in values


def test_recovery_zero_series(otel):
    from devcopilot.telemetry import metrics

    _, reader = otel
    metrics.init_zero()
    values = snapshot(reader)
    for kind in ("resume", "retry"):
        assert ("devcopilot.recoveries", (("kind", kind),)) in values


def test_report_verdict_metric_is_the_combined_one(otel):
    """Regression: the report verdict was recorded after the deterministic grounding check,
    before the LLM validator ran, so a report the validator sent back was counted as "ok"
    and ValidatorRejectsOften undercounted."""
    from devcopilot.agents.team import ReportCheck
    from devcopilot.runner import run_question

    _, reader = otel
    before = snapshot(reader)
    model = fake_model(
        [AIMessage("#3: links fail"), AIMessage("Final answer")],
        Intent(reason="analysis", allow_changes=False),
        decide(("tracker_analyst", "summarize #3")),
        ReportCheck(unsupported=["links fail"], verdict="revise"),
        decide(),
        OK,
    )
    asyncio.run(run_question("q", "team", World().tools(), model=model, validate_reports=True))
    after = snapshot(reader)
    assert delta(before, after, "devcopilot.validator.verdicts", stage="report", verdict="revise") == 1
    assert delta(before, after, "devcopilot.validator.verdicts", stage="report", verdict="ok") == 0


def test_timed_out_run_counts_as_failed(otel):
    """Regression: eval timeouts cancel the run with CancelledError, which `except Exception`
    let through: the run was neither ok nor error, and RunsFailing never saw timeouts."""
    from devcopilot.runner import run_question

    class Hanging(FakeTeamModel):
        async def ainvoke(self, input, config=None, **kwargs):
            await asyncio.sleep(60)

    _, reader = otel
    before = snapshot(reader)

    async def go():
        await asyncio.wait_for(run_question("q", "single", [], model=Hanging(messages=iter([]))), 0.5)

    with pytest.raises(TimeoutError):
        asyncio.run(go())
    assert delta(before, snapshot(reader), "devcopilot.runs", mode="single", outcome="error") == 1
