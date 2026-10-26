"""Traces, metrics and logs via OpenTelemetry.

Off by default: without DEVCOPILOT_OTEL_ENDPOINT the OTel API stays a no-op, so
code can always call `tracer`, `metrics` and `log` (tests, offline runs).

    process (CLI or MCP server)            observability/ (docker compose)
    ---------------------------            -------------------------------
    spans   --\\                            /--> Phoenix     traces, trajectories
    metrics ---+-- OTLP/HTTP --> Collector -+--> Prometheus  metrics --> Alertmanager
    logs    --/                            \\--> Loki        logs     \\-> Grafana

Two details matter for short-lived processes such as a CLI run:
- metrics use DELTA temporality: each process reports only its own increments and
  the collector sums them across processes (deltatocumulative). Cumulative counters
  from many short processes would look like constant resets to Prometheus.
- shutdown() must flush on exit, or the last spans and metrics are lost.
"""

import logging
import os
from contextlib import contextmanager

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace

from devcopilot.config import get_settings

PROJECT = "devcopilot"  # Phoenix groups traces by this project name

tracer = trace.get_tracer("devcopilot")
log = logging.getLogger("devcopilot")
_providers: list = []


def setup(service_name: str) -> bool:
    """Configure exporters for this process. Returns False if telemetry is off."""
    endpoint = get_settings().otel_endpoint
    if not endpoint or _providers:
        return bool(_providers)

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.mcp import MCPInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
    from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # No service.instance.id: series from different runs must merge in the collector.
    resource = Resource.create(
        {"service.name": service_name, "service.namespace": "devcopilot", ResourceAttributes.PROJECT_NAME: PROJECT}
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")))
    trace.set_tracer_provider(tracer_provider)

    delta = {Counter: AggregationTemporality.DELTA, Histogram: AggregationTemporality.DELTA}
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics", preferred_temporality=delta),
        export_interval_millis=10_000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader], views=metric_views())
    otel_metrics.set_meter_provider(meter_provider)
    from devcopilot import resources

    # Software metrics: CPU, memory and disk of the process and of its machine (the sandbox).
    resources.register(meter_provider.get_meter("devcopilot.resources"), ollama=service_name == "devcopilot-cli")

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{endpoint}/v1/logs")))
    set_logger_provider(logger_provider)
    # Log records emitted inside a span automatically carry its trace_id and span_id.
    log.addHandler(LoggingHandler(level=logging.INFO, logger_provider=logger_provider))
    log.setLevel(logging.INFO)

    # Auto-instrumentation: a span for every LangGraph node, LLM call and tool call,
    # and trace context carried over MCP into the server processes.
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    MCPInstrumentor().instrument(tracer_provider=tracer_provider)

    _providers.extend([tracer_provider, meter_provider, logger_provider])
    return True


def shutdown() -> None:
    """Flush everything. Call before a short-lived process exits."""
    for provider in _providers:
        provider.shutdown()
    _providers.clear()


def server_env() -> dict[str, str]:
    """Environment for MCP server processes so they export to the same collector."""
    s = get_settings()
    env = {}
    if s.otel_endpoint:
        env["DEVCOPILOT_OTEL_ENDPOINT"] = s.otel_endpoint
    if s.chaos_tool_error_rate:
        env["DEVCOPILOT_CHAOS_TOOL_ERROR_RATE"] = str(s.chaos_tool_error_rate)
    return env


def metric_views():
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View

    def buckets(name: str, bounds: list[float]) -> View:
        return View(instrument_name=name, aggregation=ExplicitBucketHistogramAggregation(bounds))

    return [
        buckets("gen_ai.client.operation.duration", [0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300]),
        buckets("gen_ai.client.token.usage", [100, 300, 1000, 2000, 4000, 8000, 16000, 32000]),
        buckets("devcopilot.run.duration", [5, 10, 30, 60, 120, 300, 600, 1200]),
        buckets("devcopilot.tool.duration", [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30]),
        buckets("devcopilot.agent.steps", [1, 2, 3, 4, 6, 8, 10, 12, 16]),
        buckets("devcopilot.decider.duration", [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60]),
        buckets("gen_ai.server.time_to_first_token", [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60]),
        buckets("gen_ai.server.time_per_output_token", [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]),
        buckets("gen_ai.client.context.utilization", [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
        buckets("devcopilot.team.rounds", [1, 2, 3, 4, 5, 6]),
        buckets("devcopilot.memory.context.size", [0, 250, 500, 1000, 1500, 2000, 3000, 5000]),
    ]


class Metrics:
    """All instruments in one place. Names follow the OTel GenAI semantic
    conventions where they exist (gen_ai.*), the rest are ours (devcopilot.*)."""

    def __init__(self) -> None:
        # The global meter is a proxy: instruments created before setup() start
        # recording once a real MeterProvider is set, and are no-ops until then.
        m = otel_metrics.get_meter("devcopilot")
        self.llm_duration = m.create_histogram("gen_ai.client.operation.duration", unit="s")
        self.llm_tokens = m.create_histogram("gen_ai.client.token.usage", unit="{token}")
        self.runs = m.create_counter("devcopilot.runs", unit="{run}")
        self.run_duration = m.create_histogram("devcopilot.run.duration", unit="s")
        self.tool_calls = m.create_counter("devcopilot.tool.calls", unit="{call}")
        self.tool_duration = m.create_histogram("devcopilot.tool.duration", unit="s")
        self.agent_steps = m.create_histogram("devcopilot.agent.steps", unit="{step}")
        self.budget_exhausted = m.create_counter("devcopilot.agent.budget_exhausted", unit="{event}")
        self.recoveries = m.create_counter("devcopilot.recoveries", unit="{event}")
        self.escalations = m.create_counter("devcopilot.escalations", unit="{escalation}")
        self.approvals = m.create_counter("devcopilot.approvals", unit="{decision}")
        self.gate_blocked = m.create_counter("devcopilot.gate.blocked", unit="{assignment}")
        self.ungrounded = m.create_counter("devcopilot.grounding.ungrounded", unit="{entity}")
        self.structured_retries = m.create_counter("devcopilot.structured.retries", unit="{retry}")
        self.decider_calls = m.create_counter("devcopilot.decider.calls", unit="{call}")
        self.decider_duration = m.create_histogram("devcopilot.decider.duration", unit="s")
        self.decider_fallbacks = m.create_counter("devcopilot.decider.fallbacks", unit="{decision}")
        # Server-side latency split as reported by Ollama: time to the first output token
        # (model load + prompt processing) and per output token (generation speed).
        self.ttft = m.create_histogram("gen_ai.server.time_to_first_token", unit="s")
        self.tpot = m.create_histogram("gen_ai.server.time_per_output_token", unit="s")
        # Validator (another model family) on reports and on the final answer.
        self.validator_verdicts = m.create_counter("devcopilot.validator.verdicts", unit="{verdict}")
        # Continuous evaluation of real runs (evals/online.py): one count per check verdict.
        self.online_checks = m.create_counter("devcopilot.online_eval.checks", unit="{check}")
        # LLM: share of the context window a prompt takes (Ollama cuts the start beyond 1.0).
        self.context_utilization = m.create_histogram("gen_ai.client.context.utilization", unit="1")
        # Agents as one flow (runner.coordination_of) and the memory they read.
        self.team_rounds = m.create_histogram("devcopilot.team.rounds", unit="{round}")
        self.team_revisions = m.create_counter("devcopilot.team.revisions", unit="{revision}")
        self.handoff_entities = m.create_counter("devcopilot.team.handoff.entities", unit="{entity}")
        self.duplicate_calls = m.create_counter("devcopilot.team.duplicate_calls", unit="{call}")
        self.memory_context = m.create_histogram("devcopilot.memory.context.size", unit="{char}")

    def init_zero(self) -> None:
        """Report 0 for alerting counters at startup.

        Prometheus' increase() needs a sample *before* the first event: a series that
        is born with value 1 shows no increase, and an alert on it never fires. This
        only works for label sets known in advance, which is why rare-event counters
        have few labels: details (which agent, which tool) go to logs and traces.
        """
        for kind in ("resume", "retry"):
            self.recoveries.add(0, {"kind": kind})
        for kind in ("approval", "unverified"):
            self.escalations.add(0, {"kind": kind})
        self.budget_exhausted.add(0)
        self.gate_blocked.add(0)
        self.structured_retries.add(0)
        for component in ("gate", "stop", "review"):
            self.decider_fallbacks.add(0, {"component": component})
        for stage in ("report", "answer"):
            for verdict in ("ok", "revise"):
                self.validator_verdicts.add(0, {"stage": stage, "verdict": verdict})
        for check in ("grounded", "language", "answers_question", "admits_gaps"):
            for result in ("pass", "fail"):
                self.online_checks.add(0, {"check": check, "result": result})
        for where in ("answer", "approval"):
            self.ungrounded.add(0, {"where": where})
        for mode in ("single", "team"):
            for outcome in ("ok", "error"):
                self.runs.add(0, {"mode": mode, "outcome": outcome})


metrics = Metrics()


@contextmanager
def span(name: str, kind: str = "CHAIN", **attributes):
    """A span with an OpenInference kind, so Phoenix renders it properly."""
    with tracer.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        for key, value in attributes.items():
            if value is not None:
                s.set_attribute(key, value)
        yield s


def chaos_failure_rate() -> float:
    return float(os.environ.get("DEVCOPILOT_CHAOS_TOOL_ERROR_RATE", "0") or 0)
