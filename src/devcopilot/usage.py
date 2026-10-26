"""Counts LLM calls and tokens across a whole run, including nested agents.

LangChain passes callbacks from the run config down to every nested runnable:
the team graph -> a specialist subgraph -> its chat model. So one handler sees
all LLM calls of a run. It also records the OTel GenAI metrics per call.
"""

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from devcopilot.config import get_settings
from devcopilot.telemetry import log, metrics


def agent_label(metadata: dict | None) -> str:
    """Which agent made the call, from LangGraph's metadata.

    langgraph_checkpoint_ns looks like "code_analyst:<id>|agent:<id>" inside a
    specialist and "supervisor:<id>" for a team node; its first segment is the agent.
    """
    metadata = metadata or {}
    if metadata.get("devcopilot.agent"):
        return metadata["devcopilot.agent"]
    if metadata.get("devcopilot.mode") == "single":
        return "single"
    ns = metadata.get("langgraph_checkpoint_ns", "")
    return ns.split(":")[0] if ns else metadata.get("langgraph_node", "unknown")


class UsageTracker(BaseCallbackHandler):
    def __init__(self) -> None:
        self.llm_calls_by_agent: dict[str, int] = {}
        self.tokens_in_by_agent: dict[str, int] = {}
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        # Server-side timings from Ollama (response_metadata), summed over the run.
        self.ttft_s: list[float] = []
        self.gen_tokens = 0
        self.gen_seconds = 0.0
        self._started: dict[UUID, tuple[float, dict[str, str]]] = {}

    def on_chat_model_start(
        self, serialized: dict, messages: list, *, run_id: UUID, metadata: dict | None = None, **kwargs: Any
    ) -> None:
        # LangChain puts the model name into run metadata as ls_model_name.
        model = (metadata or {}).get("ls_model_name") or (kwargs.get("invocation_params") or {}).get("model", "unknown")
        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "devcopilot.agent": agent_label(metadata),
        }
        self._started[run_id] = (time.perf_counter(), attrs)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self.llm_calls += 1
        started, attrs = self._started.pop(run_id, (None, {}))
        agent = attrs.get("devcopilot.agent", "unknown")
        self.llm_calls_by_agent[agent] = self.llm_calls_by_agent.get(agent, 0) + 1
        if started is not None:
            metrics.llm_duration.record(time.perf_counter() - started, attrs)
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                self.tokens_in += tokens_in
                self.tokens_in_by_agent[agent] = self.tokens_in_by_agent.get(agent, 0) + tokens_in
                # Ollama silently drops the start of a prompt longer than num_ctx: the
                # agent forgets the task or earlier tool results and nothing fails.
                num_ctx = get_settings().num_ctx
                if tokens_in:
                    metrics.context_utilization.record(tokens_in / num_ctx, attrs)
                if tokens_in >= 0.9 * num_ctx:
                    log.warning(
                        "prompt near context limit",
                        extra={"agent": attrs.get("devcopilot.agent"), "tokens_in": tokens_in, "num_ctx": num_ctx},
                    )
                self.tokens_out += tokens_out
                self._record_timings(getattr(getattr(gen, "message", None), "response_metadata", None) or {}, attrs)
                metrics.llm_tokens.record(tokens_in, {**attrs, "gen_ai.token.type": "input"})
                metrics.llm_tokens.record(tokens_out, {**attrs, "gen_ai.token.type": "output"})

    def _record_timings(self, meta: dict, attrs: dict) -> None:
        """Ollama reports durations in ns: load + prompt processing stands in for the time to the
        first token (TTFT) - an estimate, the first token itself is not observed without streaming;
        eval_duration / eval_count is the mean time per output token (TPOT)."""
        if "eval_count" not in meta:
            return
        # Fields can be present but null (seen with structured output on Ollama 0.33): `or 0`.
        ttft = ((meta.get("load_duration") or 0) + (meta.get("prompt_eval_duration") or 0)) / 1e9
        self.ttft_s.append(ttft)
        metrics.ttft.record(ttft, attrs)
        count, seconds = meta["eval_count"] or 0, (meta.get("eval_duration") or 0) / 1e9
        if count and seconds:
            self.gen_tokens += count
            self.gen_seconds += seconds
            metrics.tpot.record(seconds / count, attrs)

    @property
    def tokens_per_s(self) -> float:
        return self.gen_tokens / self.gen_seconds if self.gen_seconds else 0.0

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        started, attrs = self._started.pop(run_id, (None, {}))
        if started is not None:
            metrics.llm_duration.record(time.perf_counter() - started, {**attrs, "error.type": type(error).__name__})
