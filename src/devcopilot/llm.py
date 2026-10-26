"""Chat model factory and model routing. The only place that knows which LLM provider is used.

Model routing: each role of the team can run on its own model (settings.routes):
- cheap one-bit decisions (gate) on a small model: lower latency, same accuracy (phase 4
  measured 100% on the gate for qwen3.5:4b and winnow:e4b);
- delegation and investigation on the main 9B: the 4B routed only 75% of tasks right;
- the validator on a model of ANOTHER family than the generator: a model tends to accept
  its own mistakes (self-preference bias), a different family has different blind spots.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

import httpx
import ollama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from pydantic import BaseModel

from devcopilot.config import get_settings
from devcopilot.telemetry import log, metrics

# Roles of the team that can be routed to their own model.
ROLES = ("gate", "supervisor", "specialist", "synthesizer", "validator")


def parse_routes(text: str) -> dict[str, str]:
    """'gate=qwen3.5:4b,validator=gemma4:e4b' -> {role: model}."""
    routes = {}
    for item in filter(None, (x.strip() for x in text.split(","))):
        role, sep, model = item.partition("=")
        if not sep or role not in ROLES or not model:
            raise ValueError(f"bad route {item!r}: expected <role>=<model>, roles: {', '.join(ROLES)}")
        routes[role] = model
    return routes


def route(role: str, default: str | None = None, routes: str | None = None) -> str:
    """Model name for a role: its route, else the default (the main model)."""
    s = get_settings()
    return parse_routes(s.routes if routes is None else routes).get(role) or default or s.model


_MODEL_INFO: dict[str, dict] = {}


def model_info(model: str) -> dict:
    """Family and capabilities from Ollama (/api/show), e.g. {"family": "qwen35",
    "capabilities": ["completion", "tools", "thinking"]}. Empty if Ollama is unreachable.

    Only answers are cached: a failure cached for the life of a long eval would turn
    every later lookup into "capability unknown" after one moment of Ollama being down."""
    if model in _MODEL_INFO:
        return _MODEL_INFO[model]
    import ollama

    try:
        info = ollama.Client(host=get_settings().ollama_base_url, timeout=get_settings().ollama_control_timeout_s).show(
            model
        )
    except Exception as e:
        log.warning("model info unavailable", extra={"model": model, "error": str(e)})
        return {}
    # No capabilities in the answer (older Ollama) = unknown, not "none".
    caps = list(info.capabilities) if info.capabilities else None
    _MODEL_INFO[model] = {"family": info.details.family if info.details else None, "capabilities": caps}
    return _MODEL_INFO[model]


def family(model: str) -> str:
    """Model family: from Ollama if known, else the name before ':' and digits (qwen3.5:9b -> qwen)."""
    known = model_info(model).get("family")
    return known or model.split(":")[0].rstrip("0123456789.").rstrip("-_")


def same_family(a: str, b: str) -> bool:
    # Ollama names versions of a family differently (qwen3, qwen35): compare the stem.
    def stem(x: str) -> str:
        return x.rstrip("0123456789.").lower()

    return stem(family(a)) == stem(family(b))


def has_capability(model: str, capability: str) -> bool:
    caps = model_info(model).get("capabilities")
    return True if caps is None else capability in caps  # unknown: assume yes


def get_chat_model(model: str | None = None, reasoning: bool | None = None) -> BaseChatModel:
    """The configured model; evals override model and reasoning per configuration."""
    s = get_settings()
    model = model or s.model
    reasoning = s.reasoning if reasoning is None else reasoning
    if not has_capability(model, "thinking"):
        reasoning = None  # Gemma, Llama: Ollama rejects the think flag for them
    return ChatOllama(
        model=model,
        base_url=s.ollama_base_url,
        temperature=s.temperature,
        num_ctx=s.num_ctx,
        reasoning=reasoning,
        # Keep the model loaded between calls of one investigation.
        keep_alive="10m",
    )


def routed_models(routes: str | None = None, reasoning: bool | None = None) -> dict[str, BaseChatModel]:
    """Chat models of the routed roles; roles without a route are not in the dict.
    reasoning: for every role but the validator (None = settings.reasoning), so an eval
    config with +think thinks on its routed roles too; the validator follows
    settings.validator_reasoning."""
    s = get_settings()
    routes = s.routes if routes is None else routes
    return {
        role: get_chat_model(name, reasoning=s.validator_reasoning if role == "validator" else reasoning)
        for role, name in parse_routes(routes).items()
    }


def unload_model(model: str) -> None:
    """Free the model's memory now instead of after keep_alive.

    With 16 GB two models loaded at once push the machine into swap, so evals
    unload one model before switching to the next (agent -> judge, config -> config).
    """
    import ollama

    try:
        ollama.Client(host=get_settings().ollama_base_url, timeout=get_settings().ollama_control_timeout_s).generate(
            model=model, prompt="", keep_alive=0
        )
    except Exception as e:  # not loaded, Ollama down: nothing to free
        log.warning("unload failed", extra={"model": model, "error": str(e)})


class StructuredOutputError(Exception):
    pass


def structured[T: BaseModel](
    model: BaseChatModel, schema: type[T], retries: int = 1
) -> Callable[[list[BaseMessage]], Awaitable[T]]:
    """Ask the model for an instance of `schema`, retrying once with the error shown.

    Providers differ: Ollama's constrained decoding ("json_schema") works for qwen3:8b
    but not for qwen3.5:9b (HTTP 501). "function_calling" works for any model with
    tool support, but nothing forces the model to call the tool, so it may answer in
    plain text. Hence include_raw=True and a retry instead of an exception.
    """

    async def call(messages: list[BaseMessage]) -> T:
        method = get_settings().structured_method
        name = getattr(model, "model", None)
        if method == "function_calling" and isinstance(name, str):
            if not await asyncio.to_thread(has_capability, name, "tools"):
                method = "json_schema"
        runnable = model.with_structured_output(schema, method=method, include_raw=True)
        for _ in range(retries + 1):
            llm_chaos()
            out = await runnable.ainvoke(messages)
            if out["parsed"] is not None:
                return out["parsed"]
            error = out["parsing_error"] or "you answered in plain text"
            metrics.structured_retries.add(1)
            log.warning("structured output retry", extra={"schema": schema.__name__, "error": str(error)[:300]})
            messages = [
                *messages,
                HumanMessage(f"Your reply was not valid ({error}). Reply again in exactly the required format."),
            ]
        raise StructuredOutputError(f"{schema.__name__}: no valid output after {retries + 1} attempts")

    return call


class TransientLLMError(Exception):
    """Injected transport-like failure for offline tests and chaos evals."""


def is_transient(error: BaseException) -> bool:
    """Only transport failures and retryable provider statuses may replay a node."""
    if isinstance(
        error, (TransientLLMError, httpx.TimeoutException, httpx.NetworkError, ConnectionError, TimeoutError)
    ):
        return True
    if isinstance(error, (ollama.ResponseError, httpx.HTTPStatusError)):
        status = error.status_code if isinstance(error, ollama.ResponseError) else error.response.status_code
        return status == 429 or 500 <= status < 600
    return False


def node_retry_policy() -> RetryPolicy:
    s = get_settings()
    return RetryPolicy(
        max_attempts=s.node_retry_attempts,
        initial_interval=s.node_retry_backoff_s,
        jitter=False,
        retry_on=is_transient,
    )


def llm_chaos() -> None:
    # Inject before invocation so both bound tools and structured runnables are covered.
    if random.random() < get_settings().chaos_llm_error_rate:
        raise TransientLLMError("injected LLM failure")


recovery_counts: ContextVar[dict[str, int] | None] = ContextVar("recovery_counts", default=None)


def retry_observed(fn, *, with_config: bool = False):
    """Count actual retry executions, not the predicate's final exhausted failure."""

    async def node(state, config: RunnableConfig, runtime: Runtime):
        info = runtime.execution_info
        if info is not None and info.node_attempt > 1:
            counts = recovery_counts.get()
            if counts is not None:
                counts["node_retries"] += 1
            metrics.recoveries.add(1, {"kind": "retry"})
        return await fn(state, config) if with_config else await fn(state)

    return node
