"""Application settings, read from environment variables (prefix DEVCOPILOT_) or `.env`."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEVCOPILOT_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    # LLM (Ollama)
    model: str = "qwen3.5:9b"
    # Model routing by role (llm.py). Roles without a route use `model`. The product's
    # choice (report, section 3.2): the validator on gemma4:e4b, another family than the
    # Qwen generator; the gate's LLM (used when the decision model is unreachable) on the
    # 4B, 100% on the gate in phase 4. "" = every role on `model`.
    routes: str = "gate=qwen3.5:4b,validator=gemma4:e4b"
    # Reports are always checked deterministically (their entities against the agent's tool
    # output). This adds an LLM check of each report: one more call per report.
    validate_reports: bool = False
    # The validator thinks before its verdict. Replaying 9 real report checks on gemma4:e4b:
    # without thinking it rejected 9/9 (correct claims included), with thinking 2/9, both
    # justified. Thinking costs time, so the LLM check is on the final answer by default.
    validator_reasoning: bool = True
    ollama_base_url: str = "http://localhost:11434"
    # Bound Ollama metadata/unload HTTP calls; async callers offload them to threads.
    ollama_control_timeout_s: float = 10.0
    temperature: float = 0.0
    # Context window. Ollama allocates memory for all of it up front: for qwen3.5:9b
    # 16384 took ~16 GB, 8192 about 10 GB. A longer prompt is truncated SILENTLY
    # (the start of the conversation is dropped); usage.py logs a warning near the limit.
    num_ctx: int = 8192
    # Qwen models can "think" before answering: better tool choice, much slower.
    reasoning: bool = False
    # How to get JSON matching a schema: "function_calling" works with any tool-capable
    # model; "json_schema" (constrained decoding) is stricter, but the Homebrew build of Ollama
    # on the Mac answers it with 501 (a build problem, not the model's).
    structured_method: Literal["function_calling", "json_schema", "json_mode"] = "function_calling"

    # Agent loop
    # Bound transient node retries and checkpoint resumes without masking programming errors.
    node_retry_attempts: int = 3
    node_retry_backoff_s: float = 1.0
    run_resume_attempts: int = 1
    # Probability of a synthetic transient failure immediately before each LLM call.
    chaos_llm_error_rate: float = 0.0
    max_steps: int = 12
    # Skills (skills/*/SKILL.md, see persona.py): off = no index in prompts, no load_skill tool.
    skills: bool = True

    # Team (phase 2)
    team_max_rounds: int = 4  # supervisor decisions per run
    team_max_revisions: int = 1  # reviewer -> supervisor loops
    specialist_max_steps: int = 8  # LLM calls per specialist task

    # Observability (phase 3). OTLP/HTTP endpoint of the collector, e.g.
    # http://localhost:4318. Unset = telemetry off.
    otel_endpoint: str | None = None
    # Chaos testing: probability that an MCP tool call fails on purpose.
    chaos_tool_error_rate: float = 0.0

    # Typed decisions: Jev-like decision models (see devcopilot/decide.py) answer yes/no
    # questions with a probability instead of generating text. Comma-separated components
    # that use one instead of the LLM: gate, review, stop, judge, online.
    # Default: the gate, the one place measurements justified it (phase 4: winnow:e4b 100%
    # on 45 tasks at threshold 0.5, ~0.6 s; as judge it lost to the LLM, kappa 0.40 vs 0.72).
    jev_for: str = "gate"
    jev_model: str = "winnow:e4b"
    # A local server speaking the /v1/systemone protocol: ollaya (default) or laya-serve.
    # Everything runs locally by design: no cloud decision service, no API key.
    jev_base_url: str = "http://localhost:11435"
    # laya only: token budget per request (its default window is 1024; reviews and
    # verdicts are longer). Not sent when unset: ollaya does not define it.
    jev_max_len: int | None = None
    # The protocol's usual starting points: >= 0.7 means yes, <= 0.3 means no.
    jev_yes_threshold: float = 0.7
    # Stopping the team early skips work, so the bar is higher (team.py, supervisor).
    jev_stop_threshold: float = 0.9
    # The gate's own bar; None = jev_yes_threshold. A threshold belongs to a model: for
    # winnow:e4b 0.5 separated all 45 eval tasks, 0.7 missed 2 of 5 change requests.
    jev_gate_threshold: float | None = 0.5
    # A cold start loads the model into memory: winnow:e4b took over 30 s on the first request.
    jev_timeout_s: float = 60

    # Memory (memory.py): SQLite checkpoints, episodes with embeddings, temporal facts.
    # Off by default: evals must not remember answers between runs (eval memory measures
    # it separately, on its own store).
    memory: bool = False
    memory_dir: Path = PROJECT_ROOT / ".memory"
    embed_model: str = "qwen3-embedding:0.6b"
    episode_recall_k: int = 3
    # Below it an episode is not shown. 0.6 from `eval memory` with qwen3-embedding:0.6b:
    # precision 67%, recall 80% (0.5: 55% / 90%, 0.7: 50% / 45%).
    episode_min_similarity: float = 0.6
    episode_dedup_similarity: float = 0.95  # same question at the same HEAD: replace
    episode_half_life_days: float = 14  # recency weight halves every N days
    episode_ttl_days: int = 90  # the store deletes older episodes
    episode_answer_chars: int = 800  # stored; what agents see is a digest (below)
    # Compaction (memory.digest): every agent that reads memory gets at most this much.
    # ~1500 chars is ~400 tokens: under 5% of the 8K window, read on every LLM call of a run.
    memory_context_chars: int = 1500
    episode_digest_chars: int = 300
    # Team follow-up turns: earlier turns as a digest instead of all their reports.
    history_chars: int = 1200

    # Simulated world the agent works with (see devcopilot/world/seed.py).
    workspace_dir: Path = PROJECT_ROOT / "workspace"

    @property
    def repo_dir(self) -> Path:
        return self.workspace_dir / "repo"

    @property
    def tracker_db(self) -> Path:
        return self.workspace_dir / "tracker.db"

    @property
    def ci_dir(self) -> Path:
        return self.workspace_dir / "ci"


@lru_cache
def get_settings() -> Settings:
    # DEVCOPILOT_DOTENV=0: ignore .env. Tool servers get it (mcp_client): their settings come
    # from the clean environment we pass, not from a .env with this machine's settings;
    # tests get it too, so this machine's .env does not change their behaviour.
    if os.environ.get("DEVCOPILOT_DOTENV") == "0":
        return Settings(_env_file=None)
    return Settings()
