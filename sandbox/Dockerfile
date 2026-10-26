# Image of the whole multi-agent system: CLI (agents), the three MCP servers, the
# simulated world (git repo, tracker, CI) and the code execution of run_tests.
# The model is NOT inside: Ollama runs on the host and is reached over the network.
#
# The image alone is a plain container (shared kernel). It is meant to run under a
# stronger runtime: gVisor (runsc) or a microVM sandbox, see sandbox/README.md.
FROM python:3.12-slim

# git: the repo tools and propose_patch (worktrees). No compilers, no curl, no sudo.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# Unprivileged user; the project is read-only for it, only /data is writable.
RUN useradd --create-home --uid 10001 agent && mkdir /data && chown agent /data
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY persona ./persona
COPY skills ./skills
COPY evals ./evals
RUN uv sync --frozen --no-dev && chown -R root:root /app && chmod -R a-w /app

USER agent
ENV PATH=/app/.venv/bin:$PATH \
    DEVCOPILOT_WORKSPACE_DIR=/data/workspace \
    DEVCOPILOT_MEMORY_DIR=/data/memory \
    DEVCOPILOT_RESULTS_DIR=/data/results \
    GIT_CONFIG_GLOBAL=/data/.gitconfig \
    HOME=/data
ENTRYPOINT ["devcopilot"]
CMD ["--help"]
