#!/bin/bash
# Any devcopilot command inside the sandbox with the sandbox's environment (the same as
# evals.sh): models, telemetry and the decision model over the allowlisted ports.
#   sbx exec devcopilot bash /c/src/devcopilot/sandbox/run.sh eval dialogs --config ...
set -u
export UV_PROJECT_ENVIRONMENT=$HOME/.venv UV_LINK_MODE=copy UV_OFFLINE=1
export DEVCOPILOT_OLLAMA_BASE_URL=http://host.docker.internal:11434
export DEVCOPILOT_OTEL_ENDPOINT=http://host.docker.internal:4318
export DEVCOPILOT_JEV_BASE_URL=http://host.docker.internal:11435
export DEVCOPILOT_WORKSPACE_DIR=$HOME/workspace
cd /c/src/devcopilot
uv run devcopilot "$@"
