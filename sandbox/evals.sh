#!/bin/bash
# Config arguments also accept +chaos=0.2; eval run isolates it per configuration.
# Agent evals inside the sandbox: the team as deployed, the model reached over the allowlist.
set -u
export UV_PROJECT_ENVIRONMENT=$HOME/.venv UV_LINK_MODE=copy UV_OFFLINE=1
export DEVCOPILOT_OLLAMA_BASE_URL=http://host.docker.internal:11434
export DEVCOPILOT_OTEL_ENDPOINT=http://host.docker.internal:4318
# The gate's decision model: ollaya on the Mac, reverse-tunnelled to the PC's :11435
# (the sandbox policy must allow localhost:11435, like the Ollama port).
export DEVCOPILOT_JEV_BASE_URL=http://host.docker.internal:11435
# The simulated world lives on the VM's own disk; only results go to the shared folder.
export DEVCOPILOT_WORKSPACE_DIR=$HOME/workspace
cd /c/src/devcopilot
# Configs and the results file can be passed in: evals.sh <out.jsonl> <config>...
# Default: the product configuration (decision model on the gate, routes as in config.py).
OUT=${1:-results/evals/pc-product.jsonl}; shift || true
if [ $# -eq 0 ]; then set -- "team/qwen3.5:9b+jev=gate+gate=qwen3.5:4b+validator=gemma4:e4b"; fi
args=(); for c in "$@"; do args+=(--config "$c"); done
uv run devcopilot eval run --split dev --repeat 1 --out "$OUT" "${args[@]}"
