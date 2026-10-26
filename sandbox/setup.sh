#!/bin/bash
# Installs the project inside the sandbox. Needs temporary network rules for this
# sandbox only (PyPI, GitHub, astral); remove them afterwards.
set -e
export UV_PROJECT_ENVIRONMENT=$HOME/.venv UV_LINK_MODE=copy
cd /c/src/devcopilot
uv python install 3.12 2>&1 | tail -1
uv sync --frozen --python 3.12 2>&1 | tail -2
uv run --python 3.12 pytest -q -p no:cacheprovider 2>&1 | tail -2
