#!/bin/sh
# Copy evals results from the PC to the Mac, into results/pc/evals/: kept apart from the
# Mac's own results/evals/, where files with the same names hold other runs.
#   DEVCOPILOT_PC=user@host scripts/fetch-results.sh [pattern]
set -eu
PC="${DEVCOPILOT_PC:?set DEVCOPILOT_PC=user@host}"
DEST="${DEVCOPILOT_PC_DIR:-C:/src/devcopilot}"
cd "$(dirname "$0")/.."
mkdir -p results/pc/evals
scp "$PC:$DEST/results/evals/${1:-*.json*}" results/pc/evals/
