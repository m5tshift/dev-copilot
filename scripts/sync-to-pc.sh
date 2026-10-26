#!/bin/sh
# Copy the working tree from the Mac to the PC: code, docs, datasets.
# Not copied: .venv, .env (each machine has its own), generated data and results
# (results live where they were produced; copy them back explicitly).
# Adds and overwrites only: a file deleted or renamed on the Mac stays on the PC (old
# tests keep running there) until removed by hand.
#
#   DEVCOPILOT_PC=user@host scripts/sync-to-pc.sh
set -eu
PC="${DEVCOPILOT_PC:?set DEVCOPILOT_PC=user@host}"
DEST="${DEVCOPILOT_PC_DIR:-C:/src/devcopilot}"
cd "$(dirname "$0")/.."
COPYFILE_DISABLE=1 tar --no-xattrs \
  --exclude=.venv --exclude=.env --exclude=workspace --exclude=.memory --exclude=results \
  --exclude='__pycache__' --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.claude \
  -czf - . | ssh "$PC" "tar -xzf - -C $DEST"
echo "synced to $PC:$DEST"
