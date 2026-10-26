"""Helpers shared by all tools.

Tool output goes straight into the model's context window, so every tool:
- returns compact plain text, not raw JSON dumps;
- caps its output size (a 9B model with a 16k context drowns fast);
- raises ToolError with a message that tells the model how to recover.
"""

import os
from pathlib import Path

MAX_OUTPUT_CHARS = 6000

# Without these a child process on Windows cannot even start Python (SYSTEMROOT is
# needed for sockets and random numbers). They hold no secrets.
WINDOWS_ENV = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "PATHEXT", "COMSPEC")


def minimal_env(**extra: str) -> dict[str, str]:
    """Environment for child processes: PATH, OS essentials, UTF-8; no secrets.

    PYTHONUTF8: on Windows Python otherwise reads and writes files in the ANSI code page
    (cp1251 here) and breaks on any non-ASCII text in data or tool output.
    """
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUTF8": "1"}
    env |= {k: os.environ[k] for k in WINDOWS_ENV if k in os.environ}
    return env | extra


class ToolError(Exception):
    """An error the model can act on. The message is shown to the model verbatim."""


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars, narrow your request]"


def resolve_inside(root: Path, relative: str) -> Path:
    """Resolve a model-supplied path and make sure it does not escape `root`."""
    root = root.resolve()
    path = (root / relative.lstrip("/")).resolve()
    if not path.is_relative_to(root):
        raise ToolError(f"Path {relative!r} is outside the repository.")
    return path
