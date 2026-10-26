"""Running git isolated from the user's own git configuration."""

import os
import subprocess
from pathlib import Path

from devcopilot.tools._common import minimal_env


class GitError(Exception):
    pass


def git(repo: Path, *args: str, env: dict[str, str] | None = None, timeout: float = 30) -> str:
    base_env = {
        **minimal_env(),
        **{k: os.environ[k] for k in ("HOME",) if k in os.environ},
        # Isolate from the user's git config (hooks, signing, default branch...).
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        env={**base_env, **(env or {})},
        capture_output=True,
        encoding="utf-8",  # git speaks UTF-8; the Windows default code page would garble it
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise GitError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def identity(name: str, email: str, date: str | None = None) -> dict[str, str]:
    """Environment variables that set both author and committer."""
    env = {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }
    if date:
        env |= {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return env
