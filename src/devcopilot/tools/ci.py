"""Read-only tools over CI runs. Published by `devcopilot/servers/ci.py`."""

import json
from typing import Annotated, Literal

from pydantic import Field

from devcopilot.config import get_settings
from devcopilot.tools._common import ToolError, truncate


def _runs() -> list[dict]:
    path = get_settings().ci_dir / "runs.json"
    if not path.exists():
        raise ToolError("CI data is not initialized. Ask the user to run `devcopilot seed`.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_ci_runs(
    limit: Annotated[int, Field(description="Maximum number of runs.")] = 10,
    status: Annotated[Literal["passed", "failed", "all"], Field(description="Filter by status.")] = "all",
) -> str:
    """List the most recent CI runs on main, newest first."""
    runs = [r for r in reversed(_runs()) if status == "all" or r["status"] == status][:limit]
    if not runs:
        return f"No CI runs (status={status})."
    return "\n".join(
        f"run {r['id']}: {r['status'].upper()} commit {r['commit']} ({r['branch']}) "
        f"at {r['started_at']}, {r['duration_s']}s, {r['trigger']}"
        for r in runs
    )


def get_ci_log(
    run_id: Annotated[int, Field(description="CI run id, e.g. 104.")],
    tail: Annotated[int, Field(description="How many lines from the end to return (max 300).")] = 80,
) -> str:
    """Get the last lines of a CI run log. Failures and tracebacks are at the end."""
    log = get_settings().ci_dir / "logs" / f"{run_id}.log"
    if not log.exists():
        known = ", ".join(str(r["id"]) for r in _runs())
        raise ToolError(f"No log for run {run_id}. Known runs: {known}.")
    lines = log.read_text(encoding="utf-8").splitlines()
    tail = max(1, min(tail, 300))
    header = f"[showing last {tail} of {len(lines)} lines]\n" if len(lines) > tail else ""
    return truncate(header + "\n".join(lines[-tail:]))
