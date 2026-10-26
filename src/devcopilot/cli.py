"""Command-line interface: `devcopilot --help`."""

import asyncio
import atexit
import json
import sys
import time

import typer
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from devcopilot.config import get_settings
from devcopilot.evals.cli import app as eval_app

# Windows: a pipe (SSH, redirect) gets the ANSI code page and cannot print ✓ or Cyrillic.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(eval_app, name="eval")
console = Console()


@app.callback()
def main() -> None:
    """Dev Copilot: a learning project on multi-agent systems."""
    from devcopilot import telemetry

    if telemetry.setup("devcopilot-cli"):
        telemetry.metrics.init_zero()
        # A CLI process is short-lived: without a flush the last spans and metrics are lost.
        atexit.register(telemetry.shutdown)


@app.command()
def seed() -> None:
    """(Re)create the simulated world: git repo, issue tracker, CI runs."""
    from devcopilot.world.seed import seed as seed_world

    workspace = get_settings().workspace_dir
    hashes = seed_world(workspace)
    console.print(f"[green]Workspace ready:[/green] {workspace}")
    console.print("Commits: " + ", ".join(f"{k}={v}" for k, v in hashes.items()))


@app.command()
def tools(schemas: bool = typer.Option(False, help="Also print the schemas the model receives.")) -> None:
    """List tools of all MCP servers with their annotations and approval policy."""
    from devcopilot.mcp_client import open_mcp_tools
    from devcopilot.policy import model_facing_schema, requires_approval

    async def run() -> None:
        async with open_mcp_tools() as all_tools:
            for t in all_tools:
                m = t.metadata or {}
                policy = (
                    f"[red]{'needs approval':<15}[/red]" if requires_approval(t) else f"[green]{'auto':<15}[/green]"
                )
                hints = ", ".join(
                    f"{k.removesuffix('Hint')}={v}" for k, v in m.items() if k.endswith("Hint") and v is not None
                )
                console.print(f"{m['mcp_server']:>8}.{t.name:<14} {policy} [dim]{hints}[/dim]")
                if schemas:
                    console.print_json(json.dumps(model_facing_schema(t)))

    asyncio.run(run())


@app.command()
def graph(team: bool = typer.Option(False, help="Draw the team graph instead of the single agent.")) -> None:
    """Print the agent graph as a Mermaid diagram."""

    async def run() -> None:
        from devcopilot.agents.single import build_agent
        from devcopilot.agents.team import build_team
        from devcopilot.mcp_client import open_mcp_tools

        async with open_mcp_tools() as all_tools:
            agent = build_team(all_tools) if team else build_agent(all_tools)
            print(agent.get_graph().draw_mermaid())

    asyncio.run(run())


@app.command()
def ask(
    question: str,
    team: bool = typer.Option(False, help="Use the supervisor + specialists team instead of a single agent."),
    show_tool_output: bool = typer.Option(True, help="Print tool results (truncated)."),
    thread: str = typer.Option(None, help="Continue this conversation (needs DEVCOPILOT_MEMORY=true)."),
) -> None:
    """Ask a question. Changes require your approval."""
    from devcopilot.mcp_client import open_mcp_tools
    from devcopilot.memory import open_checkpointer, open_memory
    from devcopilot.runner import run_question

    s = get_settings()
    mode = "team" if team else "single"
    console.print(f"[dim]mode={mode} model={s.model} reasoning={s.reasoning} routes={s.routes or '-'} "
                  f"memory={s.memory}[/dim]")  # fmt: skip
    if thread and not s.memory:
        raise typer.BadParameter("--thread needs DEVCOPILOT_MEMORY=true: threads live in the SQLite checkpointer")

    async def run():
        async with open_mcp_tools() as all_tools, open_memory() as memory, open_checkpointer() as checkpointer:
            printer = LivePrinter(show_tool_output)
            return await run_question(
                question, mode, all_tools, approve=ask_for_approval, on_update=printer,
                memory=memory, checkpointer=checkpointer, thread_id=thread,
            )  # fmt: skip

    result = asyncio.run(run())
    console.print(Panel(Markdown(result.answer or "(no answer)"), title="answer", border_style="green"))
    extra = f" · reports {result.reports} · blocked {len(result.blocked)}" if team else ""
    console.print(
        f"[dim]{result.seconds:.1f}s · LLM calls {result.llm_calls} · tool calls {result.tool_calls} · "
        f"tokens in {result.tokens_in} / out {result.tokens_out}{extra}[/dim]"
    )
    if result.ungrounded:
        console.print(f"[red]⚠ mentioned but seen in no tool output: {', '.join(result.ungrounded)}[/red]")
    if s.memory:
        console.print(f"[dim]thread {result.thread_id}: continue with --thread {result.thread_id}[/dim]")


@app.command()
def bench(
    modes: str = typer.Option("single,team", help="Comma-separated: single, team."),
    cases: str = typer.Option("", help="Comma-separated case ids; empty = all."),
    repeat: int = typer.Option(1, help="Runs per case and mode."),
) -> None:
    """Run the benchmark questions through the single agent and/or the team (writes rejected)."""
    from devcopilot.bench import CASES, run_bench

    selected = [c for c in CASES if not cases or c.id in cases.split(",")]
    s = get_settings()
    console.print(f"[dim]model={s.model} reasoning={s.reasoning} · {len(selected)} cases × {repeat} runs[/dim]")
    table = Table("case", "mode", "facts", "missing", "warned", "writes", "blocked", "invented", "LLM", "tools",
                  "tok in", "time")  # fmt: skip

    def on_row(row) -> None:
        r = row.result
        warned = {None: "", True: "yes", False: "[red]no[/red]"}[row.warned]
        cells = [
            row.case, row.mode, f"{len(row.found)}/{len(row.found) + len(row.missing)}", ", ".join(row.missing),
            warned, str(r.write_requests), str(len(r.blocked)), ", ".join(r.ungrounded), str(r.llm_calls),
            str(r.tool_calls), str(r.tokens_in), f"{r.seconds:.0f}s",
        ]  # fmt: skip
        table.add_row(*cells)
        console.print(f"done: {row.case} / {row.mode} #{row.run} ({r.seconds:.0f}s)")

    out = asyncio.run(run_bench(modes.split(","), selected, repeat, on_row))
    console.print(table)
    console.print(f"[dim]saved to {out}[/dim]")


# ------------------------------------------------------------------ approvals


def ask_for_approval(payload: dict) -> dict:
    decisions = {}
    warnings = payload.get("warnings", {})
    for call in payload["calls"]:
        body = Text(f"{call['name']}\n{json.dumps(call['args'], indent=2, ensure_ascii=False)}")
        if found := warnings.get(call["id"]):
            body.append(f"\n\n⚠ not seen in any tool output: {', '.join(found)}", style="bold red")
        console.print(Panel(body, title="approval needed", border_style="red"))
        choice = Prompt.ask(r"\[a]pprove / \[r]eject / \[e]dit", choices=["a", "r", "e"], default="r")
        if choice == "a":
            decisions[call["id"]] = {"action": "approve"}
        elif choice == "r":
            reason = Prompt.ask("reason for the agent", default="not approved")
            decisions[call["id"]] = {"action": "reject", "reason": reason}
        else:
            raw = Prompt.ask("new args as JSON", default=json.dumps(call["args"], ensure_ascii=False))
            decisions[call["id"]] = {"action": "edit", "args": json.loads(raw)}
    return decisions


# ----------------------------------------------------------------- live trace


def _unwrap(text: str) -> str:
    """Strip the <tool_output> wrapper for display."""
    lines = text.splitlines()
    if lines and lines[0].startswith("<tool_output") and lines[-1] == "</tool_output>":
        return "\n".join(lines[1:-1])
    return text


class LivePrinter:
    """Prints node updates as they stream in. Specialists' steps are indented."""

    def __init__(self, show_tool_output: bool):
        self.show_tool_output = show_tool_output
        self.last = time.perf_counter()

    def __call__(self, namespace: tuple[str, ...], update: dict) -> None:
        # namespace is () for the top graph, ("code_analyst:<task id>",) inside a specialist.
        who = namespace[0].split(":")[0] if namespace else ""
        indent = "    " if namespace else ""
        for node, delta in update.items():
            if not delta:
                continue
            if "messages" in delta:
                self._agent_step(delta, who, indent)
            elif node == "gate":
                allowed = "[red]yes[/red]" if delta["allow_changes"] else "[green]no[/green]"
                console.print(
                    f"\n[bold magenta]gate[/bold magenta] changes allowed: {allowed} · {delta['last_decision']['gate']}"
                )
            elif node == "supervisor" and "rounds" in delta:
                self._supervisor(delta)
            elif "reports" in delta:
                r = delta["reports"][0]
                console.print(
                    f"  [bold green]✓ {r['agent']}[/bold green] [dim]{r['steps']} steps, {r['tool_calls']} tools[/dim]"
                )
                console.print(Panel(Text(r["summary"]), style="green", expand=False))
            elif node == "review" and "feedback" in delta:
                problems = delta["last_decision"]["review"]
                if problems == "ok":
                    console.print("\n[bold magenta]review[/bold magenta] [green]ok[/green]")
                else:
                    console.print("\n[bold magenta]review[/bold magenta] [red]problems:[/red]")
                    for p in problems:
                        console.print(Text(f"  - {p}", style="red"))
            elif node == "synthesize":
                console.print("\n[bold magenta]synthesize[/bold magenta] draft ready")
        self.last = time.perf_counter()

    def _supervisor(self, delta: dict) -> None:
        d = delta["last_decision"]
        console.print(f"\n[bold magenta]supervisor[/bold magenta] round {delta['rounds']}: {d['thoughts']}")
        for a in d["assignments"]:
            console.print(Text(f"  ⇒ {a['agent']}: {a['task']}", style="magenta"))
        for b in delta["blocked"]:
            console.print(Text(f"  ✗ blocked by gate: {b}", style="bold red"))
        if not d["assignments"]:
            console.print("  ⇒ enough information, writing the answer")

    def _agent_step(self, delta: dict, who: str, indent: str) -> None:
        label = f"[{who}] " if who else ""
        for msg in delta["messages"]:
            if isinstance(msg, AIMessage):
                usage = msg.usage_metadata or {}
                console.print(
                    f"{indent}[cyan]{label}step {delta['steps']}[/cyan] [dim]{time.perf_counter() - self.last:.1f}s, "
                    f"in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')}[/dim]"
                )
                if thinking := msg.additional_kwargs.get("reasoning_content"):
                    console.print(Panel(Text(thinking.strip()), title="thinking", style="dim"))
                for call in msg.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    console.print(Text(f"{indent}  → {call['name']}({args})", style="yellow"))
            elif isinstance(msg, ToolMessage) and self.show_tool_output:
                text = _unwrap(msg.text)
                preview = text if len(text) < 400 else text[:400] + " …"
                style = "red" if msg.status == "error" else "dim"
                # Text(): tool output may contain "[...]" that rich would parse as markup.
                console.print(Panel(Text(preview), title=f"{label}{msg.name}", style=style, expand=False))


memory_app = typer.Typer(no_args_is_help=True, help="Inspect the agent's memory (memory.py).")
app.add_typer(memory_app, name="memory")


@memory_app.command("facts")
def memory_facts(
    subject: str = typer.Argument(None, help='E.g. "#3" or "run 107"; default: all current facts.'),
    history: str = typer.Option(None, help="Show all versions of this predicate of the subject, e.g. status."),
) -> None:
    """Temporal facts: current values, or the history of one of them."""
    from devcopilot.memory import FactStore

    get_settings().memory_dir.mkdir(parents=True, exist_ok=True)  # before the first run there is none
    store = FactStore(get_settings().memory_dir / "facts.db")
    if subject and history:
        facts = store.history(subject, history)
    else:
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(store.path)) as conn:
            subjects = {subject} if subject else {r[0] for r in conn.execute("SELECT DISTINCT subject FROM facts")}
        facts = store.about(subjects)
    table = Table("subject", "predicate", "object", "valid from", "valid until", "observed", "source", "stale")
    for f in facts:
        table.add_row(f.subject, f.predicate, f.object, f.valid_from, f.valid_until or "-", f.observed_at,
                      f.source, "re-check" if f.stale() else "")  # fmt: skip
    console.print(table)


@memory_app.command("recall")
def memory_recall(question: str, k: int = typer.Option(3, help="How many episodes.")) -> None:
    """What the agent would recall for a question: episodes and facts."""
    from devcopilot.memory import open_memory, repo_head

    async def run() -> str:
        async with open_memory(get_settings().memory_dir) as memory:
            episodes = await memory.episodes.recall(question, repo_head(), k)
            for e in episodes:
                console.print(
                    f"[bold]{e.score:.2f}[/bold] {e.created_at} {'(outdated) ' if e.outdated else ''}{e.question}"
                )
            return await memory.context(question, repo_head())

    console.print(Panel(asyncio.run(run()) or "(nothing)", title="context for the agents"))


@memory_app.command("clear")
def memory_clear(yes: bool = typer.Option(False, "--yes", help="Do not ask.")) -> None:
    """Delete all memory: checkpoints, episodes, facts."""
    import shutil

    d = get_settings().memory_dir
    if not d.exists():
        console.print("nothing to clear")
        return
    if not yes and not typer.confirm(f"Delete {d}?"):
        raise typer.Abort()
    shutil.rmtree(d)
    console.print(f"deleted {d}")


if __name__ == "__main__":  # last: after every subcommand is registered
    app()
