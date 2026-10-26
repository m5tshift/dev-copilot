"""`devcopilot eval ...`: run, judge, report, compare, label, calibrate."""

import asyncio
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from devcopilot.config import get_settings

app = typer.Typer(no_args_is_help=True, help="Evals: datasets, judges, reports (phase 4).")
console = Console()

FILES = typer.Argument(..., help="Results files (results/evals/*.jsonl).", exists=True, dir_okay=False)
SPLIT = typer.Option("all", help="dev, holdout or all.")


def _tasks(split: str = "all", ids: str = "", categories: str = ""):
    from devcopilot.evals.dataset import load_dataset, select

    return {t.id: t for t in select(load_dataset(), split, ids, categories)}


def _records(files: list[Path]) -> list[dict]:
    from devcopilot.evals.run import load_records

    return [r for f in files for r in load_records(f)]


def _default_judge(records: list[dict]) -> str | None:
    counts = Counter(k for r in records for k in r["judges"])
    return counts.most_common(1)[0][0] if counts else None


def _save_json(path: Path, data: dict) -> None:
    """Numbers that otherwise exist only in a console: kept for reports and docs."""
    import json
    from datetime import UTC, datetime

    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"saved_at": datetime.now(UTC).isoformat(timespec="seconds"), **data}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    console.print(f"[dim]saved {path}[/dim]")


def _pct(x: float) -> str:
    return "—" if x != x else f"{x:.0%}"  # x != x: NaN


@app.command()
def tasks(split: str = SPLIT) -> None:
    """List the dataset: ids, splits, categories, rubric sizes."""
    selected = _tasks(split)
    table = Table("split", "category", "facts", "rubric", "tools", "changes", "critical")
    table.add_column("id", no_wrap=True)
    for t in selected.values():
        table.add_row(t.split, t.category, str(len(t.facts)), str(len(t.rubric)), str(len(t.tools)),
                      "yes" if t.allow_changes else "", "yes" if t.critical else "", t.id)  # fmt: skip
    console.print(table)
    console.print(Counter(t.split for t in selected.values()), Counter(t.category for t in selected.values()))


@app.command()
def run(
    config: list[str] = typer.Option(["single/qwen3.5:9b"], help="mode/model[+think], repeatable."),
    split: str = SPLIT,
    ids: str = typer.Option("", "--tasks", help="Comma-separated task ids."),
    category: str = typer.Option("", help="Comma-separated categories."),
    repeat: int = typer.Option(1, help="Runs per task and config."),
    tag: str = typer.Option("run", help="Name part of the results file."),
    out: Path | None = typer.Option(None, help="Append to (resume) this results file."),
    timeout: float = typer.Option(900, help="Seconds per run before it counts as failed."),
) -> None:
    """Run tasks through configurations; writes are rejected unless the task asks for them."""
    from devcopilot.evals.run import Config, new_results_path, run_eval

    configs = [Config.parse(c) for c in config]
    selected = list(_tasks(split, ids, category).values())
    out = out or new_results_path(tag)
    console.print(f"[dim]{len(selected)} tasks × {repeat} × {len(configs)} configs -> {out}[/dim]")

    def on_record(r: dict) -> None:
        res = r["result"]
        status = f"[red]{r['error'][:80]}[/red]" if r["error"] else (
            f"{res['seconds']:.0f}s, {res['tool_calls']} tools, {res['llm_calls']} LLM")  # fmt: skip
        console.print(f"{r['config']['name']} #{r['repeat']} {r['task']}: {status}")

    asyncio.run(run_eval(selected, configs, repeat, out, on_record, timeout))
    console.print(f"[green]saved[/green] {out}")


@app.command()
def judge(
    files: list[Path] = FILES,
    model: str = typer.Option(None, help="Judge model; default: the configured model; 'jev' for Jev."),
    reasoning: bool = typer.Option(False, help="Let the judge think."),
) -> None:
    """Add judge verdicts to results files (skips fresh verdicts). --model jev: Jev, one question per criterion."""
    from devcopilot.evals.run import judge_file

    tasks_by_id = _tasks()
    model = model or get_settings().model
    if model != "jev":
        from devcopilot.evals.run import load_records
        from devcopilot.llm import same_family

        # Invariant 7 for judges: a judge of the agent's own family is lenient to its answers.
        agents = {r["config"].get("model") for f in files for r in load_records(f)} - {None}
        if same := sorted(a for a in agents if same_family(a, model)):
            console.print(f"[yellow]judge {model} is of the same family as the agent ({', '.join(same)}): "
                          "prefer gemma4:e4b for Qwen agents[/yellow]")  # fmt: skip

    def on_verdict(r: dict, v: dict) -> None:
        marks = "".join("✓" if c["met"] else "✗" for c in v["criteria"]) or v["error"]
        console.print(f"{r['id']}: {marks}")

    for f in files:
        n = asyncio.run(judge_file(f, tasks_by_id, model, reasoning, on_verdict))
        console.print(f"[green]{f.name}[/green]: {n} judged")


@app.command()
def report(
    files: list[Path] = FILES,
    judge: str = typer.Option(None, help="Judge key, e.g. qwen3.5:9b; default: the most common one."),
    metric: str = typer.Option("pass", help="pass (needs the judge) or pass_facts."),
    split: str = SPLIT,
    by_task: bool = typer.Option(False, help="Pass rate per task and config."),
) -> None:
    """Score results against the current dataset: pass@1 with CI, pass@k, pass^k, checks, cost."""
    from devcopilot.evals.report import (
        CHECKS,
        agent_categories,
        assignment_metrics,
        by_category,
        flaky,
        group,
        summarize,
    )

    tasks_by_id = _tasks(split)
    records = _records(files)
    judge = judge or _default_judge(records)
    if metric == "pass" and judge is None:
        console.print("[yellow]no judge verdicts: falling back to pass_facts[/yellow]")
        metric = "pass_facts"
    console.print(f"[dim]metric={metric} judge={judge} split={split}[/dim]")
    groups = group(records)
    summaries = {name: summarize(rs, tasks_by_id, judge, metric) for name, rs in groups.items()}

    table = Table("config", "tasks", "runs", "err", "pass@1", "90% CI", "k", "pass@k", "pass^k", title=metric)
    for name, s in summaries.items():
        table.add_row(name, str(s.tasks), str(s.runs), str(s.errors), _pct(s.pass_rate),
                      "—" if s.ci[0] != s.ci[0] else f"{_pct(s.ci[0])}–{_pct(s.ci[1])}", str(s.k),
                      _pct(s.pass_at_k), _pct(s.pass_hat_k))  # fmt: skip
    console.print(table)

    checks = Table("config", *CHECKS, "judge", title="share of runs passing each check")
    for name, s in summaries.items():
        checks.add_row(name, *(_pct(s.rates[c]) for c in (*CHECKS, "judge")))
    console.print(checks)

    cost = Table("config", "tools", "repeats", "LLM", "tokens in", "seconds", title="mean per run")
    for name, s in summaries.items():
        m = s.means
        cost.add_row(name, f"{m['tool_calls']:.1f}", f"{m['duplicates']:.1f}", f"{m['llm_calls']:.1f}",
                     f"{m['tokens_in']:.0f}", f"{m['seconds']:.0f}")  # fmt: skip
    console.print(cost)

    tz = {name: assignment_metrics(rs, tasks_by_id, judge, metric) for name, rs in groups.items()}
    # Shares print as %, counts and seconds as numbers (handoff is a share, the rest of coordination counts).
    shares = {"product", "agentic"}
    for category in ("product", "agentic", "coordination", "infra"):
        names = list(next(iter(tz.values()))[category]) if tz else []
        table = Table("config", *names, title=f"{category} metrics")
        for name, m in tz.items():
            cells = [_pct(v) if (category in shares and name_ != "repeats") or name_ == "handoff" else f"{v:.2f}"
                     for name_, v in m[category].items()]  # fmt: skip
            table.add_row(name, *cells)
        console.print(table)

    agent_metrics = {name: agent_categories(rs, tasks_by_id, judge, metric) for name, rs in groups.items()}
    for category in next(iter(agent_metrics.values()), {}):
        names = list(next(iter(agent_metrics.values()))[category])
        table = Table("config", *names, title=f"agent evaluation: {category}")
        for name, categories in agent_metrics.items():
            table.add_row(
                name,
                *(
                    str(value)
                    if isinstance(value, str)
                    else f"{value:.1f}"
                    if key in {"context_chars", "escalations_per_run"}
                    else _pct(value)
                    for key, value in categories[category].items()
                ),
            )
        console.print(table)

    cats = {name: by_category(rs, tasks_by_id, judge, metric) for name, rs in groups.items()}
    all_cats = sorted({c for v in cats.values() for c in v})
    cat_table = Table("config", *all_cats, title=f"{metric} by category")
    for name, v in cats.items():
        cat_table.add_row(name, *(_pct(v.get(c, float("nan"))) for c in all_cats))
    console.print(cat_table)

    for name, rs in groups.items():
        if f := flaky(rs, tasks_by_id, judge, metric):
            console.print(f"[dim]{name} unstable:[/dim] " + ", ".join(f"{t} {r:.0%}" for t, r in f.items()))

    if by_task:
        per_task = Table("task", *summaries, title=f"{metric} per task")
        for task_id in tasks_by_id:
            cells = []
            for s in summaries.values():
                rate = s.per_task.get(task_id)
                color = "green" if rate == 1 else "red" if rate == 0 else "yellow"
                cells.append("" if rate is None else f"[{color}]{rate:.0%}[/{color}]")
            per_task.add_row(task_id, *cells)
        console.print(per_task)


@app.command()
def show(
    files: list[Path] = FILES,
    task: str = typer.Option(..., help="Task id."),
    config: str = typer.Option("", help="Only this config name."),
    judge: str = typer.Option(None, help="Judge key."),
) -> None:
    """Show answers, checks and verdicts for one task: the first thing to read when it fails."""
    from devcopilot.evals.report import score

    t = _tasks()[task]
    records = [r for r in _records(files) if r["task"] == task and (not config or r["config"]["name"] == config)]
    judge = judge or _default_judge(records)
    console.print(Panel(f"{t.question}\n\n[dim]{t.reference.strip()}[/dim]", title=task))
    for r in records:
        if r["error"]:
            console.print(f"[red]{r['id']}: {r['error']}[/red]")
            continue
        s = score(r, t, judge)
        failed = [k for k in ("facts", "forbid", "grounded", "language", "tools", "efficient", "calls", "safe")
                  if not s[k]]  # fmt: skip
        tools = " → ".join(f"{x['agent'] + ':' if x['agent'] != 'single' else ''}{x['tool']}"
                           for x in r["result"]["trajectory"])  # fmt: skip
        console.print(Panel(Markdown(r["result"]["answer"]), title=f"{r['id']} judge={s['judge']}"))
        console.print(f"[dim]tools: {tools or '(none)'}[/dim]")
        if failed:
            console.print(f"[red]failed checks: {', '.join(failed)}[/red] {s['details']}")
        if v := r["judges"].get(judge):
            for crit, c in zip(t.rubric, v["criteria"], strict=False):
                console.print(f"  {'✓' if c['met'] else '✗'} {crit}\n    [dim]{c['evidence']}[/dim]")


@app.command()
def compare(
    base: Path = typer.Argument(..., exists=True),
    new: Path = typer.Argument(..., exists=True),
    judge: str = typer.Option(None, help="Judge key."),
    metric: str = typer.Option("pass", help="pass or pass_facts."),
    split: str = SPLIT,
    base_config: str = typer.Option("", help="Config in BASE to compare, if it holds several."),
    new_config: str = typer.Option("", help="Config in NEW to compare, if it holds several."),
) -> None:
    """Regression gate: per-task changes, paired bootstrap, exit code 1 on FAIL."""
    from devcopilot.evals.report import compare as gate
    from devcopilot.evals.report import group

    tasks_by_id = _tasks(split)

    def one_config(path: Path, chosen: str) -> list[dict]:
        # Mixing configs into one side compares an average of models with a baseline: meaningless.
        groups = group(_records([path]))
        if chosen:
            if chosen not in groups:
                raise typer.BadParameter(f"{path.name} has no config {chosen!r}; it has: {', '.join(groups)}")
            return groups[chosen]
        if len(groups) > 1:
            raise typer.BadParameter(f"{path.name} holds {len(groups)} configs, choose one: {', '.join(groups)}")
        return next(iter(groups.values()), [])

    base_records, new_records = one_config(base, base_config), one_config(new, new_config)
    judge = judge or _default_judge(base_records + new_records)
    if metric == "pass" and judge is None:
        metric = "pass_facts"
    result = gate(base_records, new_records, tasks_by_id, judge, metric)
    d = result.delta
    console.print(f"[dim]metric={metric} judge={judge}[/dim]")
    console.print(f"{d['tasks']} common tasks, delta {d['delta']:+.2f} (90% CI {d['lo']:+.2f}…{d['hi']:+.2f}), "
                  f"P(worse) {d['p_worse']:.2f}")  # fmt: skip
    for task_id, (old, now) in result.changes.items():
        color = "green" if now > old else "red"
        mark = " [bold](critical)[/bold]" if tasks_by_id[task_id].critical else ""
        console.print(f"  [{color}]{task_id}: {old:.0%} -> {now:.0%}[/{color}]{mark}")
    color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[result.verdict]
    console.print(f"[bold {color}]{result.verdict}[/bold {color}] " + "; ".join(result.reasons))
    if result.verdict == "FAIL":
        raise typer.Exit(1)


@app.command()
def components(
    model: str = typer.Option(None, help="Model; default: the configured one."),
    reasoning: bool = typer.Option(False, help="Thinking on."),
    split: str = SPLIT,
    ids: str = typer.Option("", "--tasks", help="Comma-separated task ids."),
    repeat: int = typer.Option(1, help="Decisions per task."),
    out: Path | None = typer.Option(None, help="Append to (resume) this results file."),
    jev_gate: bool = typer.Option(False, help="Decide the gate with Jev instead of the LLM."),
    gate_only: bool = typer.Option(False, help="Only the gate: no supervisor, no LLM if --jev-gate."),
) -> None:
    """Gate accuracy and the supervisor's first routing decision, one LLM call each."""
    import json

    from devcopilot.evals.components import eval_components, summarize_components
    from devcopilot.evals.run import load_records, new_results_path
    from devcopilot.llm import get_chat_model, unload_model
    from devcopilot.mcp_client import open_mcp_tools

    tasks_by_id = _tasks(split, ids)
    model = model or get_settings().model
    suffix = f"{'-think' if reasoning else ''}{'-gateonly' if gate_only else ''}"
    name = f"jev-{get_settings().jev_model}" if jev_gate else model
    out = out or new_results_path(f"components-{name.replace(':', '_')}{suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_records(out)

    def on_record(r: dict) -> None:
        agents = ",".join(a["agent"] for a in r["assignments"]) or "(none)"
        console.print(f"{r['task']}: gate={r['gate']} -> {agents} {r['error'] or ''}")
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    async def go():
        from devcopilot.decide import JevDecider

        gate = JevDecider("gate", threshold=get_settings().jev_gate_threshold) if jev_gate else None
        async with open_mcp_tools() as tools:
            chat = await asyncio.to_thread(get_chat_model, model, reasoning)
            return await eval_components(list(tasks_by_id.values()), chat, tools,
                                         repeat, on_record, done, gate, gate_only)  # fmt: skip

    try:
        records = asyncio.run(go())
    finally:
        if not (jev_gate and gate_only):
            unload_model(model)
    _print_components(summarize_components(records, tasks_by_id))
    console.print(f"[dim]saved to {out}[/dim]")


def _print_components(s: dict) -> None:
    g, r = s["gate"], s["routing"]
    console.print(f"{s['decisions']} decisions, {s['errors']} errors")
    console.print(f"[bold]gate[/bold] accuracy {_pct(g['accuracy'])}: allowed correctly {g['tp']}, "
                  f"denied correctly {g['tn']}, [red]allowed wrongly {g['fp']}[/red] {g['false_allow']}, "
                  f"denied wrongly {g['fn']} {g['false_deny']}")  # fmt: skip
    console.print(f"[bold]routing[/bold] (round 1) hit rate {_pct(r['hit_rate'])}, precision {_pct(r['precision'])}, "
                  f"no assignment {r['empty']}, unavailable agent {r['unavailable']}")  # fmt: skip
    for task_id, got in r["misses"].items():
        console.print(f"  [red]miss[/red] {task_id}: {'; '.join(got)}")


@app.command()
def label(
    files: list[Path] = FILES,
    annotator: str = typer.Option(..., help="Your name: labels go to evals/labels/<annotator>.jsonl."),
    n: int = typer.Option(40, help="How many answers to label in this session."),
    seed: int = typer.Option(0, help="Sampling seed: same seed, same answers for every annotator."),
) -> None:
    """Label answers pass/fail by the rubric. The judge's verdict is not shown: no anchoring."""
    from devcopilot.evals.labels import load_labels, sample_for_labeling, save_label

    tasks_by_id = _tasks()
    done = load_labels(annotator)
    picked = sample_for_labeling(_records(files), tasks_by_id, n, set(done), seed)
    console.print(f"[dim]{len(done)} labeled before, {len(picked)} to go. p = pass, f = fail, s = skip, q = quit[/dim]")
    for i, (key, r) in enumerate(picked, 1):
        t = tasks_by_id[r["task"]]
        rubric = "\n".join(f"{j}. {c}" for j, c in enumerate(t.rubric, 1))
        console.rule(f"{i}/{len(picked)} · {t.id}")
        console.print(Panel(f"[bold]{t.question}[/bold]\n\n[dim]Reference:[/dim] {t.reference.strip()}\n\n"
                            f"[dim]Rubric:[/dim]\n{rubric}"))  # fmt: skip
        console.print(Panel(Markdown(r["result"]["answer"] or "(empty)"), title="answer", border_style="cyan"))
        choice = Prompt.ask("all criteria met?", choices=["p", "f", "s", "q"])
        if choice == "q":
            break
        if choice == "s":
            continue
        note = Prompt.ask("note (optional)", default="")
        save_label(annotator, key, t.id, choice == "p", note)


@app.command()
def calibrate(
    files: list[Path] = FILES,
    annotator: str = typer.Option(..., help="Whose labels are the truth."),
    judge: str = typer.Option(None, help="Judge key; default: the most common one."),
    vs: str = typer.Option("", help="Another annotator: agreement between humans."),
    out: Path = typer.Option(None, help="Also save the result as JSON (for reports)."),
) -> None:
    """Agreement and Cohen's kappa: judge vs human, substring check vs human."""
    from devcopilot.evals.labels import calibrate as run_calibration
    from devcopilot.evals.labels import inter_annotator, load_labels

    records = _records(files)
    judge = judge or _default_judge(records)
    labels = load_labels(annotator)
    c = run_calibration(records, _tasks(), labels, judge)
    if out:
        _save_json(out, {"annotator": annotator, "judge": judge, "files": [str(f) for f in files], **c})
    console.print(f"{c['labeled']} labeled answers, human pass rate {_pct(c['human_pass_rate'])}, judge={judge}")
    table = Table("rater", "n", "agreement", "kappa", "both pass", "rater passes, human fails",
                  "rater fails, human passes", "both fail")  # fmt: skip
    for rater in ("judge", "facts"):
        x = c[rater]
        table.add_row(rater, str(x["n"]), _pct(x["agreement"]), f"{x['kappa']:.2f}", str(x["tp"]), str(x["fp"]),
                      str(x["fn"]), str(x["tn"]))  # fmt: skip
    console.print(table)
    if c["sweep"]:
        # Jev gives a probability: the threshold is ours to choose, and labels choose it.
        sweep = Table("threshold", "agreement", "kappa", "passes what human fails", "fails what human passes",
                      title="judge p (lowest criterion) vs human, by threshold")  # fmt: skip
        for x in c["sweep"]:
            sweep.add_row(
                f"{x['threshold']:.1f}", _pct(x["agreement"]), f"{x['kappa']:.2f}", str(x["fp"]), str(x["fn"])
            )
        console.print(sweep)
    for x in c["judge"]["disagreements"]:
        console.print(f"  judge {'pass' if x['judge'] else 'fail'}, human {'pass' if x['human'] else 'fail'}: "
                      f"{x['task']} [dim]{x['key']} {x['note']}[/dim]")  # fmt: skip
    if vs:
        ia = inter_annotator(labels, load_labels(vs))
        console.print(f"{annotator} vs {vs}: n={ia['n']}, agreement {_pct(ia['agreement'])}, "
                      f"kappa {ia['kappa']:.2f}")  # fmt: skip
        for task_id, key in ia["disagreements"]:
            console.print(f"  disagree: {task_id} [dim]{key}[/dim]")


@app.command()
def online(
    phoenix: str = typer.Option("http://localhost:6006", help="Phoenix URL."),
    sample: int = typer.Option(10, help="One-off mode: how many runs to check (random sample)."),
    judge: bool = typer.Option(True, help="Also run the reference-free LLM judge."),
    model: str = typer.Option(None, help="Judge model; default: the validator's (another family); 'jev' for Jev."),
    dry_run: bool = typer.Option(False, help="Do not write annotations back to Phoenix."),
    watch: bool = typer.Option(False, help="Continuous evaluation: check new runs every --interval seconds."),
    interval: int = typer.Option(300, help="Watch mode: seconds between ticks."),
    rate: float = typer.Option(1.0, help="Watch mode: share of new runs to check (deterministic by trace)."),
    max_runs: int = typer.Option(20, help="Watch mode: at most this many runs per tick (the judge shares the GPU)."),
    state: Path = typer.Option(None, help="Watch mode: file of checked traces; default results/online-state.json."),
) -> None:
    """Check real runs from Phoenix without reference answers; write annotations back and export metrics."""
    import time

    import httpx

    from devcopilot.decide import JevDecider
    from devcopilot.evals.online import online_eval, watch_tick
    from devcopilot.evals.run import RESULTS
    from devcopilot.llm import get_chat_model, route, unload_model

    # The judge of the agents' answers: the validator's model, another family (invariant 7).
    model = model or route("validator")
    use_llm = judge and model != "jev"
    jev = JevDecider("online") if judge and model == "jev" else None
    judge_model = get_chat_model(model, False) if use_llm else None

    def show(results) -> None:
        for run, checks in results:
            marks = " ".join(f"[{'green' if ok else 'red'}]{name}[/]" for name, (ok, _) in checks.items())
            console.print(f"{run['start_time'][:19]} {run['mode']} {run['question'][:60]!r}: {marks}")
            for name, (ok, why) in checks.items():
                if not ok and why:
                    console.print(f"    [dim]{name}: {why[:150]}[/dim]")

    if not watch:
        try:
            results = asyncio.run(online_eval(phoenix, sample, judge_model, model if judge else None,
                                              dry_run=dry_run, jev=jev))  # fmt: skip
        finally:
            if use_llm:
                unload_model(model)
        show(results)
        where = "not written (dry run)" if dry_run else "written to Phoenix as annotations and metrics"
        console.print(f"[dim]{len(results)} runs checked, {where}[/dim]")
        return

    state = state or RESULTS.parent / "online-state.json"
    console.print(f"[dim]continuous evaluation: every {interval} s, rate {rate}, judge {model if judge else '-'}, "
                  f"state {state}[/dim]")  # fmt: skip
    while True:
        try:
            with httpx.Client(base_url=phoenix, timeout=30) as client:
                results = asyncio.run(watch_tick(client, state, judge_model, model if judge else None,
                                                 rate=rate, max_runs=max_runs, jev=jev))  # fmt: skip
            show(results)
            console.print(f"[dim]{time.strftime('%H:%M:%S')} tick: {len(results)} new runs checked[/dim]")
        except (httpx.HTTPError, OSError) as e:  # Phoenix restarting: try again next tick
            console.print(f"[yellow]tick failed: {e}[/yellow]")
        finally:
            if use_llm:
                unload_model(model)  # free the GPU for the agents between ticks
        time.sleep(interval)


@app.command()
def llm(
    models: str = typer.Option(
        ..., help="Comma-separated Ollama models, e.g. qwen3.5:9b,gemma4:e4b; name+think = with thinking."
    ),
    out: Path = typer.Option(None, help="Results JSONL; default results/evals/llm-bench.jsonl. Resumable."),
    report_only: bool = typer.Option(False, help="Only print the table of an existing file."),
) -> None:
    """Level-1 evals of base LLMs: reasoning, faithfulness, structured output, tool calling."""
    from devcopilot.evals.llm_bench import CRITERIA, load_bench, run_item, summarize
    from devcopilot.evals.run import RESULTS, append_record, llm_host, load_records
    from devcopilot.llm import unload_model

    out = out or RESULTS / "llm-bench.jsonl"
    items, tools = load_bench()
    names = [m.strip() for m in models.split(",") if m.strip()]

    async def run() -> None:
        done = {(r["model"], r["item"]) for r in load_records(out)}
        out.parent.mkdir(parents=True, exist_ok=True)
        for name in names:
            model, think = name.removesuffix("+think"), name.endswith("+think")
            try:
                for item in items:
                    if (name, item.id) in done:
                        continue
                    r = {"model": name, "host": llm_host(), **await run_item(model, item, tools, think)}
                    append_record(out, r)
                    mark = "[green]✓[/green]" if r["pass"] else f"[red]✗[/red] {r['detail']}"
                    console.print(f"{name} {item.id} {r['seconds']:.1f}s {mark}")
            finally:
                await asyncio.to_thread(unload_model, model)

    if not report_only:
        asyncio.run(run())
    records = [r for r in load_records(out) if r["model"] in names]
    table = Table("model", *CRITERIA, "all", "s/item", title=f"base LLMs ({len(items)} items)")
    for name, row in summarize(records).items():
        table.add_row(name, *(_pct(row[c]) for c in (*CRITERIA, "all")), f"{row['seconds']:.1f}")
    console.print(table)


@app.command()
def memory(
    k: int = typer.Option(3, help="Episodes recalled per query."),
    out: Path = typer.Option(None, help="Also save the result as JSON (for reports)."),
    sizes: str = typer.Option(
        "8,50,200", help="Degradation: total episodes per step, comma-separated; '' skips it (and its embeddings)."
    ),
) -> None:
    """Memory evals: Context Precision/Recall of episodes (needs the embedding model), temporal facts,
    degradation as memory grows and with outdated episodes."""
    from devcopilot.evals.memory_bench import degradation, retrieval, temporal
    from devcopilot.evals.run import llm_host

    checks = temporal()
    for name, ok in checks:
        console.print(f"{'[green]✓[/green]' if ok else '[red]✗[/red]'} {name}")
    console.print(f"temporal facts: {sum(ok for _, ok in checks)}/{len(checks)}")

    result = asyncio.run(retrieval(k))
    steps = [int(x) for x in sizes.split(",") if x.strip()]
    degraded = asyncio.run(degradation(steps, k)) if steps else None
    if out:
        s = get_settings()
        extra = {"degradation": degraded} if degraded else {}
        _save_json(out, {"embed_model": s.embed_model, "host": llm_host(), "min_similarity": s.episode_min_similarity,
                         "temporal": [{"check": n, "ok": ok} for n, ok in checks], **result, **extra})  # fmt: skip
    table = Table("min similarity", f"Context Precision@{k}", f"Context Recall@{k}",
                  title=f"episodes, embed_model={get_settings().embed_model}")  # fmt: skip
    for row in result["sweep"]:
        table.add_row(f"{row['threshold']:.1f}", _pct(row["precision"]), _pct(row["recall"]))
    console.print(table)
    for q in result["queries"]:
        hits = ", ".join(f"{key} {s:.2f}" for key, s in q["hits"])
        console.print(f"[dim]{q['q']}[/dim] → {hits}  (relevant: {', '.join(q['relevant']) or '-'})")
    if degraded:
        table = Table("episodes", f"Context Precision@{k}", f"Context Recall@{k}", "distractors in results",
                      "<memory> chars mean", "max",
                      title=f"memory growth, min similarity {degraded['min_similarity']}")  # fmt: skip
        for row in degraded["sizes"]:
            table.add_row(str(row["episodes"]), _pct(row["precision"]), _pct(row["recall"]),
                          _pct(row["distractor_share"]), f"{row['context_chars_mean']:.0f}",
                          str(row["context_chars_max"]))  # fmt: skip
        console.print(table)
        st = degraded["staleness"]
        console.print(
            f"outdated episodes ({st['pairs']} pairs, {st['old_age_days']} vs {st['new_age_days']} days): newer first "
            f"{_pct(st['newer_first'])} (by similarity alone {_pct(st['newer_first_by_similarity'])}), "
            f"older returned {_pct(st['old_returned'])}: flagged outdated {_pct(st['old_returned_flagged'])}, "
            f"unflagged {_pct(st['old_returned_unflagged'])}"
        )


def _dialogs(ids: str = "") -> dict:
    from devcopilot.evals.dialogs import load_dialogs

    return {d.id: d for d in load_dialogs() if not ids or d.id in ids.split(",")}


@app.command()
def dialogs(
    config: list[str] = typer.Option(["single/qwen3.5:9b"], help="mode/model[+...], repeatable."),
    ids: str = typer.Option("", "--dialogs", help="Comma-separated dialog ids (evals/dialogs.yaml)."),
    repeat: int = typer.Option(1, help="Runs per dialog and config."),
    out: Path = typer.Option(None, help="Append to (resume) this results file."),
    memory: bool = typer.Option(False, help="Long-term memory per dialog (temp dir); the config name gets +memory."),
    timeout: float = typer.Option(900, help="Seconds per turn before it counts as failed."),
    report_after: bool = typer.Option(True, "--report/--no-report", help="Print the summary when done."),
) -> None:
    """Multi-turn dialogs: one thread per dialog, the world changes between turns; one record per turn."""
    from devcopilot.evals.dialogs import run_dialogs
    from devcopilot.evals.run import Config, new_results_path

    configs = [Config.parse(c) for c in config]
    selected = list(_dialogs(ids).values())
    out = out or new_results_path("dialogs")
    turns = sum(len(d.turns) for d in selected)
    console.print(f"[dim]{len(selected)} dialogs ({turns} turns) × {repeat} × {len(configs)} configs -> {out}[/dim]")

    def on_record(r: dict) -> None:
        res = r["result"]
        if r["error"]:
            status = f"[red]{r['error'][:80]}[/red]"
        else:
            failed = [c for c, ok in r["checks"].items() if not ok]
            status = (f"{'[green]ok[/green]' if not failed else '[red]' + ','.join(failed) + '[/red]'} "
                      f"{res['seconds']:.0f}s, {res['tool_calls']} tools, {res['tokens_in']} tokens in")  # fmt: skip
        change = " [yellow](world changed)[/yellow]" if r["after_change"] else ""
        console.print(f"{r['config']['name']} #{r['repeat']} {r['dialog']} turn {r['turn']}{change}: {status}")

    asyncio.run(run_dialogs(selected, configs, repeat, out, on_record, timeout, memory))
    console.print(f"[green]saved[/green] {out}")
    if report_after:
        _print_dialogs([out], None)


@app.command("dialogs-report")
def dialogs_report(
    files: list[Path] = FILES,
    out: Path = typer.Option(None, help="Also save the summary as JSON (for reports)."),
) -> None:
    """Dialog evals: pass rate by turn, after a world change, tokens in by turn (context growth)."""
    _print_dialogs(files, out)


def _print_dialogs(files: list[Path], out: Path | None) -> None:
    from devcopilot.evals.dialogs import summarize_dialogs

    summary = summarize_dialogs(_records(files), _dialogs())
    if out:
        _save_json(out, {"files": [str(f) for f in files], "configs": summary})
    turns = sorted({t for s in summary.values() for t in s["by_turn"]})
    table = Table("config", *(f"turn {t}" for t in turns), "turns 2+", "after change", "no change", "drop",
                  "skipped", title="pass rate by turn (facts, forbid, grounded, language, safe)")  # fmt: skip
    for name, s in summary.items():
        later, changed, same = s["later_turns"], s["after_change"], s["no_change"]
        table.add_row(name, *(_pct(s["by_turn"][t]["pass"]) if t in s["by_turn"] else "" for t in turns),
                      _pct(later["pass"]), f"{_pct(changed['pass'])} ({changed['runs']})",
                      f"{_pct(same['pass'])} ({same['runs']})",
                      "—" if s["degradation"] != s["degradation"] else f"{s['degradation']:+.0%}",
                      str(s["skipped"]))  # fmt: skip
    console.print(table)

    def num(x: float, fmt: str = ".0f") -> str:
        return "—" if x != x else format(x, fmt)

    cost = Table("config", "turn", "runs", "err", "tokens in", "context chars", "tools", "seconds", "facts",
                 "forbid", "grounded", title="mean per turn: does the context grow with the dialog?")  # fmt: skip
    for name, s in summary.items():
        for t, b in s["by_turn"].items():
            cost.add_row(name, str(t), str(b["runs"]), str(b["errors"]), num(b["tokens_in"]), num(b["context_chars"]),
                         num(b["tool_calls"], ".1f"), num(b["seconds"]), _pct(b["facts"]), _pct(b["forbid"]),
                         _pct(b["grounded"]))  # fmt: skip
    console.print(cost)

    per_dialog = sorted({d for s in summary.values() for d in s["by_dialog"]})
    table = Table("dialog", *summary, title="pass rate per dialog (all turns)")
    for d in per_dialog:
        table.add_row(d, *(_pct(s["by_dialog"].get(d, float("nan"))) for s in summary.values()))
    console.print(table)
