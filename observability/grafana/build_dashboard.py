"""Generates dashboards/devcopilot.json. Dashboard as code: edit this, not the JSON.

    uv run python observability/grafana/build_dashboard.py

Design rules:
- headline numbers are stat tiles; change over time is thin lines or bars;
- an entity keeps its color everywhere: mode single = blue, team = orange;
- status colors (good / warning / critical) mean state only, never "series 4";
- no more than a few hues per chart: comparisons across 10 agents or 14 tools are
  single-hue horizontal bars with labels, not a rainbow of lines.
"""

import json
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prometheus"}
LOKI = {"type": "loki", "uid": "loki"}

# Categorical slots 1-2 and the status palette, steps for Grafana's dark theme.
BLUE, ORANGE = "#3987e5", "#d95926"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
MODE_COLORS = {"single": BLUE, "team": ORANGE}
STATUS_COLORS = {"success": GOOD, "rejected": SERIOUS, "error": CRITICAL, "unknown_tool": CRITICAL}

_next_id = iter(range(1, 1000))


def target(expr: str, legend: str = "", instant: bool = False) -> dict:
    return {"datasource": PROM, "expr": expr, "legendFormat": legend, "instant": instant, "range": not instant,
            "refId": "A"}  # fmt: skip


def color_overrides(colors: dict[str, str], by: str = "byName") -> list[dict]:
    return [
        {
            "matcher": {"id": by, "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}],
        }
        for name, c in colors.items()
    ]


def stat(title, expr, x, y, unit="short", thresholds=None, description="", decimals=None, no_value="0"):
    """A KPI tile. Neutral tiles show plain text; status tiles color the value by threshold."""
    steps = thresholds or [{"color": "text", "value": None}]
    return {
        "id": next(_next_id), "type": "stat", "title": title, "description": description, "datasource": PROM,
        "gridPos": {"x": x, "y": y, "w": 4, "h": 4},
        "targets": [target(expr, instant=True)],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals, "noValue": no_value,
                                     "thresholds": {"mode": "absolute", "steps": steps}}, "overrides": []},
        "options": {"colorMode": "value" if thresholds else "none", "graphMode": "none", "textMode": "value",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
    }  # fmt: skip


def timeseries(title, targets, x, y, w, overrides=(), unit="short", bars=False, stacked=False, description=""):
    return {
        "id": next(_next_id), "type": "timeseries", "title": title, "description": description, "datasource": PROM,
        "gridPos": {"x": x, "y": y, "w": w, "h": 8},
        "targets": [{**t, "refId": chr(65 + i)} for i, t in enumerate(targets)],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": "fixed", "fixedColor": BLUE},
                "custom": {
                    "drawStyle": "bars" if bars else "line", "lineWidth": 2, "fillOpacity": 80 if bars else 0,
                    "pointSize": 8, "showPoints": "never" if not bars else "auto", "spanNulls": True,
                    "stacking": {"mode": "normal" if stacked else "none"}, "axisSoftMin": 0,
                    "gradientMode": "none",
                },
            },
            "overrides": list(overrides),
        },
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
    }  # fmt: skip


def bars(title, expr, legend, x, y, w, unit="short", description="", h=8):
    """Single-hue horizontal bars: compares many labeled items without needing many colors."""
    return {
        "id": next(_next_id), "type": "bargauge", "title": title, "description": description, "datasource": PROM,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr, legend, instant=True)],
        "fieldConfig": {"defaults": {"unit": unit, "min": 0, "color": {"mode": "fixed", "fixedColor": BLUE},
                                     "thresholds": {"mode": "absolute", "steps": [{"color": BLUE, "value": None}]}},
                        "overrides": []},
        "options": {"orientation": "horizontal", "displayMode": "basic", "showUnfilled": True, "valueMode": "text",
                    "namePlacement": "left", "sizing": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
    }  # fmt: skip


def row(title, y):
    return {"id": next(_next_id), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}  # fmt: skip


R = "$__range"  # totals over the selected time range
INTERVAL = "$__rate_interval"  # per-bucket increments for "over time" panels
ERRORS = 'status=~"error|unknown_tool"'

panels = [
    row("Overview (selected time range)", 0),
    stat("Runs", f"sum(increase(devcopilot_runs_total[{R}]))", 0, 1, decimals=0),
    stat("Run duration p95", f"histogram_quantile(0.95, sum by (le) (increase(devcopilot_run_duration_seconds_bucket[{R}])))", 4, 1, unit="s"),
    stat("Input tokens per run", f'sum(increase(gen_ai_client_token_usage_sum{{gen_ai_token_type="input"}}[{R}])) / clamp_min(sum(increase(devcopilot_runs_total{{outcome="ok"}}[{R}])), 1)', 8, 1, decimals=0),
    stat("Tool error rate", f"sum(increase(devcopilot_tool_calls_total{{{ERRORS}}}[{R}])) / sum(increase(devcopilot_tool_calls_total[{R}]))", 12, 1,
         unit="percentunit", thresholds=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 0.05}, {"color": CRITICAL, "value": 0.2}]),
    stat("Blocked by gate", f"sum(increase(devcopilot_gate_blocked_total[{R}]))", 16, 1, decimals=0,
         thresholds=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}],
         description="Write-capable agent requested on a read-only request: possible prompt injection."),
    stat("Ungrounded entities", f"sum(increase(devcopilot_grounding_ungrounded_total[{R}]))", 20, 1, decimals=0,
         thresholds=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 1}],
         description="Links, hashes, issue numbers the agent mentioned but never saw in tool output."),

    row("Runs", 5),
    timeseries("Runs by mode", [target(f'sum by (mode) (increase(devcopilot_runs_total{{outcome="ok"}}[{INTERVAL}]))', "{{mode}}")],
               0, 6, 12, color_overrides(MODE_COLORS), bars=True, stacked=True),
    timeseries("Run duration p50 / p95 by mode", [
        target(f"histogram_quantile(0.95, sum by (le, mode) (rate(devcopilot_run_duration_seconds_bucket[{INTERVAL}])))", "{{mode}} p95"),
        target(f"histogram_quantile(0.50, sum by (le, mode) (rate(devcopilot_run_duration_seconds_bucket[{INTERVAL}])))", "{{mode}} p50"),
    ], 12, 6, 12, [
        *color_overrides({f"/^{m} /": c for m, c in MODE_COLORS.items()}, by="byRegexp"),
        {"matcher": {"id": "byRegexp", "options": "/p50$/"},
         "properties": [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [6, 4]}}]},
    ], unit="s", description="Solid: p95, dashed: p50."),

    row("LLM calls by agent (selected time range)", 14),
    bars("LLM latency p95", f"histogram_quantile(0.95, sum by (le, devcopilot_agent) (increase(gen_ai_client_operation_duration_seconds_bucket[{R}])))",
         "{{devcopilot_agent}}", 0, 15, 8, unit="s"),
    bars("Input tokens", f'sum by (devcopilot_agent) (increase(gen_ai_client_token_usage_sum{{gen_ai_token_type="input"}}[{R}]))',
         "{{devcopilot_agent}}", 8, 15, 8),
    bars("LLM calls", f"sum by (devcopilot_agent) (increase(gen_ai_client_operation_duration_seconds_count[{R}]))",
         "{{devcopilot_agent}}", 16, 15, 8),

    row("Tools", 23),
    timeseries("Tool calls by status", [target(f"sum by (status) (increase(devcopilot_tool_calls_total[{INTERVAL}]))", "{{status}}")],
               0, 24, 12, color_overrides(STATUS_COLORS), bars=True, stacked=True,
               description="Status colors: success, rejected by a human, error."),
    bars("Calls by tool", f"sum by (tool) (increase(devcopilot_tool_calls_total[{R}]))", "{{tool}}", 12, 24, 6),
    bars("Duration p95 by tool", f"histogram_quantile(0.95, sum by (le, tool) (increase(devcopilot_tool_duration_seconds_bucket[{R}])))",
         "{{tool}}", 18, 24, 6, unit="s"),

    row("Control and safety", 32),
    bars("Approvals by decision", f"sum by (decision) (increase(devcopilot_approvals_total[{R}]))", "{{decision}}", 0, 33, 6),
    bars("Agent steps p95", f"histogram_quantile(0.95, sum by (le, devcopilot_agent) (increase(devcopilot_agent_steps_bucket[{R}])))",
         "{{devcopilot_agent}}", 6, 33, 6, description="Steps an agent needed to finish. Near max_steps means loops."),
    {
        "id": next(_next_id), "type": "table", "title": "Firing alerts", "datasource": PROM,
        "gridPos": {"x": 12, "y": 33, "w": 12, "h": 8},
        "targets": [target('ALERTS{alertstate="firing"}', instant=True) | {"format": "table"}],
        "transformations": [{"id": "organize", "options": {
            "excludeByName": {"Time": True, "Value": True, "__name__": True, "alertstate": True},
            "indexByName": {"alertname": 0, "severity": 1}}}],
        "fieldConfig": {"defaults": {"noValue": "No alerts firing"}, "overrides": []},
        "options": {"showHeader": True},
    },

    row("Software: resources and service", 41),
    timeseries("CPU cores used by service", [target(f"sum by (service_name) (rate(process_cpu_time_seconds_total[{INTERVAL}]))", "{{service_name}}")],
               0, 42, 8, description="CPU seconds per second: 1 means one fully used core."),
    timeseries("Process memory by service", [target("sum by (service_name) (process_memory_usage_bytes)", "{{service_name}}")],
               8, 42, 8, unit="bytes"),
    timeseries("Sandbox memory used / total", [target("max by (state) (system_memory_usage_bytes)", "{{state}}")],
               16, 42, 8, color_overrides({"used": BLUE, "total": ORANGE}), unit="bytes",
               description="Inside the sandbox this is microVM memory, as used by SandboxMemoryHigh."),
    timeseries("Load average / CPUs", [target('max by (window) (system_cpu_load_average{window=~"1m|cpus"})', "{{window}}")],
               0, 50, 8, color_overrides({"1m": BLUE, "cpus": ORANGE})),
    timeseries("Workspace disk used / total", [target("max by (state) (devcopilot_workspace_disk_bytes)", "{{state}}")],
               8, 50, 8, color_overrides({"used": BLUE, "total": ORANGE}), unit="bytes",
               description="Usage of the filesystem containing the workspace, not the directory size."),
    timeseries("Ollama model memory: VRAM / RAM", [target("max by (model, memory) (devcopilot_ollama_model_memory_bytes)", "{{model}} {{memory}}")],
               16, 50, 8, unit="bytes", description="Loaded model memory from Ollama /api/ps."),

    row("LLM", 58),
    timeseries("Context utilization p95 by agent", [target(f"histogram_quantile(0.95, sum by (le, devcopilot_agent) (rate(gen_ai_client_context_utilization_bucket[{INTERVAL}])))", "{{devcopilot_agent}}")],
               0, 59, 8, unit="percentunit", description="Prompt tokens as a share of num_ctx."),
    timeseries("TTFT p95 by agent", [target(f"histogram_quantile(0.95, sum by (le, devcopilot_agent) (rate(gen_ai_server_time_to_first_token_seconds_bucket[{INTERVAL}])))", "{{devcopilot_agent}}")],
               8, 59, 8, unit="s"),
    timeseries("TPOT p95 by agent", [target(f"histogram_quantile(0.95, sum by (le, devcopilot_agent) (rate(gen_ai_server_time_per_output_token_seconds_bucket[{INTERVAL}])))", "{{devcopilot_agent}}")],
               16, 59, 8, unit="s"),

    row("Agents and team flow", 67),
    timeseries("Team rounds p50 / p95", [
        target(f"histogram_quantile(0.50, sum by (le) (rate(devcopilot_team_rounds_bucket[{INTERVAL}])))", "p50"),
        target(f"histogram_quantile(0.95, sum by (le) (rate(devcopilot_team_rounds_bucket[{INTERVAL}])))", "p95"),
    ], 0, 68, 8, color_overrides({"p50": BLUE, "p95": ORANGE})),
    stat("Revisions per team run", f'sum(increase(devcopilot_team_revisions_total[{R}])) / sum(increase(devcopilot_runs_total{{mode="team"}}[{R}]))',
         8, 68, no_value="No data", description="Selected range; no value when there are no team runs."),
    stat("Handoff delivered share", f'sum(increase(devcopilot_team_handoff_entities_total{{outcome="delivered"}}[{R}])) / sum(increase(devcopilot_team_handoff_entities_total{{outcome=~"delivered|lost"}}[{R}]))',
         12, 68, unit="percentunit", no_value="No data", description="Delivered / (delivered + lost), selected range."),
    stat("Duplicate calls per team run", f'sum(increase(devcopilot_team_duplicate_calls_total[{R}])) / sum(increase(devcopilot_runs_total{{mode="team"}}[{R}]))',
         16, 68, no_value="No data", description="Selected range; no value when there are no team runs."),
    timeseries("Memory context size p95 by mode", [target(f"histogram_quantile(0.95, sum by (le, mode) (rate(devcopilot_memory_context_size_bucket[{INTERVAL}])))", "{{mode}}")],
               0, 76, 12, color_overrides(MODE_COLORS), description="Characters of memory context supplied to the agent."),

    timeseries("Escalations to a human", [target(f"sum by (kind) (rate(devcopilot_escalations_total[{INTERVAL}]))", "{{kind}}")],
               12, 76, 12, description="Escalations per second by kind: write approval requests or unverified team answers."),

    row("Quality (continuous evaluation)", 84),
    timeseries("Online check pass rate", [target(f'sum by (check) (increase(devcopilot_online_eval_checks_total{{result="pass"}}[{INTERVAL}])) / sum by (check) (increase(devcopilot_online_eval_checks_total{{result=~"pass|fail"}}[{INTERVAL}]))', "{{check}}")],
               0, 85, 16, unit="percentunit", description="Pass / (pass + fail) for each check."),
    stat("Online fail share (last hour)", 'sum(increase(devcopilot_online_eval_checks_total{check=~"grounded|answers_question",result="fail"}[1h])) / sum(increase(devcopilot_online_eval_checks_total{check=~"grounded|answers_question"}[1h]))',
         16, 85, unit="percentunit", no_value="No data",
         thresholds=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 0.3}],
         description="Same share as OnlineQualityDrop: grounded and answers_question combined. Alert also requires at least 10 checks and 15 minutes above 30%."),

    row("Logs: warnings and errors (click trace_id to open the trajectory in Phoenix)", 93),
    {
        "id": next(_next_id), "type": "logs", "title": "", "datasource": LOKI,
        "gridPos": {"x": 0, "y": 94, "w": 24, "h": 12},
        "targets": [{"datasource": LOKI, "refId": "A",
                     "expr": '{service_name=~"devcopilot.*"} | severity_text=~"WARNING|ERROR"'}],
        "options": {"showTime": True, "wrapLogMessage": True, "enableLogDetails": True, "sortOrder": "Descending",
                    "dedupStrategy": "none", "prettifyLogMessage": False},
    },
]  # fmt: skip

dashboard = {
    "uid": "devcopilot",
    "title": "Dev Copilot",
    "tags": ["devcopilot"],
    "timezone": "browser",
    "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"},
    "schemaVersion": 39,
    "panels": panels,
}

out = Path(__file__).parent / "dashboards" / "devcopilot.json"
out.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}")
