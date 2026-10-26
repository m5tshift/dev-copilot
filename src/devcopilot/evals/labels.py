"""Human labels and calibration of the judge against them.

A label says whether an answer meets the task's rubric: the same question the judge
answers, so the two can be compared. Labels are keyed by a hash of (task, answer):
they survive rerunning the judge and apply to the same answer in any results file.

They live in evals/labels/<annotator>.jsonl and are committed: labeled data is the
most expensive part of an eval set.

Calibration reports agreement and Cohen's kappa of the judge, and of the substring
check for comparison. Rough reading of kappa: < 0.4 poor, 0.4-0.6 moderate,
0.6-0.8 substantial, > 0.8 almost perfect. Two humans rarely exceed 0.8 either.
"""

import hashlib
import json
import random
from datetime import UTC, datetime

from devcopilot.config import PROJECT_ROOT
from devcopilot.evals.checks import check
from devcopilot.evals.dataset import Task
from devcopilot.evals.report import score
from devcopilot.evals.stats import cohen_kappa, confusion, mean

LABELS = PROJECT_ROOT / "evals" / "labels"


def answer_key(task_id: str, answer: str) -> str:
    return hashlib.sha256(f"{task_id}\n{answer}".encode()).hexdigest()[:16]


def load_labels(annotator: str) -> dict[str, dict]:
    path = LABELS / f"{annotator}.jsonl"
    if not path.exists():
        return {}
    labels = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {lab["key"]: lab for lab in labels}  # a later label of the same answer wins


def save_label(annotator: str, key: str, task_id: str, label: bool, note: str = "") -> None:
    LABELS.mkdir(parents=True, exist_ok=True)
    entry = {"key": key, "task": task_id, "label": label, "note": note,
             "labeled_at": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    with (LABELS / f"{annotator}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def unique_answers(records: list[dict], tasks: dict[str, Task]) -> dict[str, dict]:
    """One record per distinct (task, answer): identical answers need one label."""
    out = {}
    for r in records:
        if not r["error"] and r["task"] in tasks:
            out.setdefault(answer_key(r["task"], r["result"]["answer"]), r)
    return out


def sample_for_labeling(records: list[dict], tasks: dict[str, Task], n: int, skip: set[str], seed: int = 0):
    """Spread over tasks and configurations: every task once before any task twice.

    Random within that order, and independent of any verdict: sampling only the
    answers the judge failed would make its errors look rarer than they are.
    """
    by_task: dict[str, list[tuple[str, dict]]] = {}
    for key, r in unique_answers(records, tasks).items():
        if key not in skip:
            by_task.setdefault(r["task"], []).append((key, r))
    rng = random.Random(seed)
    for items in by_task.values():
        rng.shuffle(items)
    order = list(by_task)
    rng.shuffle(order)
    picked = []
    while len(picked) < n and any(by_task.values()):
        for task_id in order:
            if by_task[task_id] and len(picked) < n:
                picked.append(by_task[task_id].pop())
    return picked


def calibrate(records: list[dict], tasks: dict[str, Task], labels: dict[str, dict], judge: str) -> dict:
    rows = []
    for key, r in unique_answers(records, tasks).items():
        if key not in labels:
            continue
        task = tasks[r["task"]]
        s = score(r, task, judge)
        facts, _ = check(task, r["result"])
        verdict = r["judges"].get(judge) or {}
        fresh = verdict.get("rubric") == task.rubric_hash
        rows.append({"key": key, "task": r["task"], "human": labels[key]["label"], "judge": s["judge"],
                     "judge_p": verdict.get("p") if fresh else None,
                     "facts": facts["facts"], "note": labels[key].get("note", "")})  # fmt: skip

    def compare(rater: str) -> dict:
        pairs = [x for x in rows if x[rater] is not None]
        human, other = [x["human"] for x in pairs], [x[rater] for x in pairs]
        return {
            "n": len(pairs),
            "agreement": mean([h == o for h, o in zip(human, other, strict=True)]),
            "kappa": cohen_kappa(human, other),
            # Rows: what the human said; tp = both pass, fp = rater passes what the human failed.
            **confusion(human, other),
            "disagreements": [x for x in pairs if x["human"] != x[rater]],
        }

    # A judge with probabilities (Jev): agreement at each threshold, to choose one from labels.
    sweep = []
    if any(x["judge_p"] is not None for x in rows):
        for t in (0.3, 0.5, 0.7, 0.9):
            for x in rows:
                x[f"p>={t}"] = None if x["judge_p"] is None else x["judge_p"] >= t
            sweep.append({"threshold": t, **compare(f"p>={t}")})
    return {
        "labeled": len(rows),
        "human_pass_rate": mean([x["human"] for x in rows]),
        "judge": compare("judge"),
        "facts": compare("facts"),
        "sweep": sweep,
    }


def inter_annotator(a: dict[str, dict], b: dict[str, dict]) -> dict:
    common = sorted(a.keys() & b.keys())
    x, y = [a[k]["label"] for k in common], [b[k]["label"] for k in common]
    return {
        "n": len(common),
        "agreement": mean([p == q for p, q in zip(x, y, strict=True)]),
        "kappa": cohen_kappa(x, y),
        "disagreements": [(a[k]["task"], k) for k in common if a[k]["label"] != b[k]["label"]],
    }
