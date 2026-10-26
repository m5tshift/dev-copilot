"""Statistics for small, noisy eval sets.

- pass@k: probability that at least one of k runs passes (Chen et al., 2021, the
  unbiased estimator from n >= k runs). Answers "can the agent do it at all?".
- pass^k: probability that all k runs pass (tau-bench, Yao et al., 2024).
  Answers "can a user rely on it?". For an agent this is the honest number.
- Bootstrap over TASKS, not runs: 3 runs of one task are not 3 independent
  observations. Resampling runs would give a falsely narrow interval.
- Cohen's kappa: agreement between two raters corrected for chance.
"""

import math
import random
from collections.abc import Sequence
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """n runs, c of them passed."""
    if n - c < k:
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    return comb(c, k) / comb(n, k)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def percentile(xs: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]: p95 of run latency etc."""
    if not xs:
        return float("nan")
    ordered = sorted(xs)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def bootstrap_ci(per_task: Sequence[float], level: float = 0.9, iters: int = 2000, seed: int = 0):
    """Confidence interval of the mean of per-task scores (resampling tasks)."""
    if not per_task:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(per_task)
    means = sorted(mean([per_task[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    tail = (1 - level) / 2
    return means[int(tail * iters)], means[int((1 - tail) * iters) - 1]


EPS = 1e-9  # float noise in differences of pass rates


def paired_delta(base: dict[str, float], new: dict[str, float], level: float = 0.9, iters: int = 2000, seed: int = 0):
    """Mean difference new - base over the common tasks, its CI and P(delta < 0).

    Paired: each task is compared with itself, which removes the (large) variation
    between easy and hard tasks from the comparison.
    """
    tasks = sorted(base.keys() & new.keys())
    diffs = [new[t] - base[t] for t in tasks]
    lo, hi = bootstrap_ci(diffs, level, iters, seed)
    rng = random.Random(seed)
    n = len(diffs)
    # -EPS, not 0: rates like 1/3 and 2/3 do not cancel exactly, and an exact tie came out
    # as -5.5e-17, i.e. "worse".
    worse = sum(mean([diffs[rng.randrange(n)] for _ in range(n)]) < -EPS for _ in range(iters)) / iters if n else 0.0
    return {"tasks": n, "delta": mean(diffs), "lo": lo, "hi": hi, "p_worse": worse}


def confusion(truth: Sequence[bool], pred: Sequence[bool]) -> dict[str, int]:
    pairs = list(zip(truth, pred, strict=True))
    return {
        "tp": sum(t and p for t, p in pairs),
        "fp": sum(p and not t for t, p in pairs),
        "fn": sum(t and not p for t, p in pairs),
        "tn": sum(not t and not p for t, p in pairs),
    }


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """kappa = (p_observed - p_chance) / (1 - p_chance).

    Two raters who say "pass" 90% of the time agree by chance in 82% of cases,
    so 85% raw agreement between them is almost nothing: kappa ~ 0.17.
    """
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    if chance == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - chance) / (1 - chance)
