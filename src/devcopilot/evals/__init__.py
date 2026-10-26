"""Phase 4: evals of the agent and the team.

    evals/dataset.yaml --> run.py --> results/evals/*.jsonl --> judge.py (adds verdicts)
                                                             \\-> report.py (scores, compares)
    labels.py: human labels, calibration of the judge; components.py: gate and supervisor alone;
    online.py: reference-free checks of real traces from Phoenix.

Runs store raw results only. Deterministic checks (checks.py) are recomputed from
the current dataset at report time, so fixing an expectation never needs a rerun.
LLM verdicts are expensive and are stored, together with a hash of the rubric they used.
"""
