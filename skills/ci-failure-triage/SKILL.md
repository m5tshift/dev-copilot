---
name: ci-failure-triage
description: Find since when CI fails, the error and the commit that started it. Use when the request is about failing, red or flaky CI runs or builds.
agents: [single, ci_investigator]
---

# Triage of a CI failure

1. `list_ci_runs`: find the FIRST failing run after the last green one, not just the latest.
   Note both run ids and their commits.
2. `get_ci_log` of the first failing run: copy the error line exactly (exception type,
   message, file and line). The last lines of the log usually hold it.
3. `git_log`: list the commits between the last green and the first red run. If there are
   several, the failing test or file in the error tells which commit to look at first.
4. `git_show` of the suspect commit: confirm it touches the file or function from the error.
5. Flaky or real? If a later run of the same commit is green, call it flaky and say so.

Report:
- first failing run id and commit, last green run id;
- the error, quoted;
- the suspect commit with the evidence that links it to the error;
- what is unverified (e.g. "did not run the tests locally").
