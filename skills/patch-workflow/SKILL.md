---
name: patch-workflow
description: Make a minimal code fix on a copilot/* branch and prove it with tests. Use when asked to fix, patch or prepare a branch.
agents: [single, fixer]
---

# Patch workflow

1. `read_file` the exact lines to change: propose_patch replaces text, so copy it
   character for character, including indentation.
2. Change as little as possible: fix the cause, do not refactor or reformat nearby code.
3. `propose_patch` with a short branch name describing the fix. It asks a human for
   approval; if rejected, stop and report that the fix was not made.
4. `run_tests` on the new branch. Green: report it. Red: report the failing test and
   the error; do not claim the fix works.

Report: branch name, the changed lines (before -> after), the test result on the branch.
