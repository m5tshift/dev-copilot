---
name: root-cause-analysis
description: Explain WHY code misbehaves, down to a file line and the commit that introduced it. Use for bugs, errors, regressions and "why does X happen" questions.
agents: [single, code_analyst]
---

# Root cause analysis

1. Start from the symptom: the error text, the failing test or the behavior users see.
   `search_code` for the function, message or config key it mentions.
2. `read_file` around the hit. Find the exact line that produces the wrong result and
   explain the mechanism (wrong type, off-by-one, missing check), not just the location.
3. `git_log` for that file, then `git_show` of recent commits: find the commit that
   introduced the line. Quote the changed line from the diff.
4. Confirm with `run_tests` when a test covers it: a failing test with the same error
   is evidence; a passing one weakens the hypothesis, say so.
5. One cause can have several symptoms: if the cause explains other reported problems
   (issues, other failing runs), mention them; if you only suspect it, say "possibly".

Report: cause (`path:line` + mechanism), introducing commit, evidence (test, diff),
alternative explanations you ruled out and how.
