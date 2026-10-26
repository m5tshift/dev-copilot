"""Deterministic grounding check: did the agent actually see what it claims?

LLMs invent specifics that look authoritative: a commit hash, an issue number,
a URL. These are easy to extract with regexes and compare against the text
the agent really received from tools. Anything the agent mentions but never
saw is "ungrounded": not necessarily wrong, but unverified.

It is cheap, deterministic and catches exactly the failure from phase 1
(an invented GitHub link written into a tracker comment). It does not check
meaning: "the bug is in X" with a correct path but a wrong explanation passes.
"""

import re
from collections.abc import Iterable

PATTERNS = {
    "url": re.compile(r"https?://[^\s)>\]\"'`]+"),
    # 7-40 hex chars with at least one digit and one letter: excludes words and plain numbers.
    "commit": re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b"),
    "issue": re.compile(r"(?<![\w&])#\d+\b"),
    "file": re.compile(r"\b[\w./-]+\.(?:py|md|json|log|toml)\b"),
    # CI runs are numbered too: "run 104". Normalized to "run 104".
    "run": re.compile(r"\brun\s+#?(\d+)\b", re.IGNORECASE),
}


def extract_entities(text: str) -> set[str]:
    entities = {m.rstrip(".,;:") for name, p in PATTERNS.items() if name != "run" for m in p.findall(text)}
    return entities | {f"run {n}" for n in PATTERNS["run"].findall(text)}


def _matches(entity: str, seen: str) -> bool:
    if entity == seen:
        return True
    # "329311c" vs full hash, "expiry.py" vs "shortlink/expiry.py".
    if PATTERNS["commit"].fullmatch(entity):
        return seen.startswith(entity) or entity.startswith(seen)
    if PATTERNS["file"].fullmatch(entity):
        return seen.endswith("/" + entity) or entity.endswith("/" + seen)
    # "#104" is ambiguous: an issue or a CI run. Accept either.
    if entity.startswith("#"):
        return seen == f"run {entity[1:]}"
    return False


def ungrounded(claims: str, sources: Iterable[str]) -> list[str]:
    """Entities mentioned in `claims` that appear in none of `sources`."""
    seen = set().union(*(extract_entities(s) for s in sources))
    return sorted(e for e in extract_entities(claims) if not any(_matches(e, s) for s in seen))
