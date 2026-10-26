"""Retrieval evals of the memory (evals/memory_bench.yaml): Context Precision and Recall.

Episodes are stored with the real embedding model, then each query is recalled. A
threshold sweep over the similarity shows the precision/recall trade-off, so
episode_min_similarity is chosen by data, not by guess (like the judge's threshold).
The temporal facts get a scripted scenario: their correctness is deterministic.
Degradation: the same queries as memory grows with near-duplicate distractors, and pairs
of contradicting episodes a month apart (does recency put the newer answer first?).
"""

import hashlib
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from devcopilot.config import PROJECT_ROOT, get_settings
from devcopilot.grounding import extract_entities
from devcopilot.memory import EPISODES, FactStore, iso, now, open_memory

BENCH = PROJECT_ROOT / "evals" / "memory_bench.yaml"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
HEAD = "bench00"


def precision_recall(returned: list[str], relevant: list[str]) -> tuple[float, float]:
    rel = set(relevant)
    hits = len(rel & set(returned))
    # Returning nothing is perfect only when nothing is relevant; when something is, the
    # query failed. (The first version gave 1.0 there too and showed precision 87% at 0.6.)
    precision = hits / len(returned) if returned else (0.0 if rel else 1.0)
    recall = hits / len(rel) if rel else (1.0 if not returned else 0.0)
    return precision, recall


async def retrieval(k: int = 3) -> dict:
    data = yaml.safe_load(BENCH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as d:
        async with open_memory(Path(d)) as memory:
            for key, ep in data["episodes"].items():
                await memory.episodes.remember(key, ep["q"], ep["a"], HEAD, "bench")
            raw = []
            for query in data["queries"]:
                hits = await memory.episodes.store.asearch(EPISODES, query=query["q"], limit=k)
                raw.append((query, [(h.key, h.score or 0.0) for h in hits]))
    sweep = []
    for t in THRESHOLDS:
        pr = [precision_recall([key for key, s in hits if s >= t], q["relevant"]) for q, hits in raw]
        precision, recall = (sum(x[i] for x in pr) / len(pr) for i in (0, 1))
        sweep.append({"threshold": t, "precision": precision, "recall": recall})
    return {"k": k, "sweep": sweep, "queries": [{"q": q["q"], "relevant": q["relevant"], "hits": h} for q, h in raw]}


def temporal() -> list[tuple[str, bool]]:
    """A scripted life of issue #3 and run 107: does memory answer what is true NOW and THEN?"""
    t0 = datetime(2026, 10, 1, 12, tzinfo=UTC)
    with tempfile.TemporaryDirectory() as d:
        f = FactStore(Path(d) / "facts.db")
        f.upsert("#3", "status", "open", "get_issue", at=t0)
        f.upsert("#3", "assignee", "unassigned", "get_issue", at=t0)
        f.upsert("run 107", "ci_status", "failed", "list_ci_runs", at=t0)
        f.upsert("#3", "status", "open", "search_issues", at=t0 + timedelta(hours=3))  # seen again
        f.upsert("#3", "status", "closed", "write:update_issue", at=t0 + timedelta(hours=5))
        f.upsert("#3", "assignee", "bob", "get_issue", at=t0 + timedelta(hours=6))
        now = {(x.subject, x.predicate): x for x in f.about({"#3", "run 107"})}
        then = {(x.subject, x.predicate): x for x in f.about({"#3"}, as_of=t0 + timedelta(hours=4))}
        later = t0 + timedelta(days=3)
        return [
            ("current status is the latest write", now[("#3", "status")].object == "closed"),
            ("status as of +4h is the old one", then[("#3", "status")].object == "open"),
            ("re-observing the same value does not duplicate", len(f.history("#3", "status")) == 2),
            ("old version closed exactly when the new one starts",
             f.history("#3", "status")[0].valid_until == f.history("#3", "status")[1].valid_from),
            ("assignee changed by a later observation", now[("#3", "assignee")].object == "bob"),
            ("issue facts need re-check after their TTL", now[("#3", "status")].stale(later)),
            ("a finished CI run never goes stale", not now[("run 107", "ci_status")].stale(later)),
        ]  # fmt: skip


# ------------------------------------------------------------------ degradation

SIZES = (8, 50, 200)
OLD_HEAD = "0ld0bad"  # the repo state of the older, contradicting episodes
# A month apart: with episode_half_life_days=14 the older answer keeps ~0.23 of its weight,
# the newer one ~0.95, so recency outweighs any similarity edge below ~4x.
OLD_DAYS, NEW_DAYS = 30, 1


def episode_value(question: str, answer: str, head: str, mode: str, created_at: str) -> dict:
    """The value EpisodeStore.remember() writes, with an explicit created_at: an eval has to
    date episodes in the past, and remember() always stamps now. A test keeps the two equal."""
    s = get_settings()
    return {
        "text": f"{question}\n{answer[:500]}",
        "question": question,
        "answer": answer[: s.episode_answer_chars],
        "entities": sorted(extract_entities(answer)),
        "head": head,
        "mode": mode,
        "created_at": created_at,
    }


# Distractors: what months of use would leave in memory. Same shapes as the bench episodes
# (CI failures, owners, statuses, docs, ops, triage, features) and the same files, tests and
# people of `shortlink`, but about OTHER subjects, so the relevance labels of the queries stay
# true: none mentions expiry/expired links, redirects, aliases, deploys or releases, QR codes,
# HTTP 500 or "close all issues" (a test checks it). Hard negatives, not random noise: a
# near-duplicate about #17 competes with the episode about #3 for the same query.
_PEOPLE = ("alice", "bob", "carol", "dave", "erin")
_CI = (  # file, test, error, cause
    ("shortlink/codes.py", "test_roundtrip", "AssertionError", "the base62 alphabet lost the letter 'z'"),
    ("shortlink/codes.py", "test_invalid_character_rejected", "ValueError", "decode() accepts '-' since a refactor"),
    ("shortlink/ratelimit.py", "test_window_resets", "AssertionError", "the window is counted in ms, not seconds"),
    ("shortlink/ratelimit.py", "test_keys_are_independent", "KeyError", "the bucket key ignores the client id"),
    ("shortlink/store.py", "test_store_roundtrip", "sqlite3.OperationalError", "the table is created after the insert"),
    ("shortlink/api.py", "test_shorten_rejects_empty_url", "TypeError", "shorten() gets None from the form parser"),
    ("shortlink/codes.py", "test_negative_id_rejected", "AssertionError", "encode() wraps negative ids silently"),
)
_TOPICS = (  # issue title, cause, status code it shows up as
    ("rate limiter is per-process", "each worker keeps its own counters", "429"),
    ("base62 codes collide after a restart", "the id counter is not persisted", "409"),
    ("store locks under load", "every request opens a new SQLite connection", "503"),
    ("API returns 429 without Retry-After", "the header is never set in api.py", "429"),
    ("unknown codes return 200 with an empty body", "lookup() returns '' instead of None", "404"),
    ("shorten() accepts URLs without a scheme", "validation only checks the length", "400"),
    ("slow startup with a large database", "the store scans all links on start", "502"),
)
_DOCS = ("rate limits", "the base62 code format", "the SQLite storage choice", "API error codes",
         "configuration variables", "the on-call runbook", "database backups")  # fmt: skip
_OPS = (
    ("back up the link database", "Copy workspace/links.db with sqlite3 .backup; nightly by cron."),
    ("rotate the API logs", "logrotate keeps 7 days; the config is in docs/runbook.md."),
    ("run the tests locally", "pip install -e .[test] and python -m pytest -v, Python 3.12."),
    ("add a configuration variable", "Add it to shortlink/__init__.py with a default and document it."),
    ("reset a client's rate limit", "Delete its bucket in shortlink/ratelimit.py's store; there is no CLI."),
    ("migrate the SQLite schema", "Write a numbered SQL file; store.py applies it on start."),
    ("onboard a new developer", "Read docs/architecture.md and ADR 0001, then run the tests."),
)
_TRIAGE = (
    'a bug report: "{t}"',
    'a duplicate of #{d}: "{t}"',
    'a question: "{t}"',
    'a feature request close to "{t}"',
)
_FEATURES = ("custom domains", "click analytics", "bulk shortening", "password-protected links", "UTM tags",
             "API keys", "webhooks on new links", "link folders")  # fmt: skip


def _distractor(i: int) -> tuple[str, str, str]:
    """The i-th distractor: (key, question, answer). Pure function of i: no randomness."""
    n, run, j = 11 + i, 110 + i, i // 7
    who, other = _PEOPLE[i % 5], _PEOPLE[(i + 2) % 5]
    commit = hashlib.sha1(f"distractor {i}".encode()).hexdigest()[:7]
    title, cause, code = _TOPICS[(i + j) % len(_TOPICS)]
    match i % 7:
        case 0:
            file, test, error, why = _CI[j % len(_CI)]
            return (
                f"d-ci-{i}",
                f"Why is CI red on run {run}? {test} fails",
                f"Run {run} fails with {error} in {file}:{10 + j % 40} since commit {commit}: {why}.",
            )
        case 1:
            file, test, error, why = _CI[(j + 3) % len(_CI)]
            return (f"d-test-{i}", f"What broke {test}?", f"{error} in {test} since {commit}: {why} ({file}).")
        case 2:
            return (
                f"d-owner-{i}",
                f"Who owns issue #{n}?",
                f"#{n} ({title}) is assigned to {who}; {other} works on #{n + 7}.",
            )
        case 3:
            state = ("open", "closed")[j % 2]
            return (
                f"d-status-{i}",
                f"Status of the {code} errors: {title}",
                f"#{n} is {state}, assigned to {who}, p{1 + j % 3}; cause: {cause}.",
            )
        case 4:
            topic = _DOCS[j % len(_DOCS)]
            answer = (
                f"Nowhere yet: #{n} asks {who} to document {topic}."
                if j % 2
                else f"In docs/architecture.md, section on {topic}; {who} updated it in {commit}."
            )
            return f"d-docs-{i}", f"Where can I read about {topic}?", answer
        case 5:
            op, how = _OPS[j % len(_OPS)]
            return f"d-ops-{i}", f"How do we {op}?", f"{how} {who} wrote it up in #{n}."
        case _:
            if j % 2:
                feature = _FEATURES[j % len(_FEATURES)]
                answer = (
                    f"No: #{n} asks for {feature}, labelled feature, unassigned."
                    if j % 4 == 1
                    else f"Yes, #{n} added {feature}; it is closed."
                )
                return f"d-feature-{i}", f"Do we support {feature}?", answer
            kind = _TRIAGE[j // 2 % len(_TRIAGE)].format(t=title, d=n - 3)
            return f"d-triage-{i}", f"Triage #{n}", f"#{n} is {kind}; labelled bug, assigned to {who}."  # fmt: skip


def distractors(count: int) -> list[tuple[str, str, str]]:
    """Nested by construction: the first 42 of 192 are the 42 used at the smaller size."""
    return [_distractor(i) for i in range(count)]


def _rank(episodes, key: str) -> int | None:
    return next((i + 1 for i, e in enumerate(episodes) if e.key == key), None)


async def _growth(memory, data: dict, sizes: list[int], k: int) -> list[dict]:
    store, base = memory.episodes.store, data["episodes"]
    t = iso(now())
    for key, ep in base.items():
        await store.aput(EPISODES, key, episode_value(ep["q"], ep["a"], HEAD, "bench", t))
    rows, stored = [], 0
    for size in sizes:
        extra = distractors(size - len(base))
        for key, q, a in extra[stored:]:  # memory grows: the smaller memory is a prefix of the larger
            await store.aput(EPISODES, key, episode_value(q, a, HEAD, "bench", t))
        stored = len(extra)
        queries, chars = [], []
        for query in data["queries"]:
            # recall() and context() are what the agents get: the configured threshold,
            # recency, and (in context) the compaction to the character budget.
            hits = await memory.episodes.recall(query["q"], HEAD, k)
            chars.append(len(await memory.context(query["q"], HEAD)))
            queries.append({"q": query["q"], "relevant": query["relevant"],
                            "returned": [(e.key, round(e.score, 4)) for e in hits]})  # fmt: skip
        pr = [precision_recall([key for key, _ in q["returned"]], q["relevant"]) for q in queries]
        returned = [key for q in queries for key, _ in q["returned"]]
        rows.append({
            "episodes": size,
            "distractors": stored,
            "precision": sum(p for p, _ in pr) / len(pr),
            "recall": sum(r for _, r in pr) / len(pr),
            # Precision mixes two kinds of noise; this one is what growth adds.
            "distractor_share": sum(key not in base for key in returned) / len(returned) if returned else 0.0,
            "context_chars_mean": sum(chars) / len(chars),
            "context_chars_max": max(chars),
            "queries": queries,
        })  # fmt: skip
    return rows


async def _staleness(memory, data: dict, k: int) -> dict:
    store, t = memory.episodes.store, now()
    for key, ep in data["episodes"].items():  # the usual background
        await store.aput(EPISODES, key, episode_value(ep["q"], ep["a"], HEAD, "bench", iso(t)))
    for i, pair in enumerate(data["stale"]):
        old_at, new_at = iso(t - timedelta(days=OLD_DAYS)), iso(t - timedelta(days=NEW_DAYS))
        await store.aput(EPISODES, f"stale-{i}-old", episode_value(pair["q"], pair["old"], OLD_HEAD, "bench", old_at))
        await store.aput(EPISODES, f"stale-{i}-new", episode_value(pair["q"], pair["new"], HEAD, "bench", new_at))
    cases = []
    for i, pair in enumerate(data["stale"]):
        recalled = await memory.episodes.recall(pair["query"], HEAD, k)
        # Raw similarity order, without recency: what a plain vector search would show.
        hits = await store.asearch(EPISODES, query=pair["query"], limit=len(data["episodes"]) + 2 * len(data["stale"]))
        by_similarity = [h.key for h in hits]
        old_rank, new_rank = _rank(recalled, f"stale-{i}-old"), _rank(recalled, f"stale-{i}-new")
        cases.append({
            "query": pair["query"],
            "new_rank": new_rank,
            "old_rank": old_rank,
            "old_outdated": recalled[old_rank - 1].outdated if old_rank else None,
            "newer_first": new_rank is not None and (old_rank is None or new_rank < old_rank),
            "newer_first_by_similarity": by_similarity.index(f"stale-{i}-new") < by_similarity.index(f"stale-{i}-old"),
        })  # fmt: skip
    n = len(cases)
    return {
        "pairs": n,
        "old_age_days": OLD_DAYS,
        "new_age_days": NEW_DAYS,
        "newer_first": sum(c["newer_first"] for c in cases) / n,
        "newer_first_by_similarity": sum(c["newer_first_by_similarity"] for c in cases) / n,
        "old_returned": sum(c["old_rank"] is not None for c in cases) / n,
        "old_returned_flagged": sum(c["old_outdated"] is True for c in cases) / n,
        "old_returned_unflagged": sum(c["old_outdated"] is False for c in cases) / n,
        "cases": cases,
    }


async def degradation(sizes=SIZES, k: int = 3, open_=open_memory, data: dict | None = None) -> dict:
    """How retrieval degrades as memory grows, and whether an older contradicting episode
    loses to the newer one. Each part gets its own fresh memory.

    open_: async context manager factory taking a directory (tests pass one with a fake
    embedding); data: the bench (tests pass a small one).
    """
    data = data or yaml.safe_load(BENCH.read_text(encoding="utf-8"))
    sizes = sorted(set(sizes))
    if sizes[0] < len(data["episodes"]):
        raise ValueError(f"sizes must be >= {len(data['episodes'])}: the bench episodes stay in every size")
    with tempfile.TemporaryDirectory() as d:
        async with open_(Path(d) / "growth") as memory:
            growth = await _growth(memory, data, sizes, k)
        async with open_(Path(d) / "stale") as memory:
            stale = await _staleness(memory, data, k)
    return {"k": k, "min_similarity": get_settings().episode_min_similarity, "sizes": growth, "staleness": stale}
