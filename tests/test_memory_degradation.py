"""Memory degradation eval (evals/memory_bench.degradation) without Ollama: real SQLite
stores with a hashed bag-of-words embedding. Checks the mechanics, not the quality: the
quality numbers come from the real embedding model on the PC."""

import asyncio
import json
import re
import zlib
from contextlib import asynccontextmanager

import pytest
from langgraph.store.sqlite.aio import AsyncSqliteStore

from devcopilot.config import get_settings
from devcopilot.evals import memory_bench
from devcopilot.evals.memory_bench import degradation, distractors, episode_value
from devcopilot.memory import EPISODES, EpisodeStore, FactStore, Memory

DIMS = 64


def hashed_embed(texts: list[str]) -> list[list[float]]:
    # crc32, not hash(): str hashes are salted per process, the vectors must not be.
    out = []
    for t in texts:
        v = [0.01] * DIMS
        for w in re.findall(r"[#\w]+", t.lower()):
            v[zlib.crc32(w.encode("utf-8")) % DIMS] += 1.0
        out.append(v)
    return out


def fake_open(counts: list[int] | None = None):
    """open_memory's stand-in; records how many episodes each memory ended with."""

    @asynccontextmanager
    async def open_(directory):
        directory.mkdir(parents=True, exist_ok=True)
        index = {"dims": DIMS, "embed": hashed_embed, "fields": ["text"]}
        async with AsyncSqliteStore.from_conn_string(str(directory / "ep.db"), index=index) as store:
            await store.setup()
            yield Memory(EpisodeStore(store), FactStore(directory / "facts.db"))
            if counts is not None:
                counts.append(len(await store.asearch(EPISODES, limit=10_000)))

    return open_


@pytest.fixture
def no_threshold(monkeypatch):
    # The fake embedding is far from the real model's scale: let every hit through, so the
    # tests see the ranking mechanics instead of an empty result.
    monkeypatch.setenv("DEVCOPILOT_EPISODE_MIN_SIMILARITY", "0")
    get_settings.cache_clear()


def test_distractors_are_deterministic_nested_and_distinct():
    big = distractors(192)
    assert distractors(42) == big[:42]  # a larger memory contains the smaller one
    assert big == distractors(192)  # no randomness
    assert len({k for k, _, _ in big}) == len({q + a for _, q, a in big}) == 192


def test_distractors_keep_the_relevance_labels_true():
    """A distractor about a query's subject would be relevant, and counting it as noise
    would make the growth curve lie. They also stay off the issue numbers of the bench."""
    forbidden = ("expir", "redirect", "alias", "qr", "deploy", "release", " 500", "close all", "close every")
    for key, q, a in distractors(192):
        text = f"{q} {a}".lower()
        assert not [w for w in forbidden if w in text], key
        assert all(int(n) >= 11 for n in re.findall(r"#(\d+)", text)), key


def test_episode_value_has_the_schema_of_remember(tmp_path):
    async def go():
        async with fake_open()(tmp_path) as memory:
            answer = "Run 104 fails in shortlink/expiry.py since 329311c, see #3. " * 30
            await memory.episodes.remember("a", "why is CI red", answer, "abc1234", "team")
            (item,) = await memory.episodes.store.asearch(EPISODES, limit=5)
            return item.value, episode_value("why is CI red", answer, "abc1234", "team", item.value["created_at"])

    stored, ours = asyncio.run(go())
    assert stored == ours


def test_growth_keeps_the_bench_and_adds_distractors(tmp_path, no_threshold):
    counts: list[int] = []
    result = asyncio.run(degradation((22, 8, 15), k=3, open_=fake_open(counts)))
    rows = result["sizes"]
    assert [r["episodes"] for r in rows] == [8, 15, 22]  # sorted, growing
    assert [r["distractors"] for r in rows] == [0, 7, 14]
    bench = memory_bench.yaml.safe_load(memory_bench.BENCH.read_text(encoding="utf-8"))
    assert counts == [22, len(bench["episodes"]) + 2 * len(bench["stale"])]  # growth memory, staleness memory
    assert rows[0]["distractor_share"] == 0.0  # nothing to intrude yet
    for r in rows:
        assert 0 <= r["precision"] <= 1 and 0 <= r["recall"] <= 1
        assert len(r["queries"]) == len(bench["queries"])
        assert all(len(q["returned"]) <= 3 for q in r["queries"])
        # With no threshold every query gets k episodes, so every reader gets a block.
        assert 0 < r["context_chars_mean"] <= r["context_chars_max"]
    json.dumps(result)  # goes into --out as is


def test_sizes_below_the_bench_are_refused(tmp_path):
    with pytest.raises(ValueError, match="sizes must be"):
        asyncio.run(degradation((5, 50), open_=fake_open()))


def test_recency_ranks_the_newer_answer_above_a_more_similar_older_one(no_threshold):
    """The older episode repeats the query words, so raw similarity prefers it; a month of
    age (two half-lives) must still put the newer answer first and flag the older one."""
    data = {
        "episodes": {"deploy": {"q": "How do we deploy shortlink?", "a": "From main after green CI."}},
        "queries": [{"q": "how to deploy", "relevant": ["deploy"]}],
        "stale": [{"q": "who owns issue #8", "query": "who owns issue #8",
                   "old": "who owns issue #8: nobody owns issue #8", "new": "carol"}],
    }  # fmt: skip
    st = asyncio.run(degradation((1,), k=3, open_=fake_open(), data=data))["staleness"]
    (case,) = st["cases"]
    assert case["newer_first_by_similarity"] is False  # the trap is real
    assert case["new_rank"] == 1 and case["old_rank"] > 1 and case["old_outdated"] is True
    assert st["newer_first"] == 1.0 and st["old_returned_flagged"] == 1.0 and st["old_returned_unflagged"] == 0.0
