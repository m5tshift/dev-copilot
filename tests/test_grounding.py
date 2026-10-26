from devcopilot.grounding import extract_entities, ungrounded


def test_extracts_entities():
    text = "Commit 329311c broke shortlink/expiry.py:12, see #3 and https://x.dev/a. Run 104."
    assert extract_entities(text) == {"329311c", "shortlink/expiry.py", "#3", "https://x.dev/a", "run 104"}


def test_ignores_words_numbers_and_html_entities():
    assert extract_entities("deadline 2026 accepted &#39; 1234567 abcdefg") == set()


def test_short_hash_and_short_path_are_grounded_by_longer_forms():
    sources = ["commit 329311cb560cb2f40cbfa674afe2623b89024e3d", "shortlink/expiry.py:13: TypeError"]
    assert ungrounded("329311c changed expiry.py", sources) == []


def test_invented_url_and_issue_are_reported():
    sources = ["#3 Short links with TTL fail", "Created branch copilot/fix"]
    claims = "Fixed #3, see https://github.com/shortlink/repo/tree/copilot/fix (also #9)"
    assert ungrounded(claims, sources) == ["#9", "https://github.com/shortlink/repo/tree/copilot/fix"]


def test_partial_path_does_not_match_different_file():
    assert ungrounded("api.py", ["shortlink/rapi.py"]) == ["api.py"]


def test_hash_number_can_refer_to_a_ci_run():
    sources = ["run 104: FAILED commit 329311c"]
    assert ungrounded("CI runs #104 and run 105 failed", sources) == ["run 105"]
