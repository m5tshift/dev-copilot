from datetime import datetime, timedelta, timezone

from shortlink.expiry import expires_at, is_expired


def test_expires_at_adds_ttl():
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert expires_at(created, 2) == datetime(2026, 1, 1, 2, tzinfo=timezone.utc)


def test_fresh_link_not_expired():
    assert not is_expired(datetime.now(timezone.utc), ttl_hours=1)


def test_old_link_expired():
    created = datetime.now(timezone.utc) - timedelta(hours=5)
    assert is_expired(created, ttl_hours=1)
