"""Link expiry rules. All timestamps are timezone-aware UTC."""

from datetime import datetime, timedelta


def expires_at(created_at: datetime, ttl_hours: int) -> datetime:
    return created_at + timedelta(hours=ttl_hours)


def is_expired(created_at: datetime, ttl_hours: int, now: datetime | None = None) -> bool:
    """Return True if the link has expired. `now` can be injected in tests."""
    now = now or datetime.now()
    return now >= expires_at(created_at, ttl_hours)
