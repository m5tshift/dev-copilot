"""Link expiry rules. All timestamps are timezone-aware UTC."""

from datetime import datetime, timedelta, timezone


def expires_at(created_at: datetime, ttl_hours: int) -> datetime:
    return created_at + timedelta(hours=ttl_hours)


def is_expired(created_at: datetime, ttl_hours: int) -> bool:
    return datetime.now(timezone.utc) >= expires_at(created_at, ttl_hours)
