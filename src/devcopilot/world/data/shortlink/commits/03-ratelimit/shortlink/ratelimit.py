"""Fixed-window per-client rate limiting.

NOTE: counters live in process memory. With N gunicorn workers the effective
limit is up to N * max_requests per window.
"""

import time

MAX_REQUESTS = 30
WINDOW_SECONDS = 60


class RateLimitExceeded(Exception):
    pass


class RateLimiter:
    def __init__(
        self,
        max_requests: int = MAX_REQUESTS,
        window_seconds: float = WINDOW_SECONDS,
        clock=time.monotonic,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clock = clock
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, key: str) -> None:
        now = self.clock()
        start, count = self._windows.get(key, (now, 0))
        if now - start >= self.window_seconds:
            start, count = now, 0
        if count >= self.max_requests:
            raise RateLimitExceeded(f"rate limit exceeded for {key}")
        self._windows[key] = (start, count + 1)
