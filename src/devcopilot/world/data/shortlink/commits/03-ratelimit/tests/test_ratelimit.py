import pytest

from shortlink.ratelimit import RateLimiter, RateLimitExceeded


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_allows_up_to_limit_then_blocks():
    limiter = RateLimiter(max_requests=3, window_seconds=60, clock=FakeClock())
    for _ in range(3):
        limiter.check("1.2.3.4")
    with pytest.raises(RateLimitExceeded):
        limiter.check("1.2.3.4")


def test_window_resets():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=1, window_seconds=60, clock=clock)
    limiter.check("a")
    clock.t = 61
    limiter.check("a")


def test_keys_are_independent():
    limiter = RateLimiter(max_requests=1, window_seconds=60, clock=FakeClock())
    limiter.check("a")
    limiter.check("b")
