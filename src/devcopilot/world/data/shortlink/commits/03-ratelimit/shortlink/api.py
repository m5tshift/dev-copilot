"""Public API of the shortener: shorten a URL and resolve a code back."""

from shortlink.codes import decode, encode
from shortlink.expiry import is_expired
from shortlink.ratelimit import RateLimiter
from shortlink.store import LinkStore


class ShortLinkService:
    def __init__(self, store: LinkStore | None = None, limiter: RateLimiter | None = None):
        self.store = store or LinkStore()
        self.limiter = limiter or RateLimiter()

    def shorten(self, url: str, ttl_hours: int | None = None, client_id: str = "anonymous") -> str:
        self.limiter.check(client_id)
        if not url.startswith(("http://", "https://")):
            raise ValueError("only http(s) URLs are supported")
        return encode(self.store.add(url, ttl_hours))

    def resolve(self, code: str) -> str | None:
        link = self.store.get(decode(code))
        if link is None:
            return None
        if link.ttl_hours is not None and is_expired(link.created_at, link.ttl_hours):
            return None
        return link.url
