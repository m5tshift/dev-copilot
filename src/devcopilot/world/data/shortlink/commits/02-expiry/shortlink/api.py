"""Public API of the shortener: shorten a URL and resolve a code back."""

from shortlink.codes import decode, encode
from shortlink.expiry import is_expired
from shortlink.store import LinkStore


class ShortLinkService:
    def __init__(self, store: LinkStore | None = None):
        self.store = store or LinkStore()

    def shorten(self, url: str, ttl_hours: int | None = None) -> str:
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
