"""Public API of the shortener: shorten a URL and resolve a code back."""

from shortlink.codes import decode, encode
from shortlink.store import LinkStore


class ShortLinkService:
    def __init__(self, store: LinkStore | None = None):
        self.store = store or LinkStore()

    def shorten(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("only http(s) URLs are supported")
        return encode(self.store.add(url))

    def resolve(self, code: str) -> str | None:
        return self.store.get(decode(code))
