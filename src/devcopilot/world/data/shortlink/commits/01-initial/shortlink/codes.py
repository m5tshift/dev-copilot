"""Base62 encoding of numeric link ids into short codes."""

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE = len(ALPHABET)


def encode(n: int) -> str:
    if n < 0:
        raise ValueError("id must be non-negative")
    if n == 0:
        return ALPHABET[0]
    chars = []
    while n:
        n, rem = divmod(n, BASE)
        chars.append(ALPHABET[rem])
    return "".join(reversed(chars))


def decode(code: str) -> int:
    n = 0
    for ch in code:
        idx = ALPHABET.find(ch)
        if idx == -1:
            raise ValueError(f"invalid character {ch!r} in code")
        n = n * BASE + idx
    return n
