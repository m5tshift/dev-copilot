import pytest

from shortlink.codes import decode, encode


@pytest.mark.parametrize("n", [0, 1, 61, 62, 3843, 10**9])
def test_roundtrip(n):
    assert decode(encode(n)) == n


def test_negative_id_rejected():
    with pytest.raises(ValueError):
        encode(-1)


def test_invalid_character_rejected():
    with pytest.raises(ValueError):
        decode("ab-c")
