"""Exact end-of-day random words of the reference engine.

The reference (Python and C++) builds random.Random((seed * 1_000_003) ^ day) at the end of every day and draws
random() once per EMPTY tile (farm 0 row-major, then farm 1), then choice(sorted(SHOPS)) on shop days. The JAX
engine consumes the raw MT19937 outputs of that generator (`words`, uint32[624]); getrandbits(32) returns exactly
genrand_uint32(), so the words are available from the seed alone - no need to read them back from the engine.
The C++ computes the key in uint64 arithmetic; Kaggle seeds are < 2^31 so no wraparound occurs, but we mirror it.
"""
import random
import numpy as np

try:
    from . import consts as C
except ImportError:
    import consts as C


def day_words(seed, day, n=C.NW_WORDS):
    key = ((int(seed) * 1000003) & (2 ** 64 - 1)) ^ int(day)
    r = random.Random(key)
    x = r.getrandbits(32 * n)                 # words little-endian: first output = lowest 32 bits
    return np.frombuffer(x.to_bytes(4 * n, "little"), dtype="<u4").astype(np.uint32)


def game_words(seed, days=30):
    """uint32[days, 624]: words for every end-of-day of one game."""
    return np.stack([day_words(seed, d) for d in range(days)])


def batch_words(seeds, days=30):
    """uint32[days, B, 624]."""
    return np.stack([game_words(s, days) for s in seeds], axis=1)


def _selftest():
    r = random.Random(123 * 1000003 ^ 5)
    w = [r.getrandbits(32) for _ in range(10)]
    assert list(day_words(123, 5)[:10]) == w
    return True


if __name__ == "__main__":
    print(_selftest())
