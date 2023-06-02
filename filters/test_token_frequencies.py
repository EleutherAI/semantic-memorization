from typing import Iterable, Mapping
from .token_frequencies import token_frequencies, merge_token_frequencies

def test_token_frequencies():
    tokens = [1233, 1233, 1234, 1235, 1235, 1235, 1236, 1236, 1236, 1236]
    expected = {1233: 2, 1234: 1, 1235: 3, 1236: 4}
    assert token_frequencies(tokens) == expected

def test_merge_token_frequencies():
    freqs1 = {1233: 2, 1234: 1, 1235: 3, 1236: 4}
    freqs2 = {1233: 3, 1234: 2, 1235: 1, 1236: 0}
    expected = {1233: 5, 1234: 3, 1235: 4, 1236: 4}
    assert merge_token_frequencies(freqs1, freqs2) == expected


if __name__ == '__main__':
    test_token_frequencies()
    test_merge_token_frequencies()