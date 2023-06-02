from typing import Iterable, Mapping

def token_frequencies(tokens: Iterable[int]) -> Mapping[int, int]:
    """Return a dictionary mapping tokens to their frequencies.
    
    Args:
        tokens (Iterable[int]): A sequence of tokens (e.g., a list of integers).
    
    Returns:
        A dictionary mapping tokens to their frequencies.

    >>> token_frequencies([1, 2, 3, 1, 2, 1])
    """
    frequencies = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies

def merge_token_frequencies(freq1: Mapping[int, int], freq2: Mapping[int, int]) -> Mapping[int, int]:
    """Return the merge of two token frequency dictionaries.

    Args:
        freq1 (dict[int, int]): A dictionary mapping tokens to their frequencies.
        freq2 (dict[int, int]): A dictionary mapping tokens to their frequencies.
    
    Returns:
        A dictionary mapping tokens to their frequencies.

    >>> merge_token_frequencies({1: 2, 2: 1}, {2: 1, 3: 1})
    """
    merged = {}
    for token, freq in freq1.items():
        merged[token] = merged.get(token, 0) + freq
    for token, freq in freq2.items():
        merged[token] = merged.get(token, 0) + freq
    return merged