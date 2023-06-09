from typing import Iterable, Mapping

def seq_token_frequencies(tokens: Iterable[int], token_frequencies: Mapping[int, int]) -> Mapping[int, int]:
    """Return a dictionary mapping tokens to their frequencies in Corpus.
    
    Args:
        tokens (Iterable[int]): A sequence of tokens (e.g., a list of integers).
        token_frequencies (Mapping[int, int]): A dictionary mapping tokens to their
            number of occurances in a corpus.
    
    Returns:
        A dictionary mapping tokens to their frequencies in the sequence of tokens.

    >>> tokens = [1233, 1233, 1234, 1235, 1235, 1235, 1236, 1236, 1236, 1236]
    >>> token_frequencies = {1233: 2, 1234: 1, 1235: 3, 1236: 4}
    >>> seq_token_frequencies(tokens, token_frequencies)
    """
    frequencies = {}
    for token in tokens:
        frequencies[token] = token_frequencies.get(token, 0)
    return frequencies