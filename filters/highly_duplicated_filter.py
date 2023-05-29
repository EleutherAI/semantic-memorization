from collections import Counter
from typing import Callable, List

import pandas as pd

def _concat_token_indices(token_indices: List[int], delimiter: str = '_') -> str:
    """
    Concatenates a list of tokens into a single string.

    Args:
        token_indices (List[int]): List of token indices to concatenate.
        delimiter (str, optional): Delimiter to use for concatenation. Defaults to '_'.
    
    Returns:
        str: Concatenated string of tokens indices.
    """
    return delimiter.join([str(t) for t in token_indices])

def generate_token_string_histogram(token_series: pd.Series, delimiter: str = '_') -> Counter:
    """
    Generates a histogram from a Pandas Series of token indices. The histogram is based on the concatenated strings of token indices.

    Args:
        token_series (pd.Series): Series of token indices.
        delimiter (str, optional): Delimiter to use for concatenation. Defaults to '_'.

    Returns:
        Counter: Histogram of strings of token indices.
    """
    return Counter(token_series.apply(lambda x: _concat_token_indices(x, delimiter=delimiter)))

def get_highly_duplicated_filter_func(histogram: Counter, frequency_threshold: int = 1, delimiter: str = '_') -> Callable[[List[int]], bool]:
    """
    Generates a filter function that checks if a list of token indices is highly duplicated.

    Args:
        histogram (Counter): Histogram of strings of token indices.
        frequency_threshold (int, optional): Frequency threshold to use for filtering. Defaults to 1.
        delimiter (str, optional): Delimiter to use for concatenation. Defaults to '_'.

    Returns:
        Callable[[List[int]], bool]: Filter function that checks if a list of token indices is highly duplicated.
    """
    def _highly_duplicated_filter_func(token_indices: List[int]) -> bool:
        """
        Checks if a list of token indices is highly duplicated.

        Args:
            token_indices (List[int]): List of token indices to check.

        Returns:
            bool: True if the list of token indices is highly duplicated, False otherwise.
        """
        token_string = _concat_token_indices(token_indices, delimiter=delimiter)
        return histogram[token_string] > frequency_threshold
    
    return _highly_duplicated_filter_func
