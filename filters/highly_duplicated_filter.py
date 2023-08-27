import os

from collections import Counter
from typing import Callable, List

import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())


def _concat_token_indices(token_indices: List[int], delimiter: str = "_") -> str:
    """
    Concatenates a list of tokens into a single string.

    Args:
        token_indices (List[int]): List of token indices to concatenate.
        delimiter (str, optional): Delimiter to use for concatenation. Defaults to '_'.

    Returns:
        str: Concatenated string of tokens indices.
    """
    return delimiter.join(map(str, token_indices))


def generate_sequence_histogram(token_indices: pd.Series, delimiter: str = "_") -> Counter[str, int]:
    """
    Generates a histogram from a Pandas Series of token indices. The histogram is based on the concatenated strings of token indices.

    Args:
        token_index_sequences (pd.Series): Pandas Series of token indices.
        delimiter (str, optional): Delimiter to use for concatenation. Defaults to '_'.

    Returns:
        Counter[str, int]: Histogram of strings of token indices.
    """
    return Counter(token_indices.parallel_apply(lambda x: _concat_token_indices(x, delimiter=delimiter)))


def get_highly_duplicated_filter_func(
    histogram: Counter[str, int], frequency_threshold: int = 1, delimiter: str = "_"
) -> Callable[[pd.Series], bool]:
    """
    Generates a filter function that checks if a list of token indices is highly duplicated.

    Args:
        histogram (Counter[str, int]): Histogram of strings of token indices.
        frequency_threshold (int, optional): Frequency threshold to use for filtering. Defaults to 1.
        delimiter (str, optional): Delimiter to use for concatenation. Defaults to '_'.

    Returns:
        Callable[[pd.Series], bool]: Filter function that checks if a list of token indices is highly duplicated.
    """

    def _highly_duplicated_filter_func(row: pd.Series) -> bool:
        """
        Checks if a list of token indices is highly duplicated.

        Args:
            row (pd.Series): Pandas Series containing a list of token indices.

        Returns:
            bool: True if the list of token indices is highly duplicated, False otherwise.
        """
        token_indices = row["tokens"]
        token_string = _concat_token_indices(token_indices, delimiter=delimiter)
        return histogram[token_string] > frequency_threshold

    return _highly_duplicated_filter_func
