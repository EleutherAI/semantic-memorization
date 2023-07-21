import pandas as pd

from .highly_duplicated_filter import get_highly_duplicated_filter_func, generate_sequence_histogram


def test_highly_duplicated_filter_on_seen_indices():
    data = pd.Series([[1, 2, 3], [4, 5, 6], [4, 5, 6]])
    histogram = generate_sequence_histogram(data)
    threshold = 1
    filter_func = get_highly_duplicated_filter_func(histogram, frequency_threshold=threshold)

    sample = [4, 5, 6]
    assert filter_func(sample) == True


def test_highly_duplicated_filter_on_unseen_indices():
    data = pd.Series([[1, 2, 3], [4, 5, 6], [4, 5, 6]])
    histogram = generate_sequence_histogram(data)
    threshold = 1
    filter_func = get_highly_duplicated_filter_func(histogram, frequency_threshold=threshold)

    sample = [7, 8, 9]
    assert filter_func(sample) == False


def test_highly_duplicated_filter_on_infrequent_indices():
    data = pd.Series([[1, 2, 3], [4, 5, 6], [4, 5, 6]])
    histogram = generate_sequence_histogram(data)
    threshold = 2
    filter_func = get_highly_duplicated_filter_func(histogram, frequency_threshold=threshold)

    sample = [4, 5, 6]
    assert filter_func(sample) == False
