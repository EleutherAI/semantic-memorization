from typing import Callable

import pandas as pd

"""
This file manages the modeling-related parameters that are **fixed**.
If we want to run parameter search on certain parameters, then consider parameterize them in CLI.
"""
EXPERIMENT_ROOT = "experiments"
MODEL_SIZE = "12b"
DATA_SCHEME = "deduped"
GENERATION_HF_DATASET_NAME = "usvsnsp/semantic-filters"

"""
Feature Catalog
"""
NATURAL_LANGUAGE_SCORE_COLUMN = "nl_scores"
# https://eleutherai.notion.site/Evaluate-NL-Code-Classifier-on-Memorized-Samples-7742b4b768d54131afc07b06f0610148
NATURAL_LANGAUGE_SCORE_THRESHOLD = 0.457414
CONTINUOUS_FEATURE_COLUMNS = [
    "sequence_duplicates",
    "max_frequency",
    "avg_frequency",
    "min_frequency",
    "median_frequency",
    "p25_frequency",
    "p75_frequency",
    "generation_perplexity",
    "prompt_perplexity",
    "sequence_perplexity",
    "0_8_templates",
    "0_8_snowclones",
    "huffman_coding_length",
]
CATEGORICAL_FEATURE_COLUMNS = [
    # This feature needs to be derived from the dataset
    "is_templating",
]
ALL_FEATURE_COLUMNS = CONTINUOUS_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

"""
Derived Features
"""


def derive_is_templating_feature(row: pd.Series) -> int:
    """
    This function derives the `is_templating` feature from the dataset.

    Args:
        row (pd.Series): A row of the dataset

    Returns:
        int: 1 if the row exhibits templating behavior, 0 otherwise
    """
    if row.is_incrementing or row.is_repeating:
        return 1

    return 0


"""
Taxonomy Catalog
"""
TAXONOMIES = ["recitation", "reconstruction", "recollection"]
TAXONOMY_QUANTILES = [0.25, 0.5, 0.75]
TAXONOMY_SEARCH_FEATURES = [
    "max_frequency",
    "avg_frequency",
    "min_frequency",
    "median_frequency",
    "0_8_templates",
    "0_8_snowclones",
    "sequence_duplicates",
    "huffman_coding_length",
    "is_templating",
    "generation_perplexity",
]

"""
Taxonomy Function and Parameters
"""


def taxonomy_function(sequence_duplication_threshold: int = 10) -> Callable[[pd.Series], str]:
    """
    Get the taxonomy function for each sample.

    Args:
        sequence_duplication_threshold (int, optional): The threshold to classify a sample as recitation. Defaults to 10.

    Returns:
        Callable[[pd.Series], str]: The taxonomy function.
    """

    def classify_row(row: pd.Series) -> str:
        if row.sequence_duplicates >= sequence_duplication_threshold:
            return "recitation"
        if row.is_templating:
            return "reconstruction"

        return "recollection"

    return classify_row


"""
Model Training Hyper-parameters
"""
GLOBAL_SEED = 1024

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.2

MAX_MODEL_ITERATIONS = 10000
FIT_INTERCEPT = True
REG_NAME = "l2"
# Inverse of regularization strength; smaller values specify stronger regularization.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
REG_STRENGTH = 1.0
# num_samples / (num_classes * np.bincount(labels))
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
CLASS_WEIGHT_METHOD = "balanced"
