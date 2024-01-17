"""
This file manages the modeling-related parameters with hard-coded values.
If we want to change the parameters, then we can parameterize them in CLI.
"""
EXPERIMENT_BASE = "experiments"
MODEL_SIZE = "12b"
DATA_SCHEME = "deduped"
GENERATION_HF_DATASET_NAME = "usvsnsp/generation-semantic-filters"
ENTROPY_HF_DATASET_NAME = "usvsnsp/semantic-memorization-entropies"

"""
Feature Catalog
"""
CONTINUOUS_FEATURE_COLUMNS = [
    "max_frequency",
    "avg_frequency",
    "min_frequency",
    "median_frequency",
    "p25_frequency",
    "p75_frequency",
    # These perplexities came from `generation-semantic-memorization-filters`, suffixed by `_generation`
    "generation_perplexity_generation",
    "prompt_perplexity_generation",
    "sequence_perplexity_generation",
    "0_8_templates",
    "0_8_snowclones",
    "huffman_coding_length",
    # entropy/gini came from `semantic-memorization-entropies`
    "avg entropy",
    "avg gini",
]
CATEGORICAL_FEATURE_COLUMNS = [
    # This feature needs to be derived from the dataset
    "is_templating",
]
ALL_FEATURE_COLUMNS = CONTINUOUS_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


def derive_is_templating_feature(row):
    if row.is_incrementing or row.is_repeating:
        return True

    return False


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
    "avg entropy",
    "avg gini",
    "is_templating",
    "generation_perplexity_generation",
]
SEQUENCE_DUPLICATE_THRESHOLD = 10

"""
Model Training Hyper-parameters
"""
GLOBAL_SEED = 2024_01_01
TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.2
MAX_MODEL_ITERATIONS = 10000
FIT_INTERCEPT = True
REG_NAME = "l2"
# Inverse of regularization strength; smaller values specify stronger regularization.
# Reference https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
REG_STRENGTH = 1.0
