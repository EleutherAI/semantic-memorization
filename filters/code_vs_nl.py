from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .base import PIPELINE_SINGLETON, PrecomputedFeatures
from .constants import PrecomputedFeatureName


@PIPELINE_SINGLETON.register_filter()
def code_vs_nl_filter(dataset: DataFrame, features: PrecomputedFeatures) -> DataFrame:
    """
    Compute the number of duplicates (frequency) of a sequence.

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
        features (PrecomputedFeatures): Precomputed features

    Returns:
        DataFrame: Dataframe with additional columns of `sequence_duplicates`, number of times that
            64-gram sequence occurs in Pile corpus
    """
    main = dataset.alias("main")
    code_vs_nl = features[PrecomputedFeatureName.IS_CODE].alias("code_vs_nl")

    # Join on `sequence_id` to extract the sequence frequency
    final = main.join(code_vs_nl, on="sequence_id", how="left").select(
        "main.*",
        F.col("code_vs_nl.nl_scores").alias("nl_scores"),
    )

    return final