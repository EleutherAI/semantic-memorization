from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .base import PIPELINE_SINGLETON, PrecomputedFeatures
from .constants import PrecomputedFeatureName


@PIPELINE_SINGLETON.register_filter()
def sequence_duplicates_filter(dataset: DataFrame, features: PrecomputedFeatures) -> DataFrame:
    """Compute the number of duplicates (frequency) of a sequence.

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
        features (PrecomputedFeatures):

    Returns:
        DataFrame: Dataframe with additional columns of `sequence_duplicates`, number of times that
            64-gram sequence occurs in Pile corpus
    """
    main = dataset.alias("main")
    sequence_frequencies = features[PrecomputedFeatureName.SEQUENCE_FREQUENCIES].alias("sequence_frequencies")

    # Join on `sequence_id` to extract the sequence frequency
    final = main.join(sequence_frequencies, on="sequence_id", how="inner").select(
        "main.*",
        F.col("sequence_frequencies.frequency").alias("sequence_duplicates"),
    )

    return final
