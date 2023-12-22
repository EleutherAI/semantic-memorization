from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .base import PIPELINE_SINGLETON, PrecomputedFeatures
from .constants import PrecomputedFeatureName


@PIPELINE_SINGLETON.register_filter()
def semantic_duplicates_filter(dataset: DataFrame, features: PrecomputedFeatures) -> DataFrame:
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
    snowclones = features[PrecomputedFeatureName.SEMANTIC_SNOWCLONES].alias("semantic_snowclones")
    templates = features[PrecomputedFeatureName.SEMANTIC_TEMPLATES].alias("semantic_templates")


    # Join on `sequence_id` to extract snowclones
    snowclones_df = main.join(snowclones, on="sequence_id", how="left").select(
        "main.*",
        F.col("semantic_snowclones.0_8_frequencies").alias("0_8_snowclones"),
        F.col("semantic_snowclones.0_9_frequencies").alias("0_9_snowclones"),
    ).alias("main")

    final = snowclones_df.join(templates, on="sequence_id", how="left").select(
        "main.*",
        F.col("semantic_templates.0_8_frequencies").alias("0_8_templates"),
        F.col("semantic_templates.0_9_frequencies").alias("0_9_templates"),
    )
    return final