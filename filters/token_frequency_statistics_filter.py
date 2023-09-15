from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .base import PIPELINE_SINGLETON, PrecomputedFeatures
from .constants import PrecomputedFeatureName


@PIPELINE_SINGLETON.register_filter()
def token_frequency_statistics_filter(dataset: DataFrame, features: PrecomputedFeatures) -> DataFrame:
    """Compute token frequency statistics of a list of token frequencies ordered by the token index (not ID) in the sequence.

    Statistics include:
        - `max_frequency`: maximum frequency of token frequencies in the sequence
        - `min_frequency`: minimum frequency of token frequencies in the sequence
        - `avg_frequency`: average frequency of token frequencies in the sequence
        - `median_frequency`: median frequency of token frequencies in the sequence
        - `p25_frequency`: 25th percentile of token frequencies in the sequence
        - `p75_frequency`: 75th percentile of token frequencies in the sequence

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
        features (PrecomputedFeatures): Precomputed features

    Returns:
        DataFrame: Dataframe with additional columns of token frequency statistics
    """
    main = dataset.alias("main")
    memorized_frequencies = features[PrecomputedFeatureName.MEMORIZED_TOKEN_FREQUENCIES].alias("memorized")
    non_memorized_frequencies = features[PrecomputedFeatureName.NON_MEMORIZED_TOKEN_FREQUENCIES].alias("non_memorized")

    # First, we expand the token indices, then join to extract the frequencies
    # Note that we dropped the memorization score, we'll re-join it later.
    flattened_main = main.select("sequence_id", F.posexplode("tokens").alias("token_index", "token_id"))
    token_frequencies = (
        flattened_main.join(memorized_frequencies, on="token_id", how="left")
        .join(non_memorized_frequencies, on="token_id", how="left")
        .select(
            "sequence_id",
            "token_index",
            "token_id",
            F.col("memorized.frequency").alias("memorized_frequency"),
            F.col("non_memorized.frequency").alias("non_memorized_frequency"),
            (F.col("memorized.frequency") + F.col("non_memorized.frequency")).alias("frequency"),
        )
    )

    # Next, we aggregate the frequencies back as lists of (`token_index`, `frequency`) pairs, sorted by token_index in ascending order.
    # Also, we'll compute frequency statistics per sequence.
    aggregated_frequencies = token_frequencies.groupby("sequence_id").agg(
        F.sort_array(F.collect_list(F.struct("token_index", "token_id"))).alias("tokens"),
        F.sort_array(F.collect_list(F.struct("token_index", "frequency"))).alias("frequencies"),
        F.max("frequency").alias("max_frequency"),
        F.mean("frequency").alias("avg_frequency"),
        F.min("frequency").alias("min_frequency"),
        F.median("frequency").alias("median_frequency"),
        F.percentile_approx("frequency", 0.25).alias("p25_frequency"),
        F.percentile_approx("frequency", 0.75).alias("p75_frequency"),
    )

    # Then, we re-format the list by dropping `token_index`
    filtered_frequencies = aggregated_frequencies.select(
        "sequence_id",
        "max_frequency",
        "avg_frequency",
        "min_frequency",
        "median_frequency",
        "p25_frequency",
        "p75_frequency",
        F.transform(F.col("frequencies"), lambda x: x.frequency).alias("frequencies"),
    ).alias("filtered")

    # Finally, re-attach the memorization score from the original dataset
    final = filtered_frequencies.join(main, on="sequence_id", how="left").drop(filtered_frequencies.sequence_id).select("main.*", "filtered.*")
    return final
