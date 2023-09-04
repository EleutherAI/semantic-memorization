from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .base import PIPELINE_SINGLETON, PrecomputedFeatures
from .constants import PrecomputedFeatureName


@PIPELINE_SINGLETON.register_filter()
def token_frequency_filter(dataset: DataFrame, features: PrecomputedFeatures) -> DataFrame:
    main = dataset.alias("main")
    memorized_frequencies = features[PrecomputedFeatureName.MEMORIZED_TOKEN_FREQUENCIES].alias("memorized")
    non_memorized_frequencies = features[PrecomputedFeatureName.NON_MEMORIZED_TOKEN_FREQUENCIES].alias("non_memorized")

    # First, we expand the token indices, then join to extract the frequencies
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
        )
    )

    # Next, we aggregate the frequencies back as lists of (`token_index`, `frequency`) pairs, sorted by token_index in ascending order
    aggregated_frequencies = token_frequencies.groupby("sequence_id").agg(
        F.sort_array(F.collect_list(F.struct("token_index", "token_id"))).alias("tokens"),
        F.sort_array(F.collect_list(F.struct("token_index", "memorized_frequency"))).alias("memorized_frequencies"),
        F.sort_array(F.collect_list(F.struct("token_index", "non_memorized_frequency"))).alias("non_memorized_frequencies"),
    )
    # Then, we re-format the list by dropping `token_index`
    filtered_frequencies = aggregated_frequencies.select(
        "sequence_id",
        F.transform(F.col("tokens"), lambda x: x.token_id).alias("tokens"),
        F.transform(F.col("memorized_frequencies"), lambda x: x.memorized_frequency).alias("memorized_frequencies"),
        F.transform(F.col("non_memorized_frequencies"), lambda x: x.non_memorized_frequency).alias("non_memorized_frequencies"),
    ).alias("filtered")

    # Finally, re-attach the memorization score from the original dataset
    final = filtered_frequencies.join(main, on="sequence_id", how="left").select(
        "filtered.*",
        F.col("main.memorization_score").alias("memorization_score"),
    )

    return final


@PIPELINE_SINGLETON.register_filter()
def sequence_frequency_filter(dataset: DataFrame, features: PrecomputedFeatures) -> DataFrame:
    main = dataset.alias("main")
    sequence_frequencies = features[PrecomputedFeatureName.SEQUENCE_FREQUENCIES].alias("sequence_frequencies")

    # Join on `sequence_id` to extract the sequence frequency
    final = main.join(sequence_frequencies, on="sequence_id", how="inner").select(
        "main.*",
        F.col("sequence_frequencies.frequency").alias("sequence_frequency"),
    )

    return final
