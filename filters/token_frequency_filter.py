from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON, PrecomputedFeatures
from .constants import PrecomputedFeatureName

@PIPELINE_SINGLETON.register_filter()
def token_frequency_filter(dataset: DataFrame, features: PrecomputedFeatures) -> DataFrame:
    """Returns maximum, minimum and average token frequencies

    Returns token frequencies of a sequence in the given dataframe

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
    
    Returns:
        DataFrame: Dataframe with additional columns of `max_frequency`, `min_frequency` and 
            `avg_frequency`; maximum, minimum and average counts of occurance of tokens of that
            sequence in Pile corpus
    """
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
    token_frequencies = (
        token_frequencies.withColumn("frequency", 
        token_frequencies.memorized_frequency + token_frequencies.non_memorized_frequency) 
        .select(
            "sequence_id",
            "token_index",
            "token_id",
            "frequency"
        )
    )


    # Next, we aggregate the frequencies back as lists of (`token_index`, `frequency`) pairs, sorted by token_index in ascending order
    aggregated_frequencies = token_frequencies.groupby("sequence_id").agg(
        F.sort_array(F.collect_list(F.struct("token_index", "token_id"))).alias("tokens"),
        F.sort_array(F.collect_list(F.struct("token_index", "frequency"))).alias("frequencies"),
    )

    # Then, we re-format the list by dropping `token_index`
    filtered_frequencies = aggregated_frequencies.select(
        "sequence_id",
        F.transform(F.col("tokens"), lambda x: x.token_id).alias("tokens"),
        F.transform(F.col("frequencies"), lambda x: x.frequency).alias("frequencies"),
    ).alias("filtered")
    

    # Now we calculate maximum, minimum and average token frequencies per sequence
    columnGetMaxUdf = F.udf(lambda seq:max(seq), T.LongType()) 
    columnGetMinUdf = F.udf(lambda seq:min(seq), T.LongType()) 
    columnGetAvgUdf = F.udf(lambda seq:int(sum(seq)/len(seq)), T.LongType())
    filtered_frequencies = (
        filtered_frequencies.withColumn("max_frequency", columnGetMaxUdf("frequencies"))
        .withColumn("min_frequency", columnGetMinUdf("frequencies"))
        .withColumn("avg_frequency", columnGetAvgUdf("frequencies"))
        .select(
            "sequence_id",
            "max_frequency",
            "min_frequency",
            "avg_frequency"
        )
    )

    # Finally, re-attach the memorization score from the original dataset
    final = filtered_frequencies.join(main, on="sequence_id", how="left")

    # return final
    return final