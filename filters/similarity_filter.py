from pyspark.ml.feature import Normalizer
from pyspark.ml.functions import array_to_vector
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON, PrecomputedFeatures
from .constants import PrecomputedFeatureName


@PIPELINE_SINGLETON.register_filter()
def similarity_filter(dataset: DataFrame, features: PrecomputedFeatures) -> DataFrame:
    """
    Compute the number of similar sequences to a given sequence. The similarity is computed using
    cosine similarity and Levenshtein distance.

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
        features (PrecomputedFeatures): Precomputed features

    Returns:
        DataFrame: Dataframe with additional columns of `similar_sequence_count`, number of similar sequences
    """
    cosine_similarity_threshold = 0.9
    levenstein_distance_threshold = 20

    first_normalizer = Normalizer(p=2.0, inputCol="first_embeddings", outputCol="first_normalized_embeddings")
    second_normalizer = Normalizer(p=2.0, inputCol="second_embeddings", outputCol="second_normalized_embeddings")
    dot_product_udf = F.udf(lambda v: v[0].dot(v[1]), T.DoubleType())

    main = dataset.alias("main")
    text = dataset.select("sequence_id", "text").alias("text")
    embeddings = features[PrecomputedFeatureName.EMBEDDINGS].alias("embeddings")
    sentence_embeddings = text.join(embeddings, on="sequence_id", how="inner").select("text.sequence_id", "text.text", "embeddings.embeddings")
    first = sentence_embeddings.alias("first")
    second = sentence_embeddings.alias("second")

    pairwise_embeddings = (
        first.crossJoin(second)
        .select(
            F.col("first.sequence_id").alias("first_sequence_id"),
            F.col("first.text").alias("first_text"),
            array_to_vector(F.col("first.embeddings")).alias("first_embeddings"),
            F.col("second.sequence_id").alias("second_sequence_id"),
            array_to_vector(F.col("second.embeddings")).alias("second_embeddings"),
            F.col("second.text").alias("second_text"),
        )
        .filter(F.col("first_sequence_id") != F.col("second_sequence_id"))
        .alias("pairwise_embeddings")
        .cache()
    )
    normalized_embeddings = second_normalizer.transform(first_normalizer.transform(pairwise_embeddings)).alias("normalized_embeddings")
    distances = normalized_embeddings.select(
        F.col("first_sequence_id").alias("sequence_id"),
        "second_sequence_id",
        dot_product_udf(F.array("first_normalized_embeddings", "second_normalized_embeddings")).alias("cosine_similarity"),
        F.levenshtein("first_text", "second_text").alias("levenshtein_distance"),
    ).cache()

    filtered_distances = distances.filter(
        (F.col("cosine_similarity") >= cosine_similarity_threshold) & (F.col("levenshtein_distance") <= levenstein_distance_threshold)
    ).alias("filtered_distances")
    similar_sequences = (
        filtered_distances.groupby("sequence_id").agg(F.count("second_sequence_id").alias("similar_sequence_count")).alias("similar_sequences")
    )
    final = main.join(similar_sequences, on="sequence_id", how="left").select(
        "main.*",
        "similar_sequences.similar_sequence_count",
    )

    return final
