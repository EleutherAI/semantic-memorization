from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON

from transformers import AutoTokenizer


@PIPELINE_SINGLETON.register_filter()
def detokenize(dataset: DataFrame, _) -> DataFrame:
    """Detokenizes tokens into text as a preprocessing step.

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
        _ (PrecomputedFeatures): Unused

    Returns:
        DataFrame: with additional column of `text`, detokenized text
    """
    main = dataset.alias("main")

    tokenizer_name = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizeUDF = F.udf(lambda seq: tokenizer.decode(seq), T.StringType())

    final = main.withColumn("text", tokenizeUDF("tokens"))

    return final
