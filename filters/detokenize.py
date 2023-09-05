from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON, PrecomputedFeatures
from .constants import PrecomputedFeatureName

from transformers import AutoTokenizer

@PIPELINE_SINGLETON.register_filter()
def detokenize(dataset: DataFrame, features: PrecomputedFeatures) -> DataFrame:
    """Detokenizes tokens into text as a preprocessing step. 

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
    
    Returns:
        DataFrame: with additional `text` column
    """
    main = dataset.alias("main")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizeUDF = F.udf(lambda seq:tokenizer.decode(seq), T.StringType())

    main = main.withColumn("text", tokenizeUDF("tokens"))

    return main
    
