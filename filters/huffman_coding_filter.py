import numpy as np
from collections import Counter

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON

def calculate_huffman_code_length(text):
    """
    Calculate the length of the huffman code for the given text

    Args:
        text: Text to calculate huffman coding length of

    Returns:
        int: Huffman Coding length
    """
    # Get the number of times each character appears in the text.
    char_counts = np.array(list(Counter(text).values()))
    # Get the probability of each character appearing in the text.
    char_probs = char_counts / len(text)
    # Calculate the length of the huffman code for each character.
    char_code_lengths = np.ceil(-np.log2(char_probs))
    # Calculate the average length of the huffman code for the text.
    huffman_code_length = np.sum(char_code_lengths * char_probs)
    return huffman_code_length.item()

@PIPELINE_SINGLETON.register_filter()
def huffman_coding_filter(dataset: DataFrame, _) -> DataFrame:
    """
    Returns huffman coding length of a sequence

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens and detokenized texts
        _ (PrecomputedFeatures): Unused

    Returns:
        DataFrame: with additional column of `huffman_coding_length`
    """
    main = dataset.alias("main")
    huffmanUDF = F.udf(lambda seq: calculate_huffman_code_length(seq), T.DoubleType())

    final = main.withColumn("huffman_coding_length", huffmanUDF("text"))

    return final