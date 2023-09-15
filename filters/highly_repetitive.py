from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON


def break_and_compare(ls: list, k: int) -> list:
    """
    This function takes a list ls and an integer k as input and returns a list which is the first chunk of ls that is repeated k times. If no such chunk exists, it returns an empty list.

    Parameters:

        ls (list): The input list.
        k (int): The integer value used for splitting and comparing the list.

    """
    n = len(ls)
    while n % k != 0:
        n -= 1
    to_break = ls[:n]
    residual = ls[n:]
    chunk_size = n // k
    while len(residual) < chunk_size:
        # split into chunks
        chunks = [to_break[i : i + chunk_size] for i in range(0, len(to_break), chunk_size)]
        chunksMatch = True
        # compare all chunks to first chunk
        for chunk in chunks[1:]:
            if chunk != chunks[0]:
                chunksMatch = False
                break
        if chunksMatch:
            # compare residual to first chunk
            if residual == chunks[0][: len(residual)]:
                return chunks[0]
        chunk_size -= 1
        new_residual = to_break[chunk_size * k :]
        to_break = to_break[: chunk_size * k]
        residual = new_residual + residual
    return []


def break_and_compare_wrapper(ls: list, start_k: int, end_k: int) -> list:
    """

    This function serves as a wrapper for the `break_and_compare` function. It takes an additional two integer parameters `start_k` and `end_k` to define a range of values for `k`.
    It iterates over this range and calls `break_and_compare` for each value of `k` within the range.

    Parameters:
    - `ls` (list): The input list.
    - `start_k` (int): The starting value of `k` for the range (inclusive).
    - `end_k` (int): The ending value of `k` for the range (inclusive).

    """
    # end_k is inclusive
    ls = list(ls)
    length = len(ls)
    half = length // 2
    for k in range(start_k, end_k + 1):
        for i in range(0, half):
            # remove some tokens from the end as well
            rem = 2
            # when rem = 0 -> 0.91      0.73      0.81
            # when rem = 1 -> 0.91      0.78      0.84
            # when rem = 2 -> 0.90      0.80      0.84
            # when rem = 3 -> 0.89      0.80      0.84
            # when rem = 4 -> 0.89      0.80      0.84
            # when rem = 5 -> 0.89      0.80      0.84
            # when rem = 6 -> 0.89      0.80      0.84
            for j in range(0, rem + 1):
                result = break_and_compare(ls[i : length - j], k)
                if result:
                    return result, i, k
            result = break_and_compare(ls[i:], k)
            if result:
                return result, k
        result = break_and_compare(ls, k)
        if result:
            return result, 0, k
    return [], 0, -1


def find_smallest_repeating_unit(lst):
    if lst is None:
        return []
    n = len(lst)

    # Try all possible lengths of repeating units
    for unit_length in range(1, n // 2 + 1):
        # Check if the list can be divided into repeating units of the current length
        if n % unit_length == 0:
            unit = lst[:unit_length]  # Extract a potential repeating unit

            # Check if the entire list can be formed by repeating the unit
            if all(lst[i : i + unit_length] == unit for i in range(0, n, unit_length)):
                return unit

    # If no repeating unit is found, the list itself is the smallest repeating unit
    return lst


@PIPELINE_SINGLETON.register_filter()
def highly_repetitive_filter(dataset: DataFrame, _) -> DataFrame:
    """Returns the repeating chunk and the number of times a sequence is repeating

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
        _ (PrecomputedFeatures): Unused

    Outputs Include:
        - `num_repeating`: Number of times a sequence is repeating
        - `smallest_repeating_chunk`: Smallest repeating token sequence
    Returns:
        DataFrame: with additional column of `is_incrementing`
    """
    main = dataset.alias("main")
    repetitive_schema = T.StructType(
        [
            T.StructField("num_repeating", T.IntegerType()),
            T.StructField("repeating_offset", T.IntegerType()),
            T.StructField("repeating_chunk", T.ArrayType(T.LongType())),
        ]
    )
    repetitiveUDF = F.udf(lambda seq: break_and_compare_wrapper(seq, 2, 5), repetitive_schema)
    smallest_repeating_chunkUDF = F.udf(lambda seq: find_smallest_repeating_unit(seq), T.ArrayType(T.LongType()))

    repetitive_counts = main.select("sequence_id", "text").withColumn("repetitive", repetitiveUDF("text"))
    repetitive_counts = repetitive_counts.withColumn("smallest_repeating_chunk", smallest_repeating_chunkUDF("repetitive.repeating_chunk"))

    final = (
        repetitive_counts.join(main, on="sequence_id", how="left")
        .drop(repetitive_counts.sequence_id)
        .drop(repetitive_counts.text)
        .drop(repetitive_counts.repetitive.repeating_chunk)
        .select(
            "main.*",
            "repetitive.*",
            "smallest_repeating_chunk",
        )
    )

    return final


if __name__ == "__main__":
    #     from transformers import AutoTokenizer
    #     inp = """0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    #  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff"""
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         "EleutherAI/pythia-70m-deduped",
    #     )
    #     inp = tokenizer(inp)['input_ids']
    #     print(inp)
    #     # for token in inp:
    #     #     print(token, tokenizer.decode(token))
    #     print(break_and_compare_wrapper(inp, 2, 30))
    ls = [1]
    start_k = 1
    end_k = 3
    expected = ([1], 1)
    output = break_and_compare_wrapper(ls, start_k, end_k)
    print(output)
