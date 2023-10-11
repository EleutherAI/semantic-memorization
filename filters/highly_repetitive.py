from typing import List, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON


def break_and_compare(ls: List, k: int) -> List:
    """
    This function takes a list ls and an integer k as input and returns a list which is the first chunk of ls that is repeated k times. If no such chunk exists, it returns an empty list.

    Args:
        ls (List): The input list.
        k (int): The integer value used for splitting and comparing the list.

    Returns:
        List: The first chunk of ls that is repeated k times. If no such chunk exists, it returns an empty list.
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


def break_and_compare_wrapper(ls: List, start_k: int, end_k: int) -> Union[Tuple[List, int, int], Tuple[List, int]]:
    """

    This function serves as a wrapper for the `break_and_compare` function. It takes an additional two integer parameters `start_k` and `end_k` to define a range of values for `k`.
    It iterates over this range and calls `break_and_compare` for each value of `k` within the range.

    Args:
        ls (List): The input list.
        start_k (int): The starting value of `k` for the range (inclusive).
        end_k (int): The ending value of `k` for the range (inclusive).

    Returns:
        Union[Tuple[List, int, int], Tuple[List, int]]: A tuple containing the result of `break_and_compare` and the values of `i` and `k` for which the result was obtained.
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
                return result, i, k
        result = break_and_compare(ls, k)
        if result:
            return result, 0, k
    return [], 0, -1


def find_smallest_repeating_unit(lst) -> List:
    """
    This function takes a list as input and returns the smallest repeating unit of the list. If no such unit exists, it returns the list itself.

    Args:
        lst (List): The input list.

    Returns:
        List: The smallest repeating unit of the list. If no such unit exists, it returns the list itself.
    """
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
                return unit, n // unit_length

    # If no repeating unit is found, the list itself is the smallest repeating unit
    return lst, 1


@PIPELINE_SINGLETON.register_filter()
def highly_repetitive_filter(dataset: DataFrame, _) -> DataFrame:
    """
    Returns the repeating chunk and the number of times a sequence is repeating.

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
        _ (PrecomputedFeatures): Unused

    Outputs Include:
        - `num_repeating`: Number of times a sequence is repeating
        - `smallest_repeating_chunk`: Smallest repeating token sequence

    Returns:
        DataFrame: with additional columns
            `repeating_chunk`: Repeating Chunk
            `num_repeating`: Number of times the chunk is repeating
            `repeating_offset`: Offset of repeating sequence
    """
    main = dataset.alias("main")
    repetitive_schema = T.StructType(
        [
            T.StructField("repeating_chunk", T.ArrayType(T.LongType())),
            T.StructField("repeating_offset", T.IntegerType()),
            T.StructField("num_repeating", T.IntegerType())
        ]
    )

    start_k = 2
    end_k = 5
    repetitiveUDF = F.udf(lambda seq: break_and_compare_wrapper(seq, start_k, end_k), repetitive_schema)

    smallest_repeating_chunk_schema = T.StructType(
        [
            T.StructField("smallest_repeating_chunk", T.ArrayType(T.LongType())),
            T.StructField("num_times", T.IntegerType())
        ]
    )
    smallest_repeating_chunkUDF = F.udf(lambda seq: find_smallest_repeating_unit(seq), smallest_repeating_chunk_schema)

    repetitive_counts = main.select("sequence_id", "tokens").withColumn("repetitive", repetitiveUDF("tokens"))
    repetitive_counts = repetitive_counts.withColumn("smallest_repeating", smallest_repeating_chunkUDF("repetitive.repeating_chunk"))

    final = (
        main.join(repetitive_counts, on="sequence_id", how="left")
        .drop(repetitive_counts.sequence_id)
        .drop(repetitive_counts.tokens)
        .drop(repetitive_counts.repetitive.repeating_chunk)
        .select(
            "main.*",
            F.col("repetitive.repeating_offset").alias("repeating_offset"),
            (F.col("repetitive.num_repeating")*F.col("smallest_repeating.num_times")).alias("num_repeating"),
            F.col("smallest_repeating.smallest_repeating_chunk").alias("smallest_repeating_chunk")
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
