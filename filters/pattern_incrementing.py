from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON
import unicodedata
import re


def replace_non_numeric_with_whitespace(text: str) -> str:
    # Replace non-numeric characters with whitespace
    # cleaned_text = re.sub(r'[^0-9]', ' ', text)
    new_text = ""
    for i in range(len(text)):
        if text[i].isdigit():
            new_text += str(unicodedata.digit(text[i]))  # Fix for characters like '²' not being converted as required
        elif text[i] == "." and i > 0 and i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            new_text += text[i]
        else:
            new_text += " "
    cleaned_text = new_text

    decimal_seen = False
    notValidFloat = False
    for i in range(len(cleaned_text)):
        if cleaned_text[i] == " ":
            decimal_seen = False
        elif cleaned_text[i] == ".":
            if decimal_seen:
                notValidFloat = True
                break
            else:
                decimal_seen = True
        elif cleaned_text[i].isdigit():
            continue
        else:
            notValidFloat = True
            break

    if notValidFloat:
        # Replace non-numeric characters with whitespace
        cleaned_text = re.sub(r"[^0-9]", " ", text)

    # Replace multiple consecutive whitespaces with a single whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    return cleaned_text


def incrementing_sequences_filter_wrapper(text: str) -> bool:
    # count number of numeric and non-numeric characters
    num_numeric = 0
    num_non_numeric = 0

    for char in text:
        if char.isdigit():
            num_numeric += 1
        else:
            num_non_numeric += 1

    ratio_numeric = num_numeric / (num_numeric + num_non_numeric)

    # print("ratio_numeric", ratio_numeric)

    # if less than 5% of characters are numeric, return False
    if ratio_numeric < 0.05:
        return False

    # remove all non numeric with whitespace
    text = replace_non_numeric_with_whitespace(text)
    if text.count(" ") != 0:
        # convert them to a list
        ls = list(map(float, text.split()))
    else:
        ls = list(text)

    # print("After removing all non-numeric characters with whitespace", ls)

    # Check for incrementing in chunks
    # Adding this to handle cases like "A.1 , A.2 , A.3 , A.4, B.1 , B.2, B.3, C.1"

    # If length of list is 1, the sequence is not an incrementing pattern
    if len(ls) <= 1:
        return False

    ptr = 0
    min_max = {}
    chunk_num = 0
    min_max[chunk_num] = (ls[ptr], ls[ptr + 1], 2)
    ptr += 1
    while ptr < len(ls) - 1:
        if ls[ptr] < ls[ptr + 1]:
            min_max[chunk_num] = (min(min_max[chunk_num][0], ls[ptr]), max(min_max[chunk_num][1], ls[ptr + 1]), min_max[chunk_num][2] + 1)
        else:
            chunk_num += 1
            if ptr + 2 < len(ls):
                min_max[chunk_num] = (ls[ptr + 1], ls[ptr + 2], 1)
            else:
                min_max[chunk_num] = (ls[ptr + 1], ls[ptr + 1], 1)

        ptr += 1

    # remove chunks with size 1
    min_max = {k: v for k, v in min_max.items() if v[2] > 1}

    # if chunk ids are not consecutive, return False
    chunksAreConsecutive = True
    for i in range(len(min_max) - 1):
        if i + 1 not in min_max:
            chunksAreConsecutive = False
            break

    # print("min_max", min_max)

    if chunksAreConsecutive:
        # if all chunks have same min value and last chunk's max value is less than first chunk's max value, return True
        for i in range(len(min_max) - 1):
            if min_max[i][0] != min_max[i + 1][0]:
                break
            if i == len(min_max) - 2 and min_max[i][1] <= min_max[0][1]:
                return True

    # When the list is too small, it is not an incrementing sequence
    # Some results to decide on the threshold (P, R, F1)
    # Without Condition   - 0.48      0.69      0.57
    # when threshold is 3 - 0.58      0.69      0.63
    # when threshold is 4 - 0.60      0.68      0.64
    # when threshold is 5 - 0.62      0.65      0.64
    # when threshold is 6 - 0.64      0.64      0.64
    # when threshold is 7 - 0.67      0.63      0.65
    # These values are subject to change based on the dataset and the code modifications done post procuring them
    if len(ls) < 6:
        return False

    index_to_remove = []
    # remove all repeating at fixed intervals
    for i in range(len(ls) - 1):
        k = 1
        while k < len(ls):
            indices = []
            anySame = False
            for j in range(i, len(ls), k):
                indices.append(j)
                if ls[i] != ls[j]:
                    k += 1
                    anySame = True
                    break
            if not anySame and len(indices) > 1:
                index_to_remove.extend(indices)
                k += 1
            elif not anySame:
                k += 1

    # unravel the list
    index_to_remove = list(set(index_to_remove))

    new_list = []
    for i in range(len(ls)):
        if i not in index_to_remove:
            new_list.append(ls[i])
    ls = new_list

    # print("After removing repeating at fixed intervals", ls)

    # When post cleanup the list is too small, it is not an incrementing sequence
    # This threshold leads to P, R, F1 of 0.71, 0.63, 0.67
    # These values are subject to change based on the dataset and the code modifications done post procuring them
    if len(ls) < 4:
        return False

    # Basic case where numbers are only increasing or decreasing
    isIncreasing = True
    isDecreasing = True
    for i in range(len(ls) - 1):
        if ls[i] > ls[i + 1]:
            isIncreasing = False
            if not isDecreasing:
                break
        if ls[i] < ls[i + 1]:
            isDecreasing = False
            if not isIncreasing:
                break
        else:
            isIncreasing = False
            isDecreasing = False
            break

    if (isIncreasing or isDecreasing) and len(ls) > 1:
        return True

    # break the list into chunks where each chunk is increasing
    increasing_chunks = []
    chunk = []
    for i in range(len(ls) - 1):
        if ls[i] <= ls[i + 1]:
            chunk.append(ls[i])
        else:
            chunk.append(ls[i])
            increasing_chunks.append(chunk)
            chunk = []
    if len(ls) > 1:
        chunk.append(ls[-1])
        increasing_chunks.append(chunk)

    # break the list into chunks where each chunk is decreasing
    decreasing_chunks = []
    chunk = []
    for i in range(len(ls) - 1):
        if ls[i] >= ls[i + 1]:
            chunk.append(ls[i])
        else:
            chunk.append(ls[i])
            decreasing_chunks.append(chunk)
            chunk = []
    if len(ls) > 1:
        chunk.append(ls[-1])
        decreasing_chunks.append(chunk)

    # print lengths of chunks
    # print("increasing_chunks", increasing_chunks)
    # print("decreasing_chunks", decreasing_chunks)

    # if first chunk is of unequal size remove it
    if len(increasing_chunks) >= 2 and len(increasing_chunks[0]) != len(increasing_chunks[1]):
        increasing_chunks.pop(0)
    if len(decreasing_chunks) >= 2 and len(decreasing_chunks[0]) != len(decreasing_chunks[1]):
        decreasing_chunks.pop(0)

    # if last chunk is of unequal size remove it
    if len(increasing_chunks) >= 2 and len(increasing_chunks[-1]) != len(increasing_chunks[-2]):
        increasing_chunks.pop(-1)
    if len(decreasing_chunks) >= 2 and len(decreasing_chunks[-1]) != len(decreasing_chunks[-2]):
        decreasing_chunks.pop(-1)

    # if any chunk is of unequal size return False
    for chunk in increasing_chunks:
        if len(chunk) != len(increasing_chunks[0]):
            return False
    for chunk in decreasing_chunks:
        if len(chunk) != len(decreasing_chunks[0]):
            return False

    # print lengths of chunks
    # print("increasing_chunks", increasing_chunks)
    # print("decreasing_chunks", decreasing_chunks)

    if len(increasing_chunks) > 1:
        isIncreasing_increasing_chunks = [True] * len(increasing_chunks[0])
        isDecreasing_increasing_chunks = [True] * len(increasing_chunks[0])
        for i in range(len(increasing_chunks) - 1):
            for j in range(len(increasing_chunks[i])):
                if increasing_chunks[i][j] < increasing_chunks[i + 1][j]:
                    isDecreasing_increasing_chunks[j] = False
                    if not isIncreasing_increasing_chunks[j]:
                        break
                if increasing_chunks[i][j] > increasing_chunks[i + 1][j]:
                    isIncreasing_increasing_chunks[j] = False
                    if not isDecreasing_increasing_chunks[j]:
                        break
    else:
        isIncreasing_increasing_chunks = []
        isDecreasing_increasing_chunks = []

    if len(decreasing_chunks) > 1:
        isIncreasing_decreasing_chunks = [True] * len(decreasing_chunks[0])
        isDecreasing_decreasing_chunks = [True] * len(decreasing_chunks[0])
        for i in range(len(decreasing_chunks) - 1):
            for j in range(len(decreasing_chunks[i])):
                if decreasing_chunks[i][j] < decreasing_chunks[i + 1][j]:
                    isDecreasing_decreasing_chunks[j] = False
                    if not isIncreasing_decreasing_chunks[j]:
                        break
                if decreasing_chunks[i][j] > decreasing_chunks[i + 1][j]:
                    isIncreasing_decreasing_chunks[j] = False
                    if not isDecreasing_decreasing_chunks[j]:
                        break
    else:
        isIncreasing_decreasing_chunks = []
        isDecreasing_decreasing_chunks = []

    largest_chunk_size = max(
        len(isIncreasing_increasing_chunks),
        len(isDecreasing_increasing_chunks),
        len(isIncreasing_decreasing_chunks),
        len(isDecreasing_decreasing_chunks),
    )
    if len(isIncreasing_increasing_chunks) < largest_chunk_size:
        isIncreasing_increasing_chunks.extend([False] * (largest_chunk_size - len(isIncreasing_increasing_chunks)))
    if len(isDecreasing_increasing_chunks) < largest_chunk_size:
        isDecreasing_increasing_chunks.extend([False] * (largest_chunk_size - len(isDecreasing_increasing_chunks)))
    if len(isIncreasing_decreasing_chunks) < largest_chunk_size:
        isIncreasing_decreasing_chunks.extend([False] * (largest_chunk_size - len(isIncreasing_decreasing_chunks)))
    if len(isDecreasing_decreasing_chunks) < largest_chunk_size:
        isDecreasing_decreasing_chunks.extend([False] * (largest_chunk_size - len(isDecreasing_decreasing_chunks)))

    # print("isIncreasing_increasing_chunks", isIncreasing_increasing_chunks)
    # print("isDecreasing_increasing_chunks", isDecreasing_increasing_chunks)
    # print("isIncreasing_decreasing_chunks", isIncreasing_decreasing_chunks)
    # print("isDecreasing_decreasing_chunks", isDecreasing_decreasing_chunks)

    if len(isIncreasing_increasing_chunks) >= 1:
        resp = (
            isIncreasing_decreasing_chunks[0]
            or isDecreasing_decreasing_chunks[0]
            or isIncreasing_increasing_chunks[0]
            or isDecreasing_increasing_chunks[0]
        )
        for i, j, k, l in zip(
            isIncreasing_increasing_chunks, isDecreasing_increasing_chunks, isIncreasing_decreasing_chunks, isDecreasing_decreasing_chunks
        ):
            resp = resp and (i or j or k or l)
        if resp:
            return True

    return False


@PIPELINE_SINGLETON.register_filter()
def incrementing_sequences_filter(dataset: DataFrame, _) -> DataFrame:
    """Returns if a sequence is incrementing

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
        _ (PrecomputedFeatures): Unused

    Returns:
        DataFrame: with additional column of `is_incrementing`
    """
    main = dataset.alias("main")
    incrementingUDF = F.udf(lambda seq: incrementing_sequences_filter_wrapper(seq), T.BooleanType())

    final = main.withColumn("is_incrementing", incrementingUDF("text"))

    return final


if __name__ == "__main__":
    samp = r"""
    "A.1 , A.2 , A.3 , A.4, B.1 , B.2, B.3, C.1"
    """
    print(incrementing_sequences_filter(samp))
