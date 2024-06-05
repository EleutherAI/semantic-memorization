import re
import unicodedata

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON

def find_if_incrementing_or_repeating(splits, test_repeating=False):
    """Finds if the given list of words is incrementing
    
    A sequence is incrementing if it there are a set of integers or decimals in arithmetic progression 
    intersperced with or without repeating characters
    
    Args:
        splits (list): List of values to be analyzed.

    Returns:
        Tuple(bool, int): A tuple containing a boolean value indicating if 
            the sequence is incrementing and the calculated difference.
    """
    # We need atleast 3 integers to define an AP
    if len(splits) < 3 and not test_repeating:
        return False, 0
    elif len(splits) == 3 and not test_repeating:
        # Every element has to be a number
        if not all([type(i) in [float, int] for i in splits]): return False, 0
        return (splits[2] - 2*splits[1] + splits[0]) < 1e-5, splits[2] - splits[1]
    
    # First and last words of a sequence can be partial
    # We ignore them if length of splits is more than 4
    if len(splits) > 4 and not test_repeating:
        splits = splits[1:-1]
        
        
    is_num_inc = False
    diff = None
    
    for temp_len in range(1, len(splits)//2 + 1):
        is_inc = True
        diff_valid = False
        for every_i in range(temp_len):
            curr_diff = None
            for i in range(every_i + temp_len, len(splits), temp_len):            
                    
                if type(splits[i]) != type(splits[i - temp_len]):
                    is_inc = False
                    break
                
                if curr_diff is None and type(splits[i]) in [int, float]:
                    curr_diff = splits[i] - splits[i - temp_len]
                
                elif type(splits[i]) in [int, float]:
                    is_curr_inc = abs(splits[i] - splits[i - temp_len] - curr_diff) < 1e-10       
                    if not is_curr_inc:
                        is_inc = False
                        break
                    else:
                        diff_valid = True
                elif type(splits[i]) == str and splits[i] != splits[i - temp_len]:
                    is_inc = False
                    break
            
            if not is_inc:
                break
            
            if diff is None and curr_diff is not None and curr_diff != 0:
                diff = curr_diff
        if is_inc:
            if diff is not None and diff_valid:
                return True, diff
            elif diff is None:
                return True, None
    
    return False, 0

def split_text(text, split_type = "incrementing"):

    if split_type == "repeating":
        return list(text)

    elif split_type != "incrementing":
        raise ValueError("Invalid Split Type")

    # Check if we have hexadecimal numerals
    text = re.sub(r"\s+", " ", text)
    splits = []
    text_recon = ""
    for word in text.split(" "):
        try:
            text_recon += str(int(word, 0)) + " "
        except ValueError:
            text_recon += word + " "
    
    text_recon = repr(text_recon.strip())
    for word in text_recon.split(" "):
        
        
        # Replace escape characters
        word = word.replace("\\n", "")
        word = word.replace("\\x", "")
        word = word.replace("\\", "")
        word = word.replace("\'", "")
        word = word.replace("\"", "")
        
        
        word = re.split("([0-9]+)", word)
        splits.extend(word)
        
        
    splits_new = []
    to_continue = False
    for idx, word in enumerate(splits):
        word = word.strip("\'")
        if to_continue:
            to_continue = not to_continue
            continue
        if word.strip(" ") == "":
            continue
        
        if word == "":
            continue
        
        try:
            splits_new.append(int(word))
        except ValueError:
            splits_new.append(word)
    
    return splits_new
        
def is_pattern(text):
    splits = split_text(text)
    is_inc, diff = find_if_incrementing_or_repeating(splits)
    if is_inc and diff is not None and diff != 0:
        return True, False
    
    splits = split_text(text, split_type="repeating") 
    is_inc, diff = find_if_incrementing_or_repeating(splits)
    if is_inc: # we don't have incrementing cases when we split by characters
        return False, True
    else:
        return False, False

@PIPELINE_SINGLETON.register_filter()
def pattern_sequences_filter(dataset: DataFrame, _) -> DataFrame:
    """
    Returns if a sequence is incrementing or repetitive.

    Args:
        dataset (DataFrame): Dataset containing sequences of tokens
        _ (PrecomputedFeatures): Unused

    Returns:
        DataFrame: with additional column of `is_incrementing`
    """
    main = dataset.alias("main")
    pattern_schema = T.StructType(
        [
            T.StructField("is_incrementing", T.BooleanType()),
            T.StructField("is_repeating", T.BooleanType()),
        ]
    )
    patternUDF = F.udf(lambda seq: is_pattern(seq), pattern_schema)

    pattern = main.select(["sequence_id", "text"]).withColumn("pattern", patternUDF("text"))
    
    final = (
        main.join(pattern, on="sequence_id", how="left")
        .drop(pattern.sequence_id)
        .drop(pattern.text)
        .select(
            "main.*",
            F.col("pattern.is_incrementing").alias("is_incrementing"),
            F.col("pattern.is_repeating").alias("is_repeating")
        )
        
    )
    return final


if __name__ == "__main__":
    samp = r"""
    "A.1 , A.2 , A.3 , A.4, B.1 , B.2, B.3, C.1"
    """
    print(is_pattern(samp))
