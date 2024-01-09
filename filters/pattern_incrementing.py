import re
import unicodedata

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .base import PIPELINE_SINGLETON

def replace_non_numeric_with_whitespace(text: str) -> str:
    """
    Replaces non-numeric characters with whitespace.

    Args:
        text (str): Input text

    Returns:
        str: Text with non-numeric characters replaced with whitespace
    """
    # Replace non-numeric characters with whitespace
    # cleaned_text = re.sub(r'[^0-9]', ' ', text)
    new_text = ""
    for split in text.split():
        try:
            new_text += str(int(split, 0))
        except:
            for i in range(len(split)):
                if split[i].isdigit():
                    new_text += str(unicodedata.digit(split[i]))  # Fix for characters like '²' not being converted as required
                elif split[i] == '-' and i < (len(split) - 1) and split[i+1].isdigit():
                    new_text += '-'

    # Replace multiple consecutive whitespaces with no whitespace
    cleaned_text = re.sub(r"\s+", "", new_text)

    return cleaned_text

def incrementing_sequences_filter_wrapper(text: str) -> bool:
    """
    Returns if the given sequence is incrementing or decrementing.
    
    We define a sequence as incrementing if there exists incrementing or decrementing terms
    in a given sequence
    
    we define a sequence of integers to be incrementing if the sequenc 

    Args:
        text (str): Input text

    Returns:
        bool: True if the sequence is incrementing, False otherwise
    """
    
    ls = replace_non_numeric_with_whitespace(text)

    if len(ls) < 3:
        return False
    
    for one_s in range(len(ls)):
        for one_e in range(one_s + 1, min(len(ls), one_s + 20)):
            try:
                # One of the two numbers is only '-', so we skip the current iteration
                num_one = int(ls[one_s:one_e])
            except:
                continue
                
            for two_s in range(one_e, len(ls)):
                for two_e in range(two_s + 1, min(len(ls), two_s + 20)):
                    try:
                        num_two = int(ls[two_s:two_e])
                    except:
                        # One of the two numbers is only '-', so we skip the current iteration
                        continue
                    
                    diff = num_two - num_one
                    
                    if diff == 0:
                        # we donot consider repetitive sequences as incrementing
                        continue
                    
                    count = 2
                    curr_pos = two_e
                    curr_num = num_two
                    curr_len = two_e - two_s + one_e - one_s
                    while True:
                        curr_text = ls[curr_pos:]
                        if curr_text == "":
                            break
                        next_num = curr_num + diff
                        if str(next_num) in curr_text:
                            pos = curr_text.find(str(next_num))
                        else:
                            break
                        
                        count += 1
                        
                        curr_len += len(str(next_num))
                        curr_pos += pos + len(str(curr_num))
                        curr_num = next_num
                        
                        
                        # we want to make sure there are atleast 3 instances to find if a sequence of digits is incrementing
                        if (curr_len / len(ls)) >= 0.70:
                            return True  
    
    return False


@PIPELINE_SINGLETON.register_filter()
def incrementing_sequences_filter(dataset: DataFrame, _) -> DataFrame:
    """
    Returns if a sequence is incrementing.

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
    print(incrementing_sequences_filter_wrapper(samp))
