from .highly_repetitive import break_and_compare, break_and_compare_wrapper
# Test cases for break_and_compare

# Test case 1: Matching chunks exist
def test_break_and_compare_matching_chunks_exist():
    ls = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    k = 3
    expected = [1, 2, 3]
    output = break_and_compare(ls, k)
    assert output == expected, f"Test case 1 failed. Output: {output}, Expected: {expected}"

# Test case 2: No matching chunks
def test_break_and_compare_no_matching_chunks():
    ls = [1, 2, 3, 4, 5, 6, 7]
    k = 3
    expected = []
    output = break_and_compare(ls, k)
    assert output == expected, f"Test case 2 failed. Output: {output}, Expected: {expected}"

# Test case 3: Empty list
def test_break_and_compare_empty_list():
    ls = []
    k = 4
    expected = []
    output = break_and_compare(ls, k)
    assert output == expected, f"Test case 3 failed. Output: {output}, Expected: {expected}"

# Test case 4: Chunk size larger than list length
def test_break_and_compare_chunk_size_larger_than_list_length():
    ls = [1, 2, 3]
    k = 4
    expected = []
    output = break_and_compare(ls, k)
    assert output == expected, f"Test case 4 failed. Output: {output}, Expected: {expected}"

# Test cases for break_and_compare_wrapper

# Test case 1: Matching chunks within the range
def test_break_and_compare_wrapper_matching_chunks_within_range():
    ls = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    start_k = 2
    end_k = 4
    expected = ([1, 2, 3], 2)
    output = break_and_compare_wrapper(ls, start_k, end_k)
    assert output == expected, f"Test case 1 failed. Output: {output}, Expected: {expected}"

# Test case 2: No matching chunks within the range
def test_break_and_compare_wrapper_no_matching_chunks_within_range():
    ls = [1, 2, 3, 4, 5, 6, 7]
    start_k = 2
    end_k = 5
    expected = ([], -1)
    output = break_and_compare_wrapper(ls, start_k, end_k)
    assert output == expected, f"Test case 2 failed. Output: {output}, Expected: {expected}"

# Test case 3: Empty list with range
def test_break_and_compare_wrapper_empty_list_with_range():
    ls = []
    start_k = 1
    end_k = 3
    expected = ([], -1)
    output = break_and_compare_wrapper(ls, start_k, end_k)
    assert output == expected, f"Test case 3 failed. Output: {output}, Expected: {expected}"

# Test case 4: Single-element list with range
def test_break_and_compare_wrapper_single_element_list_with_range():
    ls = [1]
    start_k = 1
    end_k = 3
    expected = ([1], 1)
    output = break_and_compare_wrapper(ls, start_k, end_k)
    assert output == expected, f"Test case 4 failed. Output: {output}, Expected: {expected}"

