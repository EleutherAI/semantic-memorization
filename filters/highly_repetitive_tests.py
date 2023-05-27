from highly_repetitive import break_and_compare, break_and_compare_wrapper
# Test cases for break_and_compare

# Test case 1: Matching chunks exist
ls1 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
k1 = 3
expected1 = [1, 2, 3]
output1 = break_and_compare(ls1, k1)
assert output1 == expected1, f"Test case 1 failed. Output: {output1}, Expected: {expected1}"

# Test case 2: No matching chunks
ls2 = [1, 2, 3, 4, 5, 6, 7]
k2 = 3
expected2 = []
output2 = break_and_compare(ls2, k2)
assert output2 == expected2, f"Test case 2 failed. Output: {output2}, Expected: {expected2}"

# Test case 3: Empty list
ls3 = []
k3 = 4
expected3 = []
output3 = break_and_compare(ls3, k3)
assert output3 == expected3, f"Test case 3 failed. Output: {output3}, Expected: {expected3}"

# Test case 4: Chunk size larger than list length
ls4 = [1, 2, 3]
k4 = 4
expected4 = []
output4 = break_and_compare(ls4, k4)
assert output4 == expected4, f"Test case 4 failed. Output: {output4}, Expected: {expected4}"

# Test cases for break_and_compare_wrapper

# Test case 1: Matching chunks within the range
ls1 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
start_k1 = 2
end_k1 = 4
expected_result1 = ([1, 2, 3], 3)
output_result1 = break_and_compare_wrapper(ls1, start_k1, end_k1)
assert output_result1 == expected_result1, f"Test case 1 failed. Output: {output_result1}, Expected: {expected_result1}"

# Test case 2: No matching chunks within the range
ls2 = [1, 2, 3, 4, 5, 6, 7]
start_k2 = 2
end_k2 = 5
expected_result2 = ([], -1)
output_result2 = break_and_compare_wrapper(ls2, start_k2, end_k2)
assert output_result2 == expected_result2, f"Test case 2 failed. Output: {output_result2}, Expected: {expected_result2}"

# Test case 3: Empty list with range
ls3 = []
start_k3 = 1
end_k3 = 3
expected_result3 = ([], -1)
output_result3 = break_and_compare_wrapper(ls3, start_k3, end_k3)
assert output_result3 == expected_result3, f"Test case 3 failed. Output: {output_result3}, Expected: {expected_result3}"

# Test case 4: Single-element list with range
ls4 = [1]
start_k4 = 1
end_k4 = 3
expected_result4 = ([1], 1)
output_result4 = break_and_compare_wrapper(ls4, start_k4, end_k4)
assert output_result4 == expected_result4, f"Test case 4 failed. Output: {output_result4}, Expected: {expected_result4}"
