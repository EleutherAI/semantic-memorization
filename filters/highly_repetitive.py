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
        chunks = [to_break[i:i + chunk_size] for i in range(0, len(to_break), chunk_size)]
        chunksMatch = True
        # compare all chunks to first chunk
        for chunk in chunks[1:]:
            if chunk != chunks[0]:
                chunksMatch = False
                break
        if chunksMatch:
            # compare residual to first chunk
            if residual == chunks[0][:len(residual)]:
                return chunks[0]
        chunk_size -= 1
        new_residual = to_break[chunk_size * k:]
        to_break = to_break[:chunk_size * k]
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
    for k in range(start_k, end_k + 1):
        result = break_and_compare(ls, k)
        if result:
            return result, k
    return [], -1