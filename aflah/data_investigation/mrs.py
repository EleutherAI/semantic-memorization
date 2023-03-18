import itertools

def find_most_occuring_substring(lst):
    """Find the most occuring substring in a string."""
    # all substrings of length >= 2 and unique
    substrings = set([lst[i:j] for i, j in itertools.combinations(range(len(lst) + 1), 2) if len(lst[i:j]) >= 2])
    most_occuring_substr_count = 0
    most_occuring_substr = None
    for substring in substrings:
        count = lst.count(substring)
        if count > most_occuring_substr_count:
            most_occuring_substr_count = count
            most_occuring_substr = substring
    return most_occuring_substr


if __name__ == '__main__':
    lst = '1212121'
    print(find_most_occuring_substring(lst))