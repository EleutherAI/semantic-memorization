def longest_repeating_sublist(lst):
    """
    Finds the longest repeating sublist in a list.
    """
    lst = list(lst)
    longest_sublist = []
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            # check if the sublist repeats later in the list
            k = 0
            while j+k < len(lst) and lst[i+k] == lst[j+k]:
                k += 1
            if k > 0 and (not longest_sublist or k > len(longest_sublist)):
                # found a longer repeating sublist
                longest_sublist = lst[i:i+k]
    return longest_sublist


if __name__ == '__main__':
    lst = [ 1,2,3,1]
    longest_sublist = longest_repeating_sublist(lst)
    print(longest_sublist)  # Output: [5, 6, 7, 8, 9]