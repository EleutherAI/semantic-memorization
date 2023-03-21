def longestRepeatedSublist(ls):
    ls = list(ls)
    n = len(ls)
    LCSRe = [[0 for x in range(n + 1)]
                for y in range(n + 1)]
 
    res = [] # To store result
    res_length = 0 # To store length of result
 
    # building table in bottom-up manner
    index = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
             
            # (j-i) > LCSRe[i-1][j-1] to remove
            # overlapping
            if (ls[i - 1] == ls[j - 1] and
                LCSRe[i - 1][j - 1] < (j - i)):
                LCSRe[i][j] = LCSRe[i - 1][j - 1] + 1
 
                # updating maximum length of the
                # substring and updating the finishing
                # index of the suffix
                if (LCSRe[i][j] > res_length):
                    res_length = LCSRe[i][j]
                    index = max(i, index)
                 
            else:
                LCSRe[i][j] = 0
 
    # If we have non-empty result, then insert
    # all characters from first character to
    # last character of string
    if (res_length > 0):
        for i in range(index - res_length + 1,
                                    index + 1):
            res.append(ls[i - 1])
    
    return res

def count_subls(ls, sub_ls):
    ls = list(ls)
    sub_ls = list(sub_ls)
    ls = ''.join(map(str, ls))
    sub_ls = ''.join(map(str, sub_ls))
    c = ls.count(sub_ls)
    return c
 
# Driver Code
if __name__ == "__main__":
    ls = "121313121"
    ls = list(map(int, ls))
    output = longestRepeatedSublist(ls)
    print(output, count_subls(ls, output))
    ls = "121212121"
    ls = list(map(int, ls))
    output = longestRepeatedSublist(ls)
    print(output, count_subls(ls, output))
    ls = "1121212"
    ls = list(map(int, ls))
    output = longestRepeatedSublist(ls)
    print(output, count_subls(ls, output))
    ls = "123123123123123121"
    ls = list(map(int, ls))
    output = longestRepeatedSublist(ls)
    print(output, count_subls(ls, output))