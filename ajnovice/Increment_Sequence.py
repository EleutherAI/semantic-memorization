import re

def is_incrementing_arithmetic(seq):
    diff = seq[1] - seq[0]
    return all(seq[i] - seq[i-1] == diff for i in range(2, len(seq)))

def is_incrementing_geometric(seq):
    ratio = seq[1] / seq[0]
    return all(seq[i] / seq[i-1] == ratio for i in range(2, len(seq)))

def is_incrementing_fibonacci(seq):
    a, b = seq[0], seq[1]
    for i in range(2, len(seq)):
        c = a + b
        if seq[i] != c:
            return False
        a, b = b, c
    return True

def is_incrementing_quadratic(seq):
    n = len(seq)
    A = [[i**2, i, 1] for i in range(1, n+1)]
    b = seq
    try:
        a, b, c = tuple(map(float, np.linalg.solve(A, b)))
        return all(seq[i] == a*(i+1)**2 + b*(i+1) + c for i in range(n))
    except np.linalg.LinAlgError:
        return False

def is_incrementing_cubic(seq):
    n = len(seq)
    A = [[i**3, i**2, i, 1] for i in range(1, n+1)]
    b = seq
    try:
        a, b, c, d = tuple(map(float, np.linalg.solve(A, b)))
        return all(seq[i] == a*(i+1)**3 + b*(i+1)**2 + c*(i+1) + d for i in range(n))
    except np.linalg.LinAlgError:
        return False

def is_incrementing_exponential(seq):
    ratio = seq[1] / seq[0]
    return all(seq[i] / seq[i-1] == ratio for i in range(2, len(seq)))

def is_incrementing_triangular(seq):
    n = len(seq)
    return all(seq[i] == (i+1)*(i+2)//2 for i in range(n))

def is_incrementing_square(seq):
    return all(int(seq[i]**0.5)**2 == seq[i] for i in range(len(seq)))

def is_incrementing_pentagonal(seq):
    n = len(seq)
    return all(seq[i] == i*(3*i-1)//2 + 1 for i in range(n))

def is_incrementing_prime(seq):
    return all(seq[i] > 1 and all(seq[i] % j != 0 for j in range(2, int(seq[i]**0.5)+1)) and seq[i] == seq[i-1]+1 for i in range(len(seq)))

def is_incrementing_DNA(seq):
    return all(nucleotide in 'ACGT' and seq[i][0] == seq[i-1][0] and int(seq[i][1:]) == int(seq[i-1][1:])+1 for i in range(1, len(seq)))

def is_incrementing_morse(seq):
    morse_code = {
        '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
        '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
        '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
        '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
        '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
        '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
        '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
        '-----': '0', '--..--': ',', '.-.-.-': '.', '..--..': '?', '-..-.': '/',
        '-....-': '-', '-.--.': '(', '-.--.-': ')', '': ' '}
    return all(code in morse_code and code == seq[i-1]+'.' for i, code in enumerate(seq[1:], start=1))

def is_incrementing_binary(seq):
    return all(int(seq[i], 2) == int(seq[i-1], 2) + 1 for i in
