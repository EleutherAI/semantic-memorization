def __increasing_sequence_alphanumeric(lst):
    try:
        n = len(lst)
        max_length = 1
        current_length = 1
        max_start_index = 0
        for i in range(1, n):
            if __extract_numeric_and_string_parts(lst[i]) > __extract_numeric_and_string_parts(lst[i - 1]):
                current_length += 1
                if current_length > max_length:
                    max_length = current_length
                    max_start_index = i - max_length + 1
            else:
                current_length = 1
        return lst[max_start_index:max_start_index + max_length]
    except:
        return []


def __extract_numeric_and_string_parts(element):
    numeric_part = ''
    string_part = ''
    element= str(element)
    for c in element:
        if c.isdigit():
            numeric_part += c
        else:
            string_part += c
    if numeric_part:
        numeric_part = int(numeric_part)
    return numeric_part, string_part


def __morse_to_alpha(morse_seq):
    try:
        morse_dict = {'.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D',
                      '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H',
                      '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
                      '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P',
                      '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
                      '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
                      '-.--': 'Y', '--..': 'Z', '-----': '0', '.----': '1',
                      '..---': '2', '...--': '3', '....-': '4', '.....': '5',
                      '-....': '6', '--...': '7', '---..': '8', '----.': '9'}

        # words = morse_seq.strip().split(' / ')
        alpha_seq = []
        for word in morse_seq:
            chars = word.split(' ')
            alpha_word = ''
            for char in chars:
                if char in morse_dict:
                    alpha_word += morse_dict[char]
            alpha_seq.append(alpha_word)
        return alpha_seq
    except:
        return morse_seq


def __is_morse(message):
    try:
        message = "".join(message)
        allowed = {".", "-", " "}
        return allowed.issuperset(message)
    except:
        return False


def increasing_sequence(seq):
    try:
        if __is_morse():
            seq = __morse_to_alpha(seq)
    except:
        pass
    return __increasing_sequence_alphanumeric(seq)

